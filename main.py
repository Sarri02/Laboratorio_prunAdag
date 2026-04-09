import argparse
import csv
import importlib
import os
from typing import Dict, List

import torch

from data_utils import get_data_loaders, set_seed
from models import build_model
from pruning_utils import evaluate_pruning
from prunadag_optimizer import build_optimizer
from train_eval import TrainResult, train_model

#TODO: fai commenti di questo file

def save_results_csv(
	output_dir: str,
	dataset: str,
	model: str,
	epochs: int,
	seed: int,
	variant: str,
	adam_result: TrainResult,
	prunadag_result: TrainResult,
	adam_pruning: Dict[float, Dict[str, float]],
	prunadag_pruning: Dict[float, Dict[str, float]],
) -> str:
	os.makedirs(output_dir, exist_ok=True)
	file_name = f"results_{dataset}_{model}_seed{seed}_ep{epochs}_var{variant}.csv"
	out_path = os.path.join(output_dir, file_name)

	rows = [
		{
			"dataset": dataset,
			"model": model,
			"epochs": epochs,
			"seed": seed,
			"variant": variant,
			"optimizer": "adam",
			"phase": "pre_pruning",
			"keep_ratio": 1.0,
			"test_loss": adam_result.test_loss,
			"test_accuracy": adam_result.test_accuracy,
		},
		{
			"dataset": dataset,
			"model": model,
			"epochs": epochs,
			"seed": seed,
			"variant": variant,
			"optimizer": "prunadag",
			"phase": "pre_pruning",
			"keep_ratio": 1.0,
			"test_loss": prunadag_result.test_loss,
			"test_accuracy": prunadag_result.test_accuracy,
		},
	]

	for keep_ratio in sorted(adam_pruning.keys()):
		rows.append(
			{
				"dataset": dataset,
				"model": model,
				"epochs": epochs,
				"seed": seed,
				"variant": variant,
				"optimizer": "adam",
				"phase": "post_pruning",
				"keep_ratio": keep_ratio,
				"test_loss": adam_pruning[keep_ratio]["test_loss"],
				"test_accuracy": adam_pruning[keep_ratio]["test_accuracy"],
			}
		)
		rows.append(
			{
				"dataset": dataset,
				"model": model,
				"epochs": epochs,
				"seed": seed,
				"variant": variant,
				"optimizer": "prunadag",
				"phase": "post_pruning",
				"keep_ratio": keep_ratio,
				"test_loss": prunadag_pruning[keep_ratio]["test_loss"],
				"test_accuracy": prunadag_pruning[keep_ratio]["test_accuracy"],
			}
		)

	fieldnames = [
		"dataset",
		"model",
		"epochs",
		"seed",
		"variant",
		"optimizer",
		"phase",
		"keep_ratio",
		"test_loss",
		"test_accuracy",
	]
	with open(out_path, "w", newline="", encoding="utf-8") as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

	return out_path


def save_loss_history_csv(
	output_dir: str,
	dataset: str,
	model: str,
	epochs: int,
	seed: int,
	variant: str,
	adam_losses: List[float],
	prunadag_losses: List[float],
) -> str:
	os.makedirs(output_dir, exist_ok=True)
	file_name = f"loss_history_{dataset}_{model}_seed{seed}_ep{epochs}_var{variant}.csv"
	out_path = os.path.join(output_dir, file_name)

	with open(out_path, "w", newline="", encoding="utf-8") as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(["epoch", "adam_train_loss", "prunadag_train_loss"])
		for epoch_idx, (adam_loss, prunadag_loss) in enumerate(zip(adam_losses, prunadag_losses), start=1):
			writer.writerow([epoch_idx, adam_loss, prunadag_loss])

	return out_path


def save_loss_plot(losses: Dict[str, List[float]], output_dir: str) -> None:
	try:
		plt = importlib.import_module("matplotlib.pyplot")
	except ImportError:
		print("[WARN] matplotlib non installato: salto il salvataggio del grafico delle loss.")
		return

	os.makedirs(output_dir, exist_ok=True)
	plt.figure(figsize=(8, 5))
	for name, curve in losses.items():
		plt.plot(range(1, len(curve) + 1), curve, label=name)
	plt.xlabel("Epoch")
	plt.ylabel("Train loss")
	plt.title("Confronto train loss: Adam vs PrunAdag")
	plt.grid(True, alpha=0.3)
	plt.legend()
	out_path = os.path.join(output_dir, "loss_comparison.png")
	plt.tight_layout()
	plt.savefig(out_path, dpi=150)
	plt.close()
	print(f"Grafico loss salvato in: {out_path}")


def print_summary(
	adam_result: TrainResult,
	prunadag_result: TrainResult,
	adam_pruning: Dict[float, Dict[str, float]],
	prunadag_pruning: Dict[float, Dict[str, float]],
) -> None:
	print("\n=== RISULTATI FINALI (PRE-PRUNING) ===")
	print(f"Adam     -> test_loss: {adam_result.test_loss:.4f}, test_acc: {adam_result.test_accuracy:.2f}%")
	print(f"PrunAdag -> test_loss: {prunadag_result.test_loss:.4f}, test_acc: {prunadag_result.test_accuracy:.2f}%")

	print("\n=== RISULTATI POST-PRUNING (surviving weights) ===")
	print("ratio_keep | Adam_acc | PrunAdag_acc | Adam_loss | PrunAdag_loss")
	for keep_ratio in sorted(adam_pruning.keys()):
		a = adam_pruning[keep_ratio]
		p = prunadag_pruning[keep_ratio]
		print(
			f"{keep_ratio:>8.2f} | "
			f"{a['test_accuracy']:>8.2f}% | "
			f"{p['test_accuracy']:>11.2f}% | "
			f"{a['test_loss']:>8.4f} | "
			f"{p['test_loss']:>12.4f}"
		)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Benchmark: Adam vs PrunAdag")
	parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashionmnist"])
	parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "cnn"])
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--batch-size", type=int, default=128)
	parser.add_argument("--num-workers", type=int, default=2)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--data-dir", type=str, default="./data")
	parser.add_argument("--output-dir", type=str, default="./outputs")

	parser.add_argument("--lr-adam", type=float, default=1e-3)
	parser.add_argument("--lr-prunadag", type=float, default=1e-2)
	parser.add_argument("--top-k-ratio", type=float, default=0.1)
	parser.add_argument("--zeta", type=float, default=1e-2)
	parser.add_argument("--eps", type=float, default=1e-10)
	parser.add_argument("--variant", type=str, default="v1", choices=["v1", "v2", "v3", "v4"])

	return parser.parse_args()


def main() -> None:
	args = parse_args()
	set_seed(args.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")

	train_loader, test_loader = get_data_loaders(
		dataset=args.dataset,
		data_dir=args.data_dir,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
	)
	print(f"Dataset: {args.dataset} | Modello: {args.model}")

	print("\n=== TRAINING ADAM ===")
	model_adam = build_model(args.model).to(device)
	optimizer_adam = build_optimizer("adam", model_adam, args)
	adam_result = train_model(
		model=model_adam,
		optimizer=optimizer_adam,
		train_loader=train_loader,
		test_loader=test_loader,
		device=device,
		epochs=args.epochs,
	)

	print("\n=== TRAINING PRUNADAG ===")
	model_prunadag = build_model(args.model).to(device)
	optimizer_prunadag = build_optimizer("prunadag", model_prunadag, args)
	prunadag_result = train_model(
		model=model_prunadag,
		optimizer=optimizer_prunadag,
		train_loader=train_loader,
		test_loader=test_loader,
		device=device,
		epochs=args.epochs,
	)

	keep_ratios = [0.10, 0.20, 0.50]
	adam_pruning = evaluate_pruning(adam_result.model, test_loader, device, keep_ratios)
	prunadag_pruning = evaluate_pruning(prunadag_result.model, test_loader, device, keep_ratios)

	save_loss_plot(
		{
			"Adam": adam_result.train_losses,
			"PrunAdag": prunadag_result.train_losses,
		},
		output_dir=args.output_dir,
	)

	print_summary(
		adam_result=adam_result,
		prunadag_result=prunadag_result,
		adam_pruning=adam_pruning,
		prunadag_pruning=prunadag_pruning,
	)

	results_csv_path = save_results_csv(
		output_dir=args.output_dir,
		dataset=args.dataset,
		model=args.model,
		epochs=args.epochs,
		seed=args.seed,
		variant=args.variant,
		adam_result=adam_result,
		prunadag_result=prunadag_result,
		adam_pruning=adam_pruning,
		prunadag_pruning=prunadag_pruning,
	)
	loss_csv_path = save_loss_history_csv(
		output_dir=args.output_dir,
		dataset=args.dataset,
		model=args.model,
		epochs=args.epochs,
		seed=args.seed,
		variant=args.variant,
		adam_losses=adam_result.train_losses,
		prunadag_losses=prunadag_result.train_losses,
	)
	print(f"CSV risultati salvato in: {results_csv_path}")
	print(f"CSV loss per epoca salvato in: {loss_csv_path}")


if __name__ == "__main__":
	main()

