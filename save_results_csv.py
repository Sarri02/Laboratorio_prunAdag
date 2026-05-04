from train_eval import TrainResult, train_model
from typing import Dict, List
import os
import csv
import importlib

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
