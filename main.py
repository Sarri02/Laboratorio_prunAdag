import argparse
from typing import Dict, List

import torch

from data_utils import get_data_loaders, set_seed
from models import build_model
from pruning_utils import evaluate_pruning
from prunadag_optimizer import build_optimizer
from train_eval import TrainResult, train_model
from save_results_csv import save_results_csv, save_loss_history_csv, save_loss_plot, print_summary

#TODO: fai commenti di questo file

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

