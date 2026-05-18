from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "results" / "Confronto_Adam_PrunAdagV1.csv"
OUTPUT_DIR = BASE_DIR / "grafici" / "Confronto_Adam_PrunAdagV1"

TRAIN_LOSS_COLUMNS = [f"train_loss_ep{epoch}" for epoch in range(1, 11)]
TRAIN_ACCURACY_COLUMNS = [f"train_acc_ep{epoch}" for epoch in range(1, 11)]

EXPERIMENT_ORDER = [
	"MNIST | MLP",
	"FashionMNIST | CNN",
]

OPTIMIZER_ORDER = ["Adam", "PrunAdag v1"]

# Keep PrunAdag v1 in the same orange used elsewhere; Adam in purple
OPTIMIZER_COLOR_MAP = {
	"PrunAdag v1": "#F58518",
	"Adam": "#9467BD",
}

PRUNING_METRICS = [
	("test_accuracy", "Baseline"),
	("test_acc_after_pruning_50%", "Pruning 50%"),
	("test_acc_after_pruning_20%", "Pruning 20%"),
	("test_acc_after_pruning_10%", "Pruning 10%"),
]


def load_data() -> pd.DataFrame:
	if not CSV_PATH.exists():
		raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

	data = pd.read_csv(CSV_PATH)
	required_columns = {
		"optimizer_name",
		"dataset_name",
		"model_name",
		"seed",
		"test_accuracy",
		"test_acc_after_pruning_50%",
		"test_acc_after_pruning_20%",
		"test_acc_after_pruning_10%",
		"execution_time",
	}
	required_columns.update(TRAIN_LOSS_COLUMNS)
	required_columns.update(TRAIN_ACCURACY_COLUMNS)

	missing = required_columns - set(data.columns)
	if missing:
		raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

	return data


def add_labels(data: pd.DataFrame) -> pd.DataFrame:
	labeled = data.copy()
	labeled["experiment"] = labeled["dataset_name"] + " | " + labeled["model_name"]
	labeled["optimizer_label"] = labeled["optimizer_name"].replace({"PrunAdag_v1": "PrunAdag v1"})
	return labeled


def save_figure(fig: plt.Figure, filename: str) -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	# Force PDF output regardless of requested filename extension
	out_path = OUTPUT_DIR / Path(filename).with_suffix('.pdf').name
	# slightly reduce width to avoid wasted horizontal space in reports
	w, h = fig.get_size_inches()
	fig.set_size_inches(w * 0.75, h)
	fig.savefig(out_path, dpi=200, bbox_inches="tight")
	plt.close(fig)


def plot_epoch_curves(data: pd.DataFrame, columns: list[str], ylabel: str, title: str, filename: str) -> None:
	epochs = np.arange(1, len(columns) + 1)
	fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

	for ax, experiment in zip(axes, EXPERIMENT_ORDER):
		subset = data[data["experiment"] == experiment]
		if subset.empty:
			ax.set_axis_off()
			continue

		for optimizer in OPTIMIZER_ORDER:
			group = subset[subset["optimizer_label"] == optimizer]
			if group.empty:
				continue

			values = group[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
			mean_values = np.nanmean(values, axis=0)
			std_values = np.nanstd(values, axis=0)

			color = OPTIMIZER_COLOR_MAP.get(optimizer, None)
			ax.plot(epochs, mean_values, marker="o", linewidth=2, label=optimizer, color=color)
			ax.fill_between(epochs, mean_values - std_values, mean_values + std_values, color=color, alpha=0.15)

		ax.set_title(experiment)
		ax.set_xticks(epochs)
		ax.grid(alpha=0.25)
		ax.legend(title="Optimizer")

	for ax in axes:
		ax.set_xlabel("Epoch")
	axes[0].set_ylabel(ylabel)
	fig.suptitle(title, fontsize=14)
	save_figure(fig, filename)


def plot_bar_metric(
	data: pd.DataFrame,
	metric: str,
	title: str,
	ylabel: str,
	filename: str,
) -> None:
	# Create a 1xN subplot layout matching number of experiments to avoid empty panels
	n = len(EXPERIMENT_ORDER)
	fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=True)
	axes = np.atleast_1d(axes)
	# compute positions so the two bars are centered and slightly closer
	n_opts = len(OPTIMIZER_ORDER)
	spacing = 0.6
	center = 0.5
	positions = center + (np.arange(n_opts) - (n_opts - 1) / 2) * spacing
	bar_width = 0.35

	for ax, experiment in zip(axes, EXPERIMENT_ORDER):
		subset = data[data["experiment"] == experiment]
		if subset.empty:
			ax.set_axis_off()
			continue

		means = []
		stds = []
		colors = [OPTIMIZER_COLOR_MAP.get(o, "#7f7f7f") for o in OPTIMIZER_ORDER]
		for optimizer in OPTIMIZER_ORDER:
			group = subset[subset["optimizer_label"] == optimizer]
			series = pd.to_numeric(group[metric], errors="coerce")
			means.append(series.mean())
			stds.append(series.std(ddof=0))

		# draw bars once after collecting means/stds for all optimizers
		bars = ax.bar(positions, means, width=bar_width, yerr=stds, capsize=4, color=colors)
		labels = [f"{m:.3f}" if not np.isnan(m) else "" for m in means]
		ax.bar_label(bars, labels=labels, padding=3, fontsize=9, label_type='center', color='black')
		ax.set_xticks(positions)
		ax.set_xticklabels(OPTIMIZER_ORDER)
		ax.set_title(experiment)
		ax.grid(axis="y", alpha=0.25)
		# zoom y-axis for test accuracy plots (0.6 - 1.0)
		ax.set_ylim(0.6, 1.0)

	axes[0].set_ylabel(ylabel)
	# adjust x limits to keep bars centered without sticking together
	for ax in axes:
		ax.set_xlim(positions.min() - 0.4, positions.max() + 0.4)
	fig.suptitle(title, fontsize=14)
	save_figure(fig, filename)


def plot_pruning_accuracy(data: pd.DataFrame) -> None:
	metric_names = [metric for metric, _ in PRUNING_METRICS]
	metric_labels = [label for _, label in PRUNING_METRICS]
	positions = np.arange(len(metric_names))
	bar_width = 0.35
	fig, axes = plt.subplots(1, len(EXPERIMENT_ORDER), figsize=(7 * len(EXPERIMENT_ORDER), 5), sharey=True)
	axes = np.atleast_1d(axes)

	for ax, experiment in zip(axes, EXPERIMENT_ORDER):
		subset = data[data["experiment"] == experiment]
		if subset.empty:
			ax.set_axis_off()
			continue

		colors = [OPTIMIZER_COLOR_MAP.get(o, "#7f7f7f") for o in OPTIMIZER_ORDER]
		for index, optimizer in enumerate(OPTIMIZER_ORDER):
			group = subset[subset["optimizer_label"] == optimizer]
			values = group[metric_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
			means = np.nanmean(values, axis=0)
			stds = np.nanstd(values, axis=0)
			offset = (index - 0.5) * bar_width
			ax.bar(positions + offset, means, width=bar_width, yerr=stds, capsize=4, label=optimizer, color=colors[index])

		ax.set_xticks(positions)
		ax.set_xticklabels(metric_labels)
		ax.set_title(experiment)
		ax.grid(axis="y", alpha=0.25)
		ax.legend(title="Optimizer")
		# zoom y-axis for pruning accuracy plots (0.6 - 1.0)
		ax.set_ylim(0.6, 1.0)

	axes[0].set_ylabel("Accuracy")
	fig.suptitle("Average accuracy before and after pruning", fontsize=14)
	save_figure(fig, "04_pruning_accuracy.png")


def plot_pruning_drop(data: pd.DataFrame) -> None:
	drop_columns = ["test_acc_after_pruning_50%", "test_acc_after_pruning_20%", "test_acc_after_pruning_10%"]
	drop_labels = ["50%", "20%", "10%"]
	positions = np.arange(len(drop_columns))
	bar_width = 0.35
	fig, axes = plt.subplots(1, len(EXPERIMENT_ORDER), figsize=(7 * len(EXPERIMENT_ORDER), 5), sharey=True)
	axes = np.atleast_1d(axes)

	for ax, experiment in zip(axes, EXPERIMENT_ORDER):
		subset = data[data["experiment"] == experiment]
		if subset.empty:
			ax.set_axis_off()
			continue

		colors = [OPTIMIZER_COLOR_MAP.get(o, "#7f7f7f") for o in OPTIMIZER_ORDER]
		for index, optimizer in enumerate(OPTIMIZER_ORDER):
			group = subset[subset["optimizer_label"] == optimizer].copy()
			baseline = pd.to_numeric(group["test_accuracy"], errors="coerce")

			drops = []
			for column in drop_columns:
				pruned = pd.to_numeric(group[column], errors="coerce")
				drops.append((baseline - pruned).to_numpy(dtype=float))

			matrix = np.vstack(drops).T
			means = np.nanmean(matrix, axis=0)
			stds = np.nanstd(matrix, axis=0)
			offset = (index - 0.5) * bar_width
			ax.bar(positions + offset, means, width=bar_width, yerr=stds, capsize=4, label=optimizer, color=colors[index])

		ax.set_xticks(positions)
		ax.set_xticklabels(drop_labels)
		ax.set_title(experiment)
		ax.grid(axis="y", alpha=0.25)
		ax.legend(title="Optimizer")

	axes[0].set_ylabel("Accuracy drop")
	fig.suptitle("Average accuracy drop caused by pruning", fontsize=14)
	save_figure(fig, "05_pruning_drop.png")


def plot_execution_time(data: pd.DataFrame) -> None:
	# compute positions so the two bars are centered and slightly closer
	n_opts = len(OPTIMIZER_ORDER)
	spacing = 0.6
	center = 0.5
	positions = center + (np.arange(n_opts) - (n_opts - 1) / 2) * spacing
	bar_width = 0.35
	fig, axes = plt.subplots(1, len(EXPERIMENT_ORDER), figsize=(7 * len(EXPERIMENT_ORDER), 5), sharey=True)
	axes = np.atleast_1d(axes)

	for ax, experiment in zip(axes, EXPERIMENT_ORDER):
		subset = data[data["experiment"] == experiment]
		if subset.empty:
			ax.set_axis_off()
			continue

		means = []
		stds = []
		colors = [OPTIMIZER_COLOR_MAP.get(o, "#7f7f7f") for o in OPTIMIZER_ORDER]
		for optimizer in OPTIMIZER_ORDER:
			group = subset[subset["optimizer_label"] == optimizer]
			series = pd.to_numeric(group["execution_time"], errors="coerce")
			means.append(series.mean())
			stds.append(series.std(ddof=0))

		# draw bars once after collecting means/stds for all optimizers
		bars = ax.bar(positions, means, width=bar_width, yerr=stds, capsize=4, color=colors)
		labels = [f"{m:.2f}" if not np.isnan(m) else "" for m in means]
		ax.bar_label(bars, labels=labels, padding=3, fontsize=9, label_type='center', color='black')
		ax.set_xticks(positions)
		ax.set_xticklabels(OPTIMIZER_ORDER)
		ax.set_title(experiment)
		ax.grid(axis="y", alpha=0.25)

	axes[0].set_ylabel("Average execution time (s)")
	fig.suptitle("Comparison of execution times", fontsize=14)
	for ax in axes:
		ax.set_xlim(positions.min() - 0.4, positions.max() + 0.4)
	save_figure(fig, "06_execution_time.png")


def main() -> None:
	data = add_labels(load_data())

	plot_epoch_curves(
		data,
		TRAIN_LOSS_COLUMNS,
		"Loss",
		"Average training loss curve",
		"01_training_loss.png",
	)
	plot_epoch_curves(
		data,
		TRAIN_ACCURACY_COLUMNS,
		"Accuracy",
		"Average training accuracy curve",
		"02_training_accuracy.png",
	)
	plot_bar_metric(
		data,
		"test_accuracy",
		"Accuracy finale sul test",
		"Test accuracy",
		"03_test_accuracy.png",
	)
	plot_pruning_accuracy(data)
	plot_pruning_drop(data)
	plot_execution_time(data)

	print(f"Saved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
	main()