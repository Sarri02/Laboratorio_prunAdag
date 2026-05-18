from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "results" / "Confronto_TopK_PrunAdag.csv"
OUTPUT_DIR = BASE_DIR / "grafici" / "Confronto_TopK_PrunAdag"

EXPERIMENT_ORDER = [
	"MNIST | MLP",
	"FashionMNIST | CNN",
]

PRUNING_COLUMNS = [
	("test_acc_after_pruning_50%", "50%"),
	("test_acc_after_pruning_30%", "30%"),
	("test_acc_after_pruning_20%", "20%"),
	("test_acc_after_pruning_10%", "10%"),
]

PRUNING_COLOR_MAP = {
	"50%": "#4C78A8",
	"30%": "#72B7B2",
	"20%": "#54A24B",
	"10%": "#F58518",
	"5%": "#E45756",
}

# TopK color map (orange shades): 0.5 dark, 0.2 medium, 0.1 light
TOPK_COLOR_MAP = {
    0.5: "#B35400",
    0.2: "#F58518",
    0.1: "#FFD8B5",
}


def load_data() -> pd.DataFrame:
	if not CSV_PATH.exists():
		raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

	data = pd.read_csv(CSV_PATH)
	required_columns = {
		"dataset_name",
		"model_name",
		"optimizer_name",
		"top_k_ratio",
		"test_accuracy",
		"test_acc_after_pruning_50%",
		"test_acc_after_pruning_30%",
		"test_acc_after_pruning_20%",
		"test_acc_after_pruning_10%",
		"train_acc_ep1",
		"train_acc_ep2",
		"train_acc_ep3",
		"train_acc_ep4",
		"train_acc_ep5",
		"train_acc_ep6",
		"train_acc_ep7",
		"train_acc_ep8",
		"train_acc_ep9",
		"train_acc_ep10",
	}
	missing = required_columns - set(data.columns)
	if missing:
		raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

	return data


def add_labels(data: pd.DataFrame) -> pd.DataFrame:
	labeled = data.copy()
	labeled["experiment"] = labeled["dataset_name"] + " | " + labeled["model_name"]
	labeled["top_k_ratio"] = pd.to_numeric(labeled["top_k_ratio"], errors="coerce")
	return labeled


def save_figure(fig: plt.Figure, filename: str) -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	# slightly reduce width to avoid wasted horizontal space in reports
	w, h = fig.get_size_inches()
	fig.set_size_inches(w * 0.75, h)
	fig.savefig(OUTPUT_DIR / Path(filename).with_suffix(".pdf").name, bbox_inches="tight")
	plt.close(fig)


def plot_baseline_vs_topk(data: pd.DataFrame) -> None:
	# Bar chart: one panel per experiment, three bars for TopK ratios
	fig, axes = plt.subplots(1, len(EXPERIMENT_ORDER), figsize=(7 * len(EXPERIMENT_ORDER), 5), sharey=True)
	axes = np.atleast_1d(axes)

	# determine canonical order of ratios (prefer 0.1,0.2,0.5 if present)
	global_ratios = sorted(data["top_k_ratio"].dropna().unique())
	preferred = [0.1, 0.2, 0.5]
	ratios = [r for r in preferred if r in global_ratios]
	if not ratios:
		ratios = list(global_ratios)

	x = np.arange(len(ratios))
	bar_width = 0.6

	for ax, experiment in zip(axes, EXPERIMENT_ORDER):
		subset = data[data["experiment"] == experiment]
		if subset.empty:
			ax.set_axis_off()
			continue

		heights = []
		colors = []
		labels = []
		for r in ratios:
			row = subset[subset["top_k_ratio"] == r]
			if row.empty:
				heights.append(np.nan)
			else:
				heights.append(float(pd.to_numeric(row["test_accuracy"].iloc[0], errors="coerce")))
			colors.append(TOPK_COLOR_MAP.get(float(r), "#888888"))
			labels.append(f"TopK {r}")

		# Adam baseline (if present) appended as an extra bar
		adam_rows = subset[subset["optimizer_name"].astype(str).str.contains("adam", case=False, na=False)]
		adam_vals = pd.to_numeric(adam_rows["test_accuracy"], errors="coerce").dropna()
		if not adam_vals.empty:
			adam_mean = float(adam_vals.mean())
		else:
			adam_mean = np.nan
		heights.append(adam_mean)
		colors.append("#9467BD")
		labels.append("Adam")

		x_all = np.arange(len(labels))
		x_all = np.arange(len(labels))
		# replace NaNs with zeros for plotting, keep labels blank for missing bars
		heights_clean = [0.0 if np.isnan(h) else h for h in heights]
		bars = ax.bar(x_all, heights_clean, width=bar_width, color=colors, edgecolor="black")
		ax.set_xticks(x_all)
		ax.set_xticklabels(labels)
		# set consistent y-limits for readability and place numeric labels centered inside bars
		ax.set_ylim(0.55, 1.0)
		labels_text = [f"{h:.3f}" if not np.isnan(h) else "" for h in heights]
		ax.bar_label(bars, labels=labels_text, label_type='center', color='black', fontsize=9)
		ax.set_title(experiment)
		ax.set_xlabel("TopK ratio")
		ax.grid(axis="y", alpha=0.25)

	axes[0].set_ylabel("Test accuracy")
	fig.suptitle("Baseline accuracy for different TopK ratios", fontsize=14)
	save_figure(fig, "01_baseline_accuracy_vs_topk.pdf")


def plot_pruning_drop(data: pd.DataFrame) -> None:
	# plot pruning levels on x-axis and one line per top_k_ratio
	pruning_labels = [label for _, label in PRUNING_COLUMNS]
	positions = np.arange(len(pruning_labels))

	fig, axes = plt.subplots(1, len(EXPERIMENT_ORDER), figsize=(7 * len(EXPERIMENT_ORDER), 5), sharey=True)
	axes = np.atleast_1d(axes)

	for ax, experiment in zip(axes, EXPERIMENT_ORDER):
		subset = data[data["experiment"] == experiment].sort_values("top_k_ratio")
		if subset.empty:
			ax.set_axis_off()
			continue

		ratios = sorted(subset["top_k_ratio"].unique())
		if len(ratios) == 0:
			ax.set_axis_off()
			continue

		# color mapping: use orange shades — 0.2 medium, 0.1 light, 0.5 dark
		ratio_color_map = {
			0.2: "#F58518",  # medium orange
			0.1: "#FFD8B5",  # light orange
			0.5: "#B35400",  # dark orange
		}
		colors = plt.cm.tab10(np.linspace(0, 1, max(3, len(ratios))))
		for i, ratio in enumerate(ratios):
			row = subset[subset["top_k_ratio"] == ratio]
			if row.empty:
				continue
			# compute absolute accuracy per pruning level
			accs = []
			for col, _ in PRUNING_COLUMNS:
				val_pruned = pd.to_numeric(row[col].iloc[0], errors="coerce")
				accs.append(float(val_pruned))

			color = ratio_color_map.get(float(ratio), colors[i % len(colors)])
			ax.plot(
				positions,
				accs,
				marker="o",
				linewidth=2,
				label=f"TopK {ratio}",
				color=color,
			)

		# Adam line (if present) — plot absolute pruned accuracies
		adam_rows = subset[subset["optimizer_name"].astype(str).str.contains("adam", case=False, na=False)]
		if not adam_rows.empty:
			adam_accs = []
			for col, _ in PRUNING_COLUMNS:
				adam_pruned = pd.to_numeric(adam_rows[col].mean(), errors="coerce")
				adam_accs.append(float(adam_pruned))
			ax.plot(positions, adam_accs, marker="o", linewidth=2, label="Adam", color="#9467BD")

		ax.set_title(experiment)
		ax.set_xlabel("Pruning level")
		ax.set_xticks(positions)
		ax.set_xticklabels(pruning_labels)
		ax.set_ylim(0.8, 1.0)
		ax.grid(alpha=0.25)
		ax.legend(title="TopK ratio")

	axes[0].set_ylabel("Test accuracy")
	fig.suptitle("Accuracy at different pruning levels", fontsize=14)
	save_figure(fig, "02_pruning_accuracy_levels.pdf")


def plot_training_comparison(data: pd.DataFrame) -> None:
	"""Plot training accuracy across epochs comparing Adam vs PrunAdag (different TopK ratios)."""
	epoch_columns = [f"train_acc_ep{e}" for e in range(1, 11)]

	fig, axes = plt.subplots(1, len(EXPERIMENT_ORDER), figsize=(7 * len(EXPERIMENT_ORDER), 5), sharey=True)
	axes = np.atleast_1d(axes)

	for ax, experiment in zip(axes, EXPERIMENT_ORDER):
		subset = data[data["experiment"] == experiment]
		if subset.empty:
			ax.set_axis_off()
			continue

		# Adam curve (if present)
		adam_mask = subset["optimizer_name"].astype(str).str.contains("adam", case=False, na=False)
		if adam_mask.any():
			adam_mean = subset.loc[adam_mask, epoch_columns].apply(pd.to_numeric, errors="coerce").mean()
			ax.plot(range(1, 11), adam_mean.values, marker="o", linewidth=2, label="Adam", color="#9467BD")

		# PrunAdag curves per TopK
		prun_mask = subset["optimizer_name"].astype(str).str.contains("prunadag", case=False, na=False)
		prun_subset = subset[prun_mask].sort_values("top_k_ratio")
		if not prun_subset.empty:
			ratios = sorted(prun_subset["top_k_ratio"].dropna().unique())
			for ratio in ratios:
				row = prun_subset[prun_subset["top_k_ratio"] == ratio]
				if row.empty:
					continue
				mean_curve = row[epoch_columns].apply(pd.to_numeric, errors="coerce").mean()
				color = TOPK_COLOR_MAP.get(float(ratio), None)
				ax.plot(range(1, 11), mean_curve.values, marker="o", linewidth=2, label=f"PrunAdag TopK {ratio}", color=color)

		ax.set_xlabel("Epoch")
		ax.set_xticks(list(range(1, 11)))
		ax.set_title(experiment)
		ax.grid(alpha=0.25)

	axes[0].set_ylabel("Training accuracy")
	fig.suptitle("Training accuracy across epochs: Adam vs PrunAdag (TopK)", fontsize=14)
	# Put a small, separate legend on each subplot for readability
	for ax in axes:
		if not getattr(ax, "get_visible", lambda: True)():
			continue
		ax.legend(title="Optimizer", fontsize=9, title_fontsize=9, loc="lower right")
	save_figure(fig, "03_training_accuracy_comparison.pdf")


def main() -> None:
	data = add_labels(load_data())
	plot_baseline_vs_topk(data)
	plot_pruning_drop(data)
	plot_training_comparison(data)
	print(f"Saved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
	main()
