from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "results" / "Confronto_Versioni_PrunAdag.csv"
OUTPUT_DIR = BASE_DIR / "Grafici" / "Confronto_Versioni_PrunAdag"

# Color mapping for versions: v1 (orange), v2 (red), v3 (green), v4 (blue)
COLOR_MAP = {
	"v1": "#F58518",
	"v2": "#D62728",
	"v3": "#2CA02C",
	"v4": "#1F77B4",
}


def load_data() -> pd.DataFrame:
	"""Load and validate the comparison CSV."""
	if not CSV_PATH.exists():
		raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

	data = pd.read_csv(CSV_PATH)

	required_columns = {
		"optimizer_name",
		"dataset_name",
		"model_name",
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
		"test_accuracy",
		"test_acc_after_pruning_50%",
		"test_acc_after_pruning_20%",
		"test_acc_after_pruning_10%",
		"execution_time",
	}

	missing = required_columns - set(data.columns)
	if missing:
		raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

	return data


def add_labels(data: pd.DataFrame) -> pd.DataFrame:
	labeled = data.copy()
	labeled["experiment"] = labeled["dataset_name"] + " | " + labeled["model_name"]
	labeled["version"] = labeled["optimizer_name"].str.replace("PrunAdag_", "", regex=False)
	return labeled


def save_figure(fig: plt.Figure, filename: str) -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	# Force PDF output regardless of requested filename extension
	out_path = OUTPUT_DIR / Path(filename).with_suffix('.pdf').name
	fig.savefig(out_path, dpi=200, bbox_inches="tight")
	plt.close(fig)


def plot_execution_time(data: pd.DataFrame) -> None:
	# Filter to only FashionMNIST/CNN and MNIST/MLP as requested
	mask = (
		((data["dataset_name"] == "FashionMNIST") & (data["model_name"] == "CNN"))
		| ((data["dataset_name"] == "MNIST") & (data["model_name"] == "MLP"))
	)
	filtered = data[mask]
	if filtered.empty:
		raise ValueError("No rows match the requested dataset/model pairs for execution time plot.")

	summary = filtered.groupby(["experiment", "version"], as_index=False)["execution_time"].mean()
	summary = summary.sort_values(["experiment", "version"])

	experiments = summary["experiment"].unique().tolist()
	versions = summary["version"].unique().tolist()
	width = 0.22
	x = range(len(experiments))

	fig, ax = plt.subplots(figsize=(11, 6))
	colors = [COLOR_MAP.get(v, "#7f7f7f") for v in versions]
	for idx, version in enumerate(versions):
		values = [
			summary.loc[(summary["experiment"] == experiment) & (summary["version"] == version), "execution_time"].iloc[0]
			for experiment in experiments
		]
		positions = [pos + (idx - (len(versions) - 1) / 2) * width for pos in x]
		ax.bar(positions, values, width=width, label=version, color=colors[idx])

	ax.set_xticks(list(x))
	ax.set_xticklabels(experiments, rotation=20, ha="right")
	ax.set_ylabel("Average execution time (s)")
	ax.set_title("Comparison of average execution times")
	ax.legend(title="Version")
	ax.grid(axis="y", alpha=0.3)

	save_figure(fig, "01_execution_time.png")


def plot_pruning_accuracy(data: pd.DataFrame) -> None:
	metrics = [
		("test_accuracy", "Baseline"),
		("test_acc_after_pruning_50%", "Pruning 50%"),
		("test_acc_after_pruning_20%", "Pruning 20%"),
		("test_acc_after_pruning_10%", "Pruning 10%"),
	]

	metric_names = [metric for metric, _ in metrics]
	summary = data.groupby("version", as_index=False)[metric_names].mean()
	summary = summary.sort_values("version")

	versions = summary["version"].unique().tolist()
	width = 0.18
	x = range(len(metrics))

	fig, ax = plt.subplots(figsize=(12, 6))
	colors = [COLOR_MAP.get(v, "#7f7f7f") for v in versions]
	for version_index, version in enumerate(versions):
		values = summary.loc[summary["version"] == version, metric_names].iloc[0].tolist()
		offset = (version_index - (len(versions) - 1) / 2) * width
		ax.bar([pos + offset for pos in x], values, width=width, label=version, color=colors[version_index])

	ax.set_xticks(list(x))
	ax.set_xticklabels([label for _, label in metrics])
	ax.set_ylabel("Accuracy")
	ax.set_title("Average accuracy per version before and after pruning")
	ax.set_ylim(0.90, 0.92)
	ax.grid(axis="y", alpha=0.3)
	ax.legend(title="Version")

	save_figure(fig, "02_pruning_accuracy.png")


def plot_training_curve(data: pd.DataFrame) -> None:
	epoch_columns = [f"train_acc_ep{epoch}" for epoch in range(1, 11)]
	grouped = data.groupby("version", as_index=False)[epoch_columns].mean()
	grouped = grouped.sort_values("version")

	fig, ax = plt.subplots(figsize=(10, 6))
	epochs = list(range(1, 11))

	for _, row in grouped.iterrows():
		label = row["version"]
		color = COLOR_MAP.get(label, None)
		ax.plot(epochs, [row[column] for column in epoch_columns], marker="o", linewidth=2, label=label, color=color)

	ax.set_xlabel("Epoch")
	ax.set_ylabel("Average train accuracy")
	ax.set_title("Average training curves per version")
	ax.set_xticks(epochs)
	ax.set_ylim(0.83, 0.93)
	ax.grid(alpha=0.3)
	ax.legend(title="Version")

	save_figure(fig, "03_training_curve.png")


def main() -> None:
	data = add_labels(load_data())

	# Keep only the dataset/model pairs requested in the template
	mask = (
		((data["dataset_name"] == "MNIST") & (data["model_name"] == "MLP"))
		| ((data["dataset_name"] == "FashionMNIST") & (data["model_name"] == "CNN"))
	)
	data = data[mask]

	plot_execution_time(data)
	plot_pruning_accuracy(data)
	plot_training_curve(data)
	print(f"Saved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
	main()
