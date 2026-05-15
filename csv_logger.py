from pathlib import Path
import csv
import json
from typing import Dict, Any


def _fmt_num(x: Any) -> str:
    try:
        return f"{float(x):.5f}"
    except Exception:
        return ""


def save_experiment_csv(metrics: Dict, csv_path: Path | str = "results/experiments.csv") -> None:
    """Salva le metriche dell'esperimento in un CSV (una riga per esperimento).

    - `metrics` deve essere un dizionario prodotto da `ExperimentMetrics.to_dict()`.
    - Il CSV viene creato con colonne per-epoca `train_loss_epN` e `train_acc_epN`, e colonne
      per ciascun pruning ratio `test_acc_after_pruning_{pct}%`.
    - I valori numerici vengono formattati con 5 cifre decimali.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine number of epochs from train_loss
    train_loss = metrics.get("train_loss", []) or []
    train_acc = metrics.get("train_accuracy", []) or []
    num_epochs = max(len(train_loss), len(train_acc))

    # Base columns
    base = [
        "optimizer_name",
        "dataset_name",
        "model_name",
        "num_parameters",
        "num_epochs",
    ]

    epoch_loss_cols = [f"train_loss_ep{e+1}" for e in range(num_epochs)]
    epoch_acc_cols = [f"train_acc_ep{e+1}" for e in range(num_epochs)]

    test_cols = [
        "test_loss",
        "test_accuracy",
        "sparsity_before_pruning",
    ]

    pruning_ratios = metrics.get("pruning_ratios", []) or []
    per_ratio_cols = []
    for r in pruning_ratios:
        try:
            pct = int(float(r) * 100)
        except Exception:
            continue
        per_ratio_cols.append(f"test_acc_after_pruning_{pct}%")

    tail = ["execution_time"]

    fieldnames = base + epoch_loss_cols + epoch_acc_cols + test_cols + per_ratio_cols + tail

    write_header = not csv_path.exists()
    if write_header:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    # Build row
    row = {k: "" for k in fieldnames}
    row["optimizer_name"] = metrics.get("optimizer_name", "")
    row["dataset_name"] = metrics.get("dataset_name", "")
    row["model_name"] = metrics.get("model_name", "")
    row["num_parameters"] = str(metrics.get("num_parameters", ""))
    row["num_epochs"] = str(num_epochs)

    for i in range(num_epochs):
        loss_col = f"train_loss_ep{i+1}"
        acc_col = f"train_acc_ep{i+1}"
        if i < len(train_loss):
            row[loss_col] = _fmt_num(train_loss[i])
        if i < len(train_acc):
            row[acc_col] = _fmt_num(train_acc[i])

    row["test_loss"] = _fmt_num(metrics.get("test_loss", ""))
    row["test_accuracy"] = _fmt_num(metrics.get("test_accuracy", ""))
    row["sparsity_before_pruning"] = _fmt_num(metrics.get("sparsity_before_pruning", ""))

    # per-ratio
    tap = metrics.get("test_accuracy_after_pruning", {}) or {}
    for r in pruning_ratios:
        try:
            pct = int(float(r) * 100)
        except Exception:
            continue
        col = f"test_acc_after_pruning_{pct}%"
        key = f"{pct}%"
        if col in row:
            row[col] = _fmt_num(tap.get(key, ""))

    row["execution_time"] = _fmt_num(metrics.get("execution_time", ""))

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)
