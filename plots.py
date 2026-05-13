"""Generazione di grafici e tabelle riassuntive dai risultati degli esperimenti.

Questo modulo legge i JSON salvati da `main.py` e genera:
- Curve di perdita (loss) nel tempo
- Confronto della test accuracy tra optimizer
- Confronto pre/post pruning
- Tabelle riassuntive

Commenti e nomi in italiano per coerenza col progetto.
"""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

import config
from train import ExperimentMetrics


# ============================================================================
# CARICAMENTO RISULTATI
# ============================================================================

def load_all_metrics(results_dir: Path) -> List[ExperimentMetrics]:
    """Carica tutti i file JSON da una cartella risultati.

    Parametri:
    - results_dir: cartella contenente i JSON degli esperimenti

    Restituisce:
    - Lista di ExperimentMetrics
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Directory non trovata: {results_dir}")

    metrics_list = []
    json_files = list(results_dir.glob("*.json"))

    if not json_files:
        print(f"Attenzione: nessun file JSON trovato in {results_dir}")
        return []

    print(f"Caricamento {len(json_files)} file JSON...")

    for json_file in sorted(json_files):
        try:
            metrics = ExperimentMetrics.load_json(json_file)
            metrics_list.append(metrics)
            print(f"  ✓ {json_file.name}")
        except Exception as e:
            print(f"  ✗ {json_file.name}: {e}")

    return metrics_list


# ============================================================================
# FUNZIONI UTILITY PER FILTRI
# ============================================================================

def filter_by_optimizer(metrics_list: List[ExperimentMetrics], optimizer: str) -> List[ExperimentMetrics]:
    """Filtra metriche per optimizer."""
    return [m for m in metrics_list if m.optimizer_name.lower() == optimizer.lower()]


def filter_by_dataset(metrics_list: List[ExperimentMetrics], dataset: str) -> List[ExperimentMetrics]:
    """Filtra metriche per dataset."""
    return [m for m in metrics_list if m.dataset_name.lower() == dataset.lower()]


def filter_by_model(metrics_list: List[ExperimentMetrics], model: str) -> List[ExperimentMetrics]:
    """Filtra metriche per modello."""
    return [m for m in metrics_list if m.model_name.lower() == model.lower()]


# ============================================================================
# GENERAZIONE GRAFICI
# ============================================================================

def plot_loss_curves(metrics_list: List[ExperimentMetrics], output_dir: Path) -> None:
    """Genera grafici delle curve di loss per ogni combinazione dataset+modello.

    Parametri:
    - metrics_list: lista di ExperimentMetrics
    - output_dir: dove salvare i grafici
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = set(m.dataset_name for m in metrics_list)
    models = set(m.model_name for m in metrics_list)

    for dataset in sorted(datasets):
        for model in sorted(models):
            # Filtra le metriche per dataset e modello
            filtered = [m for m in metrics_list
                       if m.dataset_name == dataset and m.model_name == model]

            if not filtered:
                continue

            # Crea il grafico
            fig, ax = plt.subplots(figsize=(10, 6))

            for metrics in filtered:
                epochs = range(1, len(metrics.train_loss) + 1)
                ax.plot(epochs, metrics.train_loss, marker='o', label=metrics.optimizer_name, alpha=0.7)

            ax.set_xlabel("Epoca")
            ax.set_ylabel("Loss (training)")
            ax.set_title(f"Curve di Loss: {dataset} + {model}")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Salva il grafico
            filename = f"loss_{dataset.lower()}_{model.lower()}.pdf"
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  ✓ Grafico salvato: {filename}")


def plot_test_accuracy(metrics_list: List[ExperimentMetrics], output_dir: Path) -> None:
    """Genera grafico di confronto della test accuracy per optimizer.

    Parametri:
    - metrics_list: lista di ExperimentMetrics
    - output_dir: dove salvare i grafici
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = set(m.dataset_name for m in metrics_list)
    models = set(m.model_name for m in metrics_list)

    for dataset in sorted(datasets):
        for model in sorted(models):
            filtered = [m for m in metrics_list
                       if m.dataset_name == dataset and m.model_name == model]

            if not filtered:
                continue

            optimizers = sorted(set(m.optimizer_name for m in filtered))
            accuracies = [next((m.test_accuracy for m in filtered
                              if m.optimizer_name == opt), 0.0)
                         for opt in optimizers]

            fig, ax = plt.subplots(figsize=(8, 6))
            bars = ax.bar(optimizers, accuracies, alpha=0.7, color=['blue', 'orange'])

            # Aggiungi i valori sopra le barre
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       f'{acc:.4f}', ha='center', va='bottom')

            ax.set_ylabel("Test Accuracy")
            ax.set_title(f"Test Accuracy (pre-pruning): {dataset} + {model}")
            ax.set_ylim([0, 1.0])
            ax.grid(True, alpha=0.3, axis='y')

            # Salva il grafico
            filename = f"test_accuracy_{dataset.lower()}_{model.lower()}.pdf"
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  ✓ Grafico salvato: {filename}")


def plot_pruning_comparison(metrics_list: List[ExperimentMetrics], output_dir: Path) -> None:
    """Genera grafico di confronto pre/post pruning per diversi survival ratio.

    Parametri:
    - metrics_list: lista di ExperimentMetrics
    - output_dir: dove salvare i grafici
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = set(m.dataset_name for m in metrics_list)
    models = set(m.model_name for m in metrics_list)

    for dataset in sorted(datasets):
        for model in sorted(models):
            filtered = [m for m in metrics_list
                       if m.dataset_name == dataset and m.model_name == model]

            if not filtered:
                continue

            optimizers = sorted(set(m.optimizer_name for m in filtered))

            # Prepara i dati per il confronto pre/post pruning
            fig, axes = plt.subplots(1, len(optimizers), figsize=(5 * len(optimizers), 5))
            if len(optimizers) == 1:
                axes = [axes]

            for ax, optimizer in zip(axes, optimizers):
                opt_metrics = [m for m in filtered if m.optimizer_name == optimizer]
                if not opt_metrics:
                    continue

                metrics = opt_metrics[0]
                
                # Dati pre-pruning
                categories = ["Pre-pruning"]
                values = [metrics.test_accuracy]

                # Dati post-pruning
                for key in sorted(metrics.test_accuracy_after_pruning.keys(),
                                key=lambda x: int(x.rstrip('%'))):
                    categories.append(key)
                    values.append(metrics.test_accuracy_after_pruning[key])

                # Grafico a barre
                bars = ax.bar(categories, values, alpha=0.7, color='steelblue')

                # Aggiungi i valori sopra le barre
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                           f'{val:.4f}', ha='center', va='bottom', fontsize=9)

                ax.set_ylabel("Test Accuracy")
                ax.set_title(f"{optimizer}")
                ax.set_ylim([0, 1.0])
                ax.grid(True, alpha=0.3, axis='y')
                ax.tick_params(axis='x', rotation=45)

            fig.suptitle(f"Test Accuracy pre/post pruning: {dataset} + {model}", fontsize=14)
            plt.tight_layout()

            # Salva il grafico
            filename = f"pruning_{dataset.lower()}_{model.lower()}.pdf"
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  ✓ Grafico salvato: {filename}")


# ============================================================================
# GENERAZIONE TABELLE
# ============================================================================

def generate_summary_table(metrics_list: List[ExperimentMetrics], output_dir: Path) -> None:
    """Genera una tabella riassuntiva in formato testuale.

    Parametri:
    - metrics_list: lista di ExperimentMetrics
    - output_dir: dove salvare il file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Crea il contenuto della tabella
    lines = []
    lines.append("=" * 140)
    lines.append("TABELLA RIASSUNTIVA DEGLI ESPERIMENTI")
    lines.append("=" * 140)
    lines.append("")

    # Header
    header = (f"{'Optimizer':15} | {'Dataset':15} | {'Model':10} | "
              f"{'Test Acc':10} | {'10% Prune':10} | {'20% Prune':10} | {'50% Prune':10} | "
              f"{'Params':12} | {'Sparsity':10}")
    lines.append(header)
    lines.append("-" * 140)

    # Righe dati
    for metrics in sorted(metrics_list,
                         key=lambda m: (m.dataset_name, m.model_name, m.optimizer_name)):
        acc_10 = metrics.test_accuracy_after_pruning.get("10%", 0.0)
        acc_20 = metrics.test_accuracy_after_pruning.get("20%", 0.0)
        acc_50 = metrics.test_accuracy_after_pruning.get("50%", 0.0)

        row = (f"{metrics.optimizer_name:15} | {metrics.dataset_name:15} | {metrics.model_name:10} | "
               f"{metrics.test_accuracy:10.4f} | {acc_10:10.4f} | {acc_20:10.4f} | {acc_50:10.4f} | "
               f"{metrics.num_parameters:12} | {metrics.sparsity_before_pruning:10.4f}")
        lines.append(row)

    lines.append("=" * 140)
    lines.append("")

    # Salva in file testuale
    text_path = output_dir / "summary_table.txt"
    with open(text_path, "w") as f:
        f.write("\n".join(lines))

    print(f"  ✓ Tabella salvata: {text_path}")

    # Stampa anche a console
    print("\n" + "\n".join(lines))


# ============================================================================
# ORCHESTRAZIONE PRINCIPALE
# ============================================================================

def main(results_dir: Path = None, output_dir: Path = None) -> None:
    """Entry point per la generazione di grafici e tabelle.

    Parametri:
    - results_dir: cartella con i JSON risultati (default: config.RESULTS_DIR)
    - output_dir: cartella dove salvare i grafici (default: results/figures)
    """
    if results_dir is None:
        results_dir = config.RESULTS_DIR
    if output_dir is None:
        output_dir = config.RESULTS_DIR / "figures"

    print("=" * 80)
    print("GENERAZIONE GRAFICI E TABELLE")
    print("=" * 80)

    # Carica i risultati
    print(f"\nCaricamento risultati da {results_dir}...")
    metrics_list = load_all_metrics(results_dir)

    if not metrics_list:
        print("Nessun risultato trovato. Esegui main.py prima.")
        return

    print(f"Caricati {len(metrics_list)} esperimenti.\n")

    # Genera i grafici
    print("Generazione grafici di loss...")
    plot_loss_curves(metrics_list, output_dir)

    print("\nGenerazione grafici di test accuracy...")
    plot_test_accuracy(metrics_list, output_dir)

    print("\nGenerazione grafici di confronto pruning...")
    plot_pruning_comparison(metrics_list, output_dir)

    print("\nGenerazione tabella riassuntiva...")
    generate_summary_table(metrics_list, output_dir)

    print("\n" + "=" * 80)
    print(f"Grafici salvati in: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
