#TODO: controlla questo file

from pathlib import Path
from typing import List, Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import config
from models import get_model
from train import train_full_experiment, ExperimentMetrics



# PREPARAZIONE DATASET

def get_dataset_loader(
    dataset_name: str,
    batch_size: int,
    data_dir: Path,
    train: bool = True,
) -> Tuple[DataLoader, int]:
    dataset_name = dataset_name.upper()
    
    # Normalizzazione standard per immagini in scala di grigi
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if dataset_name == "MNIST":
        dataset = datasets.MNIST(
            root=str(data_dir),
            train=train,
            download=True,
            transform=transform
        )
        num_classes = 10
    elif dataset_name == "FASHIONMNIST":
        dataset = datasets.FashionMNIST(
            root=str(data_dir),
            train=train,
            download=True,
            transform=transform
        )
        num_classes = 10
    else:
        raise ValueError(f"Dataset non supportato: {dataset_name}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=0
    )

    return loader, num_classes



# ORCHESTRAZIONE DEGLI ESPERIMENTI

def run_experiments(
    cfg: config.ExperimentConfig,
    datasets_to_run: List[str] = None,
    optimizers_to_run: List[str] = None,
    models_to_run: List[str] = None,
) -> List[ExperimentMetrics]:

    if datasets_to_run is None:
        datasets_to_run = ["MNIST", "FashionMNIST"]
    if optimizers_to_run is None:
        optimizers_to_run = ["Adam", "PrunAdag"]
    if models_to_run is None:
        models_to_run = ["MLP", "CNN"]

    # Impostazione del seed per riproducibilità
    config.set_seed(cfg.seed)

    # Crea le cartelle necessarie
    config.ensure_dirs()

    results: List[ExperimentMetrics] = []
    total_experiments = len(datasets_to_run) * len(optimizers_to_run) * len(models_to_run)
    current_exp = 0

    print("=" * 80)
    print(f"AVVIO DI {total_experiments} ESPERIMENTI")
    print("=" * 80)

    for dataset_name in datasets_to_run:
        # Carica i dataset
        print(f"\nCaricamento {dataset_name}...")
        train_loader, num_classes = get_dataset_loader(
            dataset_name,
            cfg.batch_size,
            cfg.data_dir,
            train=True
        )
        test_loader, _ = get_dataset_loader(
            dataset_name,
            cfg.batch_size,
            cfg.data_dir,
            train=False
        )

        for model_name in models_to_run:
            # Istanzia il modello
            print(f"Creazione modello {model_name}...")
            model = get_model(model_name)

            for optimizer_name in optimizers_to_run:
                current_exp += 1
                print(f"\n[{current_exp}/{total_experiments}] {optimizer_name} + {model_name} + {dataset_name}")
                print("-" * 80)

                # Esegui l'esperimento
                metrics = train_full_experiment(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    optimizer_type=optimizer_name,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    num_epochs=cfg.num_epochs,
                    device=None,  # Usa il device di default da train.py
                    pruning_ratios=cfg.pruning_ratios,
                )

                results.append(metrics)

                # Salva immediatamente il risultato in JSON
                result_filename = (
                    f"{optimizer_name.lower()}_{model_name.lower()}_{dataset_name.lower()}.json"
                )
                result_path = cfg.results_dir / result_filename
                metrics.save_json(result_path)
                print(f"Risultato salvato: {result_path}")

    print("\n" + "=" * 80)
    print(f"ESPERIMENTI COMPLETATI: {len(results)}/{total_experiments}")
    print("=" * 80)

    return results



# RIEPILOGO RISULTATI

def print_summary(results: List[ExperimentMetrics]) -> None:
    print("\n" + "=" * 80)
    print("RIEPILOGO RISULTATI")
    print("=" * 80)

    for metrics in results:
        print(f"\n{metrics.optimizer_name:10} | {metrics.model_name:10} | {metrics.dataset_name:15}")
        print(f"  Test Accuracy: {metrics.test_accuracy:.4f}")
        print(f"  Sparsity (before pruning): {metrics.sparsity_before_pruning:.4f}")

        for key, acc in sorted(metrics.test_accuracy_after_pruning.items()):
            print(f"  Test Accuracy after {key} pruning: {acc:.4f}")



# MAIN

def main():
    # Crea la configurazione di default
    cfg = config.ExperimentConfig()

    print(f"Configurazione:")
    print(f"  Dataset: {cfg.dataset}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Epoche: {cfg.num_epochs}")
    print(f"  Data dir: {cfg.data_dir}")
    print(f"  Results dir: {cfg.results_dir}")
    print(f"  Pruning ratios: {cfg.pruning_ratios}")

    # Esegui gli esperimenti
    results = run_experiments(cfg)

    # Stampa il riepilogo
    print_summary(results)

    print(f"\nTutti i risultati sono stati salvati in {cfg.results_dir}")


if __name__ == "__main__":
    main()