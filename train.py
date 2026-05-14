from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam
from prunadag import PrunAdag

# Funzione per ottenere il dispositivo (GPU se disponibile, altrimenti CPU)
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Funzione per contare il numero totale di parametri del modello
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Funzione per contare il numero di parametri non zero del modello (con tolleranza)
def count_nonzero_parameters(model: nn.Module, tol: float = 1e-7) -> int:
    return sum(int((p.abs() > tol).sum().item()) for p in model.parameters() if p.requires_grad)

# Funzione per calcolare la sparsità del modello (percentuale di zeri)
def get_sparsity(model: nn.Module) -> float:
    total = count_parameters(model)
    nonzero = count_nonzero_parameters(model)
    if total == 0:
        return 0.0
    return 1.0 - (nonzero / total)



# TRAINING LOOP

# Funzione per addestrare il modello per un'epoca e restituire loss e accuracy
def train_epoch(
    model: nn.Module,
    optimizer: Optimizer,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumula statistiche
        total_loss += loss.item() * data.size(0)
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        total += data.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy



# VALUTAZIONE

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            total += data.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy



# PRUNING

def global_magnitude_prune(model: nn.Module, survival_ratio: float) -> None:
    if not (0.0 < survival_ratio <= 1.0):
        raise ValueError("survival_ratio deve stare in (0.0, 1.0]")

    # Raccogli tutti i parametri in un vettore appiattito
    tensors = [
        p.detach().abs().flatten()
        for p in model.parameters()
        if p.requires_grad
    ]

    if not tensors:
        return

    all_weights = torch.cat(tensors)
    keep_count = max(1, int(all_weights.numel() * survival_ratio))

    # Calcola la soglia: il k-esimo valore più grande
    threshold = torch.topk(all_weights, k=keep_count, largest=True, sorted=False).values.min()

    # Applica il pruning
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                mask = param.abs() >= threshold
                param.mul_(mask)



# RISULTATI E METRICHE

@dataclass
class ExperimentMetrics:
    """Metriche raccolte durante un esperimento di training."""
    optimizer_name: str  # "Adam" o "PrunAdag"
    dataset_name: str  # "MNIST" o "FashionMNIST"
    model_name: str  # "MLP" o "CNN"
    train_loss: List[float]  # Una per epoca
    train_accuracy: List[float]  # Una per epoca
    test_loss: float  # Finale
    test_accuracy: float  # Finale
    sparsity_before_pruning: float  # Frazione di zeri prima del pruning
    pruning_ratios: List[float]  # [0.1, 0.2, 0.5]
    test_accuracy_after_pruning: Dict[str, float]  # {"10%": 0.95, "20%": 0.94, ...}
    num_parameters: int
    execution_time: float  # Tempo totale di esecuzione in secondi

    def to_dict(self) -> Dict:
        """Converte le metriche in un dizionario serializzabile."""
        return asdict(self)

    def save_json(self, filepath: Path) -> None:
        """Salva le metriche in un file JSON."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, filepath: Path) -> "ExperimentMetrics":
        """Carica le metriche da un file JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(**data)



# ESPERIMENTO COMPLETO

def train_full_experiment(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer_type: str,
    dataset_name: str,
    model_name: str,
    num_epochs: int,
    device: torch.device | None = None,
    pruning_ratios: List[float] = None,
) -> ExperimentMetrics:

    # Inizio misurazione del tempo
    start_time = time.time()

    if pruning_ratios is None:
        pruning_ratios = [0.1, 0.2, 0.5]

    if device is None:
        device = get_device()

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Istanzia l'optimizer
    if optimizer_type.lower() == "adam":
        optimizer = Adam(model.parameters(), lr=1e-3)
    elif optimizer_type.lower() == "prunadag":
        optimizer = PrunAdag(model.parameters(), lr=1e-2, top_k_ratio=0.1)
    else:
        raise ValueError(f"Optimizer non supportato: {optimizer_type}")

    # Liste per raccogliere le metriche
    train_losses = []
    train_accuracies = []

    # FASE 1: Training
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, optimizer, train_loader, criterion, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Stampa ogni 5 epoche per monitorare l'andamento
        if (epoch + 1) % 5 == 0:
            print(
                f"[{optimizer_type} {dataset_name}] Epoch {epoch + 1}/{num_epochs} - "
                f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}"
            )

    # FASE 2: Valutazione finale (prima del pruning)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    sparsity_before = get_sparsity(model)

    print(f"\n[{optimizer_type} {dataset_name}] Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    print(f"Sparsity before pruning: {sparsity_before:.4f}")

    # FASE 3: Pruning e valutazione post-pruning
    test_acc_after_pruning = {}

    for ratio in pruning_ratios:
        # Crea una copia per non alterare il modello
        model_copy = deepcopy(model)
        model_copy = model_copy.to(device)

        # Applica pruning
        global_magnitude_prune(model_copy, survival_ratio=ratio)

        # Valuta dopo pruning
        _, acc_after = evaluate(model_copy, test_loader, criterion, device)
        key = f"{int(ratio * 100)}%"
        test_acc_after_pruning[key] = acc_after

        print(f"  After {key} survival pruning: Test Acc = {acc_after:.4f}")

    # Calcola il tempo totale di esecuzione
    execution_time = time.time() - start_time

    # Costruisci e restituisci le metriche
    metrics = ExperimentMetrics(
        optimizer_name=optimizer_type,
        dataset_name=dataset_name,
        model_name=model_name,
        train_loss=train_losses,
        train_accuracy=train_accuracies,
        test_loss=test_loss,
        test_accuracy=test_acc,
        sparsity_before_pruning=sparsity_before,
        pruning_ratios=pruning_ratios,
        test_accuracy_after_pruning=test_acc_after_pruning,
        num_parameters=count_parameters(model),
        execution_time=execution_time,
    )

    return metrics