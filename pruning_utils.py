import copy
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_eval import evaluate

#TODO: migliora commenti di questo file

# Funzione per calcolare la soglia globale per il pruning
def compute_global_threshold(model: nn.Module, keep_ratio: float) -> float:
    keep_ratio = float(max(0.0, min(1.0, keep_ratio)))
    all_weights = []
    for p in model.parameters():
        if p.requires_grad and p.ndim > 1:
            all_weights.append(p.detach().abs().flatten())

    if not all_weights:
        return 0.0

    scores = torch.cat(all_weights)
    n_total = scores.numel()
    n_keep = max(1, int(round(keep_ratio * n_total)))

    if n_keep >= n_total:
        return 0.0

    topk_vals = torch.topk(scores, k=n_keep, largest=True, sorted=False).values
    return topk_vals.min().item()

# Funzione per applicare il pruning globale
def apply_global_magnitude_pruning(model: nn.Module, keep_ratio: float) -> nn.Module:
    pruned_model = copy.deepcopy(model)
    threshold = compute_global_threshold(pruned_model, keep_ratio)

    with torch.no_grad():
        for p in pruned_model.parameters():
            if p.requires_grad and p.ndim > 1:
                mask = p.abs() >= threshold
                p.mul_(mask)

    return pruned_model

# Funzione per valutare il modello prunato su un DataLoader
def evaluate_pruning(
    trained_model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    keep_ratios: List[float],
) -> Dict[float, Dict[str, float]]:
    criterion = nn.CrossEntropyLoss()
    results: Dict[float, Dict[str, float]] = {}

    for keep_ratio in keep_ratios:
        pruned_model = apply_global_magnitude_pruning(trained_model, keep_ratio).to(device)
        test_loss, test_acc = evaluate(pruned_model, test_loader, criterion, device)
        results[keep_ratio] = {"test_loss": test_loss, "test_accuracy": test_acc}
    return results
