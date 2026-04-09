import random
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Funzione impostazione seed per riproducibilità
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Funzione per calcolo del lower bound a_{i,k} in PrunAdag 
def _dataset_config(name: str):
    dataset_name = name.lower()
    if dataset_name == "mnist":
        return datasets.MNIST, (0.1307,), (0.3081,)
    if dataset_name == "fashionmnist":
        return datasets.FashionMNIST, (0.2860,), (0.3530,)
    raise ValueError(f"Dataset non supportato: {name}")

# Funzione per creazione dei DataLoader per train e test 
def get_data_loaders(dataset: str, data_dir: str, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    dataset_cls, mean, std = _dataset_config(dataset)
    # Trasformazione standard: ToTensor + Normalizzazione
    transform = transforms.Compose(
        [
            transforms.ToTensor(),              # Converte le immagini in tensori PyTorch
            transforms.Normalize(mean, std),    # Normalizza i tensori con media e deviazione standard specificate
        ]
    )

    # Creazione dei dataset per train e test
    train_dataset = dataset_cls(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = dataset_cls(root=data_dir, train=False, download=True, transform=transform)

    # DataLoader per train e test
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader
