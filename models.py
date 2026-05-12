from typing import Iterable, List
import torch
import torch.nn as nn

# MLP: Multilayer Perceptron con due hidden layer e ReLU
class MLP(nn.Module):

	def __init__(self, input_size: int = 28 * 28, hidden_sizes: Iterable[int] = (256, 128), num_classes: int = 10):
		super().__init__()
		layers: List[nn.Module] = []
		in_dim = input_size
		for h in hidden_sizes:
			layers.append(nn.Linear(in_dim, h))                     
			layers.append(nn.ReLU(inplace=True))
			in_dim = h
		layers.append(nn.Linear(in_dim, num_classes))
		self.net = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.dim() > 2:
			x = x.view(x.size(0), -1)
		return self.net(x)

# SimpleCNN: Convolutional Neural Network semplice per immagini 28x28
class SimpleCNN(nn.Module):
	def __init__(self, in_channels: int = 1, num_classes: int = 10):
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
		)
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(64 * 7 * 7, 128),
			nn.ReLU(inplace=True),
			nn.Linear(128, num_classes),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.features(x)
		x = self.classifier(x)
		return x

# Funzione per ottenere un modello dato il nome
def get_model(name: str, **kwargs) -> nn.Module:
	name = name.lower()
	if name == "mlp":
		return MLP(**kwargs)
	if name in ("cnn", "simplecnn"):
		return SimpleCNN(**kwargs)
	raise ValueError(f"Modello non riconosciuto: {name}")


__all__ = ["MLP", "SimpleCNN", "get_model"]