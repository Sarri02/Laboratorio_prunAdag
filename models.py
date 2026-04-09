import torch
import torch.nn as nn
 
# MODELLO MLPNet: un semplice Multi-Layer Perceptron con due hidden layer e dropout
class MLPNet(nn.Module):
    def __init__(self, hidden_dim1: int = 256, hidden_dim2: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                                   # immagine 28x28 -> vettore di 784
            nn.Linear(28 * 28, hidden_dim1),                # primo hidden layer
            nn.ReLU(),                                      # attivazione
            nn.Dropout(dropout),                            # dropout per regolarizzazione
            nn.Linear(hidden_dim1, hidden_dim2),            # secondo hidden layer
            nn.ReLU(),                                      # attivazione         
            nn.Dropout(dropout),                            # layer di output con 10 classi
            nn.Linear(hidden_dim2, 10),                     # output layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# MODELLO SimpleCNN: una semplice Convolutional Neural Network con due blocchi convoluzionali e dropout
class SimpleCNN(nn.Module):
    def __init__(self, dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),     # primo blocco convoluzionale
            nn.ReLU(),                                      # attivazione
            nn.MaxPool2d(2),                                # riduzione dimensioni (28x28 -> 14x14)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),    # secondo blocco convoluzionale
            nn.ReLU(),                                      # attivazione
            nn.MaxPool2d(2),                                # riduzione dimensioni (14x14 -> 7x7)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                   # 64 canali x 7x7 -> vettore di 3136    
            nn.Linear(64 * 7 * 7, 128),                     # fully connected layer
            nn.ReLU(),                                      # attivazione
            nn.Dropout(dropout),                            # layer di output con 10 classi             
            nn.Linear(128, 10),                             # output layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


# Factory del modello in base al nome specificato
def build_model(name: str) -> nn.Module:
    model_name = name.lower()
    if model_name == "mlp":
        return MLPNet()
    if model_name == "cnn":
        return SimpleCNN()
    raise ValueError(f"Modello sconosciuto: {name}")
