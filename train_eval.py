from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
# Risultati dell'addestramento di un modello
class TrainResult:
    model: nn.Module
    train_losses: List[float]
    test_loss: float
    test_accuracy: float

# Funzione per valutazione del modello su un DataLoader (loss e accuratezza)
def evaluate(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in data_loader:                        # ciclo sui batch del DataLoader
            x = x.to(device)                            # sposta i dati sul dispositivo
            y = y.to(device)                            # sposta le etichette sul dispositivo

            logits = model(x)                           # output del modello
            loss = criterion(logits, y)                 # calcola la loss per il batch

            total_loss += loss.item() * y.size(0)       # accumula la loss totale (loss media * numero di campioni)
            preds = logits.argmax(dim=1)                # predizioni del modello (classe con probabilità più alta)
            correct += (preds == y).sum().item()        # conta il numero di predizioni corrette
            total += y.size(0)                          # conta il numero totale di campioni

    return total_loss / total, 100.0 * correct / total  # ritorna la loss media e l'accuratezza in percentuale


# Funzione per addestrare un modello e valutare i risultati su test set
def train_model(
    model: nn.Module,                                   # modello da addestrare
    optimizer: torch.optim.Optimizer,                   # ottimizzatore da utilizzare per l'addestramento
    train_loader: DataLoader,                           # DataLoader per il set di addestramento
    test_loader: DataLoader,                            # DataLoader per il set di test
    device: torch.device,                               # dispositivo su cui eseguire l'addestramento
    epochs: int,                                        # numero di epoche per l'addestramento
) -> TrainResult:
    criterion = nn.CrossEntropyLoss()                   # funzione di perdita per classificazione multi-classe
    epoch_losses: List[float] = []                      # lista per memorizzare la loss media di ogni epoca

    for epoch in range(1, epochs + 1):                  # ciclo sulle epoche di addestramento
        model.train()                                   # imposta il modello in modalità addestramento          
        running_loss = 0.0                              # variabile per accumulare la loss totale durante l'epoca
        n_samples = 0                                   # variabile per contare il numero totale di campioni processati durante l'epoca

        for x, y in train_loader:                       # ciclo sui batch del DataLoader
            x = x.to(device)                            # sposta i dati sul dispositivo
            y = y.to(device)                            # sposta le etichette sul dispositivo

            optimizer.zero_grad(set_to_none=True)       # azzera i gradienti accumulati negli step precedenti (set_to_none=True è più efficiente in memoria)
            logits = model(x)                           # output del modello
            loss = criterion(logits, y)                 # calcola la loss per il batch
            loss.backward()                             # calcola i gradienti rispetto ai parametri del modello    
            optimizer.step()                            # aggiorna i parametri del modello in base ai gradienti calcolati

            running_loss += loss.item() * y.size(0)     # accumula la loss totale (loss media * numero di campioni) per calcolare la loss media alla fine dell'epoca
            n_samples += y.size(0)                      # aggiorna il numero totale di campioni processati durante l'epoca

        avg_train_loss = running_loss / n_samples       # calcola la loss media per l'epoca e la memorizza nella lista epoch_losses
        epoch_losses.append(avg_train_loss)             # valutazione del modello sul test set alla fine di ogni epoca (loss e accuratezza)

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(
            f"[Epoch {epoch:02d}/{epochs}] "
            f"train_loss={avg_train_loss:.4f} "
            f"test_loss={test_loss:.4f} "
            f"test_acc={test_acc:.2f}%"
        )

    final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)
    return TrainResult(model=model, train_losses=epoch_losses, test_loss=final_test_loss, test_accuracy=final_test_acc)
