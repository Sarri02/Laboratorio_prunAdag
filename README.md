# Laboratorio PrunAdag: Pruning-Aware Training di Reti Neurali

Implementazione dell'algoritmo **PrunAdag** (Progressive Pruning-Aware Adagrad) per il training ottimizzato di reti neurali con riduzione post-training.

## Struttura del Progetto

```
Laboratorio_prunAdag/
├── prunadag.py                 # Optimizer PrunAdag (implementazione algoritmo)
├── config.py                   # Configurazione e costanti globali
├── models.py                   # Architetture: MLP e SimpleCNN
├── train.py                    # Training loop, valutazione, pruning
├── main.py                     # Orchestrazione esperimenti
├── plots.py                    # Generazione grafici e tabelle
├── run_all.sh                  # Script bash per eseguire tutto
├── datasets/                   # Dataset MNIST e FashionMNIST (auto-download)
├── results/                    # Output: JSON risultati e grafici PDF
└── documentazione/             # Articolo e template del progetto
```

## Cosa Fa Ogni File

### Core Optimizer

- **`prunadag.py`**: Implementazione della classe `PrunAdag`, un optimizer adattivo che classifica i parametri in "ottimizzabili" e "decrementabili" per favorire la sparsità

### Configurazione e Utility

- **`config.py`**: Costanti (learning rate, batch size, epoche), dataclass `ExperimentConfig`, funzioni `set_seed()` e `ensure_dirs()`
- **`models.py`**: MLP (fully-connected) e SimpleCNN con factory `get_model()`

### Training e Valutazione

- **`train.py`**:
  - Loop di training (`train_epoch()`)
  - Valutazione (`evaluate()`)
  - Pruning per magnitude globale (`global_magnitude_prune()`)
  - Classe `ExperimentMetrics` per salvare risultati in JSON
  - Funzione `train_full_experiment()` che orchestra tutto

### Orchestrazione

- **`main.py`**:

  - Carica MNIST e FashionMNIST automaticamente
  - Esegue griglia di esperimenti: (Adam, PrunAdag) × (MLP, CNN) × (MNIST, FashionMNIST)
  - Salva JSON con metriche

### Automation

- **`run_all.sh`**: Script bash che esegue training e generazione grafici in sequenza

## Output

### Risultati Numerici

- `results/*.json`: File JSON per ogni esperimento con metriche dettagliate

## Configurazione

Modifica i parametri in `config.py`:

```python
DATASETS = ["MNIST", "FashionMNIST"]       	# Dataset da eseguire
BATCH_SIZE = 128                            	# Dimensione del batch per il training
NUM_EPOCHS = 20                            	# Numero di epoche per il training  
LR_ADAM = 1e-3                              	# Learning rate per Adam
LR_PRUNADAG = 1e-2                          	# Learning rate per PrunAdag
TOP_K_RATIO = 0.1                           	# Percentuale di parametri rilevanti da selezionare (R_k) per PrunAdag
PRUNING_RATIOS = [0.1, 0.2, 0.5]            	# Percentuali di parametri da prunare (10%, 20%, 50%)
PRUNADAG_VARIANTS = ["v1"]  			# Versioni di PrunAdag 
PRUNADAG_SEEDS = [42]               		# Seeds da provare
OPTIMIZERS = ["Adam", "PrunAdag"]           	# Lista di ottimizzatori da testare
MODELS = ["MLP", "CNN"]				# Lista di modelli da testare

```
