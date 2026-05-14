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

### Orchestrazione e Visualizzazione

- **`main.py`**:
  - Carica MNIST e FashionMNIST automaticamente
  - Esegue griglia di esperimenti: (Adam, PrunAdag) × (MLP, CNN) × (MNIST, FashionMNIST)
  - Salva JSON con metriche
- **`plots.py`**:
  - Legge JSON da `results/`
  - Genera grafici: curve di loss, confronto accuracy, confronto pre/post pruning
  - Stampa tabella riassuntiva

### Automation

- **`run_all.sh`**: Script bash che esegue training e generazione grafici in sequenza

## Come Usare

### Prerequisiti

```bash
# Crea una virtual environment (opzionale, se non già fatta)
python3 -m venv .venv-1
source .venv-1/bin/activate

# Installa dipendenze
pip install torch torchvision matplotlib numpy
```

### Esecuzione

```bash
# Attiva la virtual environment (se non già attiva)
source .venv-1/bin/activate

# Esegui tutto: training + grafici
bash run_all.sh
```

Questo lancerà:

1. Training su MNIST e FashionMNIST con Adam e PrunAdag
2. Valutazione post-pruning a 10%, 20%, 50% survival ratio
3. Generazione di grafici PDF e tabella riassuntiva

## Output

### Risultati Numerici

- `results/*.json`: File JSON per ogni esperimento con metriche dettagliate
- `results/figures/summary_table.txt`: Tabella riassuntiva in testo

### Grafici (PDF)

- `results/figures/loss_*.pdf`: Curve di loss durante il training
- `results/figures/test_accuracy_*.pdf`: Confronto accuracy test tra optimizer
- `results/figures/pruning_*.pdf`: Impatto del pruning (pre/post) per ogni survival ratio

## Configurazione

Modifica i parametri in `config.py`:

```python
BATCH_SIZE = 128           # Dimensione mini-batch
NUM_EPOCHS = 20            # Epoche di training
LR_ADAM = 1e-3             # Learning rate per Adam
LR_PRUNADAG = 1e-2         # Learning rate per PrunAdag
TOP_K_RATIO = 0.1          # Percentuale parametri rilevanti
PRUNING_RATIOS = [0.1, 0.2, 0.5]  # Test a 10%, 20%, 50% survival
SEED = 42                  # Seed per riproducibilità
```

## Dataset

I dataset MNIST e FashionMNIST vengono **scaricati automaticamente** da `torchvision` la prima volta che vengono usati. Il download avviene in `datasets/`.

Non è necessario pre-scaricare nulla: `main.py` lo fa automaticamente.

## Note Tecniche

- **Optimizer**: PrunAdag con 4 varianti (v1, v2, v3, v4) configurabili in `train.py`
- **Modelli**: MLP (2 hidden layer da 256 e 128 neuroni) e CNN semplice
- **Device**: Automaticamente GPU se disponibile, altrimenti CPU
- **Riproducibilità**: Seed fisso per tutti i random number generator (torch, numpy, python)

## TODO

1. [ ]  Controllare sparsità sempre nulla.
2. [ ]  Aggiungere CSV.
3. [ ]  Implementare almeno 10 iterazioni per fare una media.
4. [ ]  Dividere i risultati in cartelle col nome del seed.
5. [ ]  Possibile aggiunta di altre versioni V2 V3 V4.
6. [ ]  Possibile aggiunta di altri modelli o datasets.
