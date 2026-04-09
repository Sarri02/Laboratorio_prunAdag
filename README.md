## Struttura del Progetto

```
#TODO: capire significato delle sigle (CLI, MLP, CNN, MNIST...)
.
├── main.py                      # Orchestratore esperimenti + CLI 
├── prunadag.py                  # Ottimizzatore PrunAdag (core)
├── prunadag_optimizer.py        # Factory per optimizers (Adam/PrunAdag)
├── models.py                    # Definizioni modelli (MLP, CNN)
├── data_utils.py                # Caricamento dataset (MNIST, FashionMNIST)
├── train_eval.py                # Loop training e valutazione
├── pruning_utils.py             # Algoritmi pruning post-addestramento
├── outputs/                     # CSV e grafici di risultati
└── README.md
```

## Moduli

### `prunadag.py`
**Ottimizzatore PrunAdag** - Implementazione completa dell'Algoritmo
- 4 varianti: v1, v2, v3, v4 (diversi tipi di bound inferiore)
- Stato: contatori di passo separati per parametri ottimizzabili/decrescenti
- **Parametri principali:**
  - `lr`: learning rate (default 1e-2)
  - `top_k_ratio`: percentuale pesi top-k mantenuti (default 0.1)
  - `zeta`, `eps`: parametri numerici dell'algoritmo
  - `variant`: scelta della variante algoritmica

#TODO: cambiare nome
### `prunadag_optimizer.py`
**Factory** per istanziare ottimizzatori.
- `build_optimizer(name, model, cfg)` → Adam o PrunAdag configurato

### `models.py`
**Architetture neurali** per i benchmark.
- `MLPNet`: Flatten → Dense(784→256) + ReLU + Dropout → Dense(256→128) + ReLU + Dropout → Dense(128→10)
- `SimpleCNN`: Conv(1→32) → MaxPool → Conv(32→64) → MaxPool → Classifier(64×7×7 → 128 → 10)
- `build_model(name)` → istanza del modello selezionato

### `data_utils.py`
**Gestione dataset** e riproducibilità.
- Supporto: MNIST, FashionMNIST (con normalizzazione automatica)
- `set_seed(seed)`: configura seed globale
- `get_data_loaders()`: ritorna train_loader e test_loader

### `train_eval.py`
**Training e valutazione**.
- `train_model()`: loop epoca per epoca, registra loss per ogni batch
- `evaluate()`: computa loss e accuracy su set di test
- `TrainResult`: dataclass con modello, history, metriche finali

### `pruning_utils.py`
**Pruning magnitudo globale** post-addestramento.
- `compute_global_threshold(model, keep_ratio)`: percentile dei pesi per soglia
- `apply_global_magnitude_pruning()`: azzera pesi sotto soglia
- `evaluate_pruning()`: valuta modello su 3 keep_ratio (10%, 20%, 50%)

### `main.py`
**Orchestratore** esperimenti end-to-end.
- CLI arguments: `--dataset`, `--model`, `--epochs`, `--seed`, `--variant`, ecc.
- Training: Adam + PrunAdag paralleli
- Pruning: valutazione post-addestramento
- Export: CSV risultati + CSV history loss + grafico loss
- Nominazione file: `results_{dataset}_{model}_seed{seed}_ep{epochs}_var{variant}.csv`

---

## Utilizzo

### Esecuzione basic: MNIST + MLP
```bash
python main.py --dataset mnist --model mlp --epochs 20 --batch-size 256
```

### FashionMNIST + CNN con seed personalizzato
```bash
python main.py --dataset fashionmnist --model cnn --epochs 20 --seed 123
```

### Con parametri PrunAdag customizzati
```bash
python main.py --dataset mnist --model mlp --epochs 20 \
  --lr-prunadag 0.01 --top-k-ratio 0.1 --variant v2
```

### Argomenti CLI

| Argomento | Default | Descrizione |
|-----------|---------|-------------|
| `--dataset` | mnist | Dataset: `mnist` o `fashionmnist` |
| `--model` | mlp | Modello: `mlp` o `cnn` |
| `--epochs` | 10 | Numero epoche allenamento |
| `--batch-size` | 128 | Dimensione minibatch |
| `--seed` | 42 | Random seed (riproducibilità) |
| `--lr-adam` | 0.001 | Learning rate Adam |
| `--lr-prunadag` | 0.01 | Learning rate PrunAdag |
| `--top-k-ratio` | 0.1 | Percentuale pesi top-k in PrunAdag |
| `--variant` | v1 | Variante PrunAdag (v1/v2/v3/v4) |
| `--num-workers` | 4 | Worker DataLoader |

---

## Output

Per ogni esperimento vengono generati in `outputs/`:

1. **`results_*.csv`** - Risultati per experiment
   - Colonne: dataset, model, epochs, seed, variant, optimizer, phase (pre/post), keep_ratio, test_loss, test_accuracy
   - Righe: 2 (pre-pruning: Adam, PrunAdag) + 6 (post-pruning: 2 ottimizzatori × 3 keep_ratio)

2. **`loss_history_*.csv`** - Loss training per epoca
   - Colonne: epoch, adam_train_loss, prunadag_train_loss
   - Righe: 1 per epoca

3. **`loss_plot_*.pdf`** - Grafico loss vs epoche (Adam vs PrunAdag)

---

## Flusso sperimentale

```
main()
├── Carica dataset (MNIST/FashionMNIST)
├── Allena con Adam
│   ├── Forward pass
│   ├── Backward pass  
│   └── Update step (2 accumulator)
├── Allena con PrunAdag
│   └── [Stesso loop, diverso optimizer]
├── Valuta accuracy pre-pruning
├── Applica pruning magnitudo (10%, 20%, 50%)
├── Valuta post-pruning
├── Esporta CSV risultati
├── Esporta CSV loss history
└── Salva grafico
```

---

## Note tecniche

- **Dispositivo**: Automatico (CUDA se disponibile, CPU altrimenti)
- **Criterio loss**: CrossEntropyLoss (classificazione)
- **Normalizzazione**: Specifici per dataset (mean/std MNIST vs FashionMNIST)
- **Pruning**: Solo pesi (non bias) nella ricerca del threshold globale

---

## Referenze

- Articolo: `Documentazione/Articolo.txt`
- Esperimenti: vedi CSV in `outputs/` 

