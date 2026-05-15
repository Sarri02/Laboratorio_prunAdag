# Laboratorio PrunAdag — Pruning-aware training

Implementazione sperimentale di PrunAdag (Progressive Pruning-Aware Adagrad) per l'addestramento di reti neurali con pruning progressivo.

## Avvio rapido

- Eseguire l'intera pipeline:

```bash
./run_all.sh
```

- Eseguire l'orchestratore (esperimenti configurati in `config.py`):

```bash
python main.py
```

## Struttura (file principali)

```
prunadag.py        # Implementazione dell'optimizer PrunAdag
config.py          # Parametri dell'esperimento e funzioni di utilità
models.py          # Architetture: MLP e SimpleCNN
train.py           # Loop di training, valutazione, pruning
main.py            # Orchestrazione degli esperimenti
csv_logger.py      # Logging dei risultati in CSV/JSON
run_all.sh         # Script per eseguire pipeline completa
datasets/          # Raw MNIST e FashionMNIST
results/           # Output esperimenti (JSON, grafici)
documentazione/    # Articolo e template
```

## Descrizione sintetica

- `prunadag.py`: optimizer che promuove sparsità adattando aggiornamenti e pruning.
- `train.py`: funzioni di training/valutazione e routine di pruning magnitude-based.
- `main.py`: esegue la griglia di esperimenti e salva i risultati in `results/`.
- `config.py`: cambia qui batch size, epoche, ottimizzatori, seeds.

## Output

- I risultati per esperimento sono in `results/` (JSON e grafici). Le versioni sperimentali sono raggruppate in sottocartelle `v1/`, `v2/`, ...
