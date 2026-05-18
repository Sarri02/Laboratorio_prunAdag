# Laboratorio PrunAdag — Pruning-aware training

Implementazione sperimentale di PrunAdag (Progressive Pruning-Aware Adagrad) per l'addestramento di reti neurali con pruning progressivo su MNIST / FashionMNIST.

## Avvio rapido

- Eseguire l'intera pipeline (crea output in `results/`):

```bash
./run_all.sh
```

- Eseguire l'orchestratore (esperimenti configurati in `config.py`; opzionali: variante e seed):

```bash
python main.py [prunadag_variant] [seed]
```

Esempio: `python main.py v1 42`

## Struttura (file principali)

```
prunadag.py        # Implementazione dell'optimizer PrunAdag (varianti v1..v4)
config.py          # Parametri dell'esperimento e helper (`set_seed`, `ensure_dirs`)
models.py          # Architetture: MLP e SimpleCNN + `get_model()`
train.py           # Loop di training, valutazione, pruning, salvataggio metriche
main.py            # Orchestrazione esperimenti: combina dataset/modelli/ottimizzatori
csv_logger.py      # Logging dei risultati in CSV/JSON
run_all.sh         # Script per eseguire pipeline completa
datasets/          # Raw MNIST e FashionMNIST (scaricati automaticamente)
results/           # Output esperimenti: `results/<variant>/seed_<n>/`
Grafici/           # Script e grafici per confronto dei risultati
documentazione/    # Articolo e template
```

## Descrizione sintetica

- `prunadag.py`: optimizer che promuove sparsità adattando aggiornamenti e strategie di decremento.
- `train.py`: training loop, valutazione, pruning magnitude-based globale e salvataggio metriche (`ExperimentMetrics`).
- `main.py`: orchestration degli esperimenti; costruisce combinazioni dataset/modello/ottimizzatore e salva JSON in `results/`.
- `config.py`: parametri di default (dataset, batch size, epoche, learning rate, top-k ratios, pruning ratios, seeds).
- `models.py`: definisce `MLP` e `SimpleCNN` usati negli esperimenti.

## Output

- I risultati per esperimento sono salvati in `results/<variant>/seed_<n>/` in formato JSON; gli script per generare grafici si trovano in `Grafici/`.
 - Nomi file: `<optimizer>_<model>_<dataset>[_topk_<N>].json`.
 - Per GPU: PyTorch rileva automaticamente `cuda` se disponibile.
