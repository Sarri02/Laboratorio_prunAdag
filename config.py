from dataclasses import dataclass, field
from pathlib import Path
import random
import os
import torch


# COSTANTI DI CONFIGURAZIONE
DATASETS = ["MNIST", "FashionMNIST"]       	# Dataset da eseguire
BATCH_SIZE = 128                            # Dimensione del batch per il training
NUM_EPOCHS = 10                            	# Numero di epoche per il training    
LR_ADAM = 1e-3                              # Learning rate per Adam
LR_PRUNADAG = 1e-2                          # Learning rate per PrunAdag
TOP_K_RATIO = 0.1                           # Percentuale di parametri rilevanti per PrunAdag TODO: Provare 0.2
PRUNING_RATIOS = [0.5, 0.2, 0.1]            # Percentuali di parametri da prunare (10%, 20%, 50%)
PRUNADAG_VARIANTS = ["v1"]  				# Versioni di PrunAdag 
PRUNADAG_SEEDS = [48, 49, 50, 51, 52]       # Seeds da provare
OPTIMIZERS = ["Adam", "PrunAdag"]           # Lista di ottimizzatori da testare
MODELS = ["MLP", "CNN"]						# Lista di modelli da testare


# PERCORSI DI DEFAULT
DATA_DIR = Path("datasets")
RESULTS_DIR = Path("results")


# DATACLASS PER CONFIGURAZIONE ESPERIMENTO
@dataclass
class ExperimentConfig:
	seed: int
	datasets: list = field(default_factory=lambda: list(DATASETS))
	optimizers: list = field(default_factory=lambda: list(OPTIMIZERS))
	models: list = field(default_factory=lambda: list(MODELS))
	batch_size: int = BATCH_SIZE
	num_epochs: int = NUM_EPOCHS
	lr_adam: float = LR_ADAM
	lr_prunadag: float = LR_PRUNADAG
	top_k_ratio: float = TOP_K_RATIO
	pruning_ratios: list = field(default_factory=lambda: list(PRUNING_RATIOS))
	data_dir: Path = DATA_DIR
	results_dir: Path = RESULTS_DIR


# SETUP FUNZIONI UTILI

# Funzione per impostare i seed per la riproducibilità

def set_seed(seed: int) -> None:
	random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	try:
		import numpy as _np

		_np.random.seed(seed)
	except Exception:
		pass

	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

# Funzione per assicurarsi che le directory esistano
def ensure_dirs() -> None:
	DATA_DIR.mkdir(parents=True, exist_ok=True)
	RESULTS_DIR.mkdir(parents=True, exist_ok=True)


__all__ = [
	"ExperimentConfig",
	"set_seed",
	"ensure_dirs",
	"DATASETS",
	"OPTIMIZERS",
	"MODELS",
	"BATCH_SIZE",
	"NUM_EPOCHS",
	"LR_ADAM",
	"LR_PRUNADAG",
	"TOP_K_RATIO",
	"PRUNING_RATIOS",
	"PRUNADAG_VARIANTS",
	"PRUNADAG_SEEDS",
	"DATA_DIR",
	"RESULTS_DIR",
]