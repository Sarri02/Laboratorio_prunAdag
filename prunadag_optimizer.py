import torch

from prunadag import PrunAdag

# Factory per la creazione degli optimizer usati negli esperimenti (Adam e PrunAdag)
def build_optimizer(name: str, model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
	"""Factory degli optimizer usati negli esperimenti.

	Parametri richiesti in cfg:
	- lr_adam
	- lr_prunadag
	- top_k_ratio
	- zeta
	- eps
	- variant
	"""
	opt_name = name.lower()

    # Se è Adam, creiamo un'istanza di torch.optim.Adam
	if opt_name == "adam":
		return torch.optim.Adam(model.parameters(), lr=cfg.lr_adam)
	
    # Se è PrunAdag, creiamo un'istanza di PrunAdag
	if opt_name == "prunadag":
		return PrunAdag(
			model.parameters(),
			lr=cfg.lr_prunadag,
			top_k_ratio=cfg.top_k_ratio,
			zeta=cfg.zeta,
			eps=cfg.eps,
			variant=cfg.variant,
		)

	raise ValueError(f"Optimizer sconosciuto: {name}")

