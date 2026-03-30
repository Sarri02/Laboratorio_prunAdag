import torch
from torch.optim import Optimizer

class PrunAdag(Optimizer):

    def __init__(
        self,
        params,             # Lista dei parametri da ottimizzare
        lr=1e-2,            # Learning rate
        top_k_ratio=0.1,    # Percentuale di parametri rilevanti da selezionare (R_k)
        zeta=1e-2,          # Valore iniziale per w_i^O e w_i^D
        eps=1e-10,          # Piccolo valore per evitare divisioni per zero
        variant="v1",       # Variante dell'algoritmo: "v1", "v2", "v3", "v4"
    ):
        # VALIDAZIONE DEI PARAMETRI
        if lr < 0.0:
            raise ValueError(f"Learning rate non valido: {lr}")
        if not 0.0 <= top_k_ratio <= 1.0:
            raise ValueError(f"top_k_ratio non valido: {top_k_ratio}")
        if zeta <= 0.0:
            raise ValueError(f"zeta deve essere > 0, trovato: {zeta}")
        if eps <= 0.0:
            raise ValueError(f"eps deve essere > 0, trovato: {eps}")
        if variant not in {"v1", "v2", "v3", "v4"}:
            raise ValueError(f"variant non valido: {variant}. Usa uno tra v1, v2, v3, v4")

        # STEP 0 - INIZIALIZZAZIONE DELL'OPTIMIZER
        defaults = dict(
            lr=lr,
            top_k_ratio=top_k_ratio,
            zeta=zeta,
            eps=eps,
            variant=variant,
        )
        super().__init__(params, defaults)


    @staticmethod
    def _topk_mask(abs_grad, k):

        # STEP 1 - SELEZIONE DEI PARAMETRI RILEVANTI R_k
        flat = abs_grad.flatten()                                                       # Appiattisce il gradiente assoluto per facilitare l'indicizzazione                          
        if k >= flat.numel():                                                           # Se k è maggiore o uguale al numero totale di parametri, tutti sono rilevanti
            return torch.ones_like(abs_grad, dtype=torch.bool)
        topk_idx = torch.topk(flat, k=k, largest=True, sorted=False).indices            # Altrimenti, ottiene gli indici dei top-k parametri più rilevanti
        mask = torch.zeros_like(flat, dtype=torch.bool)                             
        mask[topk_idx] = True                                                                          
        return mask.view_as(abs_grad)


    @staticmethod
    # Funzione per calcolo del lower bound a_{i,k}
    def _compute_lower_bound(variant, x_abs, relevant_norm, signed_irrelevant_norm, step_num, eps):
        if variant in {"v1", "v3"}:
            scale = relevant_norm / (signed_irrelevant_norm + eps)
            return (x_abs * scale) / step_num
        return x_abs / step_num


    @torch.no_grad()
    def step(self, closure=None):
        # Se è stata fornita una closure, la eseguiamo per calcolare il loss e i gradienti        
        loss = None
        if closure is not None:                                                     
            with torch.enable_grad():
                loss = closure()

        # CICLO ESTERNO - GRUPPI DI PARAMETRI
        for group in self.param_groups:
            lr = group["lr"]
            top_k_ratio = group["top_k_ratio"]
            zeta = group["zeta"]
            eps = group["eps"]
            variant = group["variant"]

            # CICLO INTERNO - PARAMETRI DEL GRUPPO
            for p in group["params"]:
                if p.grad is None:                                      
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("PrunAdag non supporta gradienti sparsi")

                state = self.state[p]

                # STEP 0 - INIZIALIZZAZIONE DELLO STATO PER PARAMETRO
                if len(state) == 0:
                    state["step"] = 0
                    state["w_optim"] = torch.full_like(p, zeta, memory_format=torch.preserve_format)
                    state["w_decr"] = torch.full_like(p, zeta, memory_format=torch.preserve_format)

                state["step"] += 1
                step_num = state["step"]

                # STEP 1 - SELEZIONE DEI RILEVANTI R_k
                k = max(1, int(p.numel() * top_k_ratio))
                abs_grad = grad.abs()
                relevant_mask = self._topk_mask(abs_grad, k)
                irrelevant_mask = ~relevant_mask

                # STEP 3.1 - CANDIDATI ACCETTABILI A_k
                sign_match_mask = (torch.sign(grad) == torch.sign(p)) & (p != 0)
                signed_irrelevant_mask = irrelevant_mask & sign_match_mask

                relevant_norm = grad[relevant_mask].norm() if relevant_mask.any() else torch.tensor(0.0, device=p.device, dtype=p.dtype)
                signed_irrelevant_norm = p[signed_irrelevant_mask].norm() if signed_irrelevant_mask.any() else torch.tensor(0.0, device=p.device, dtype=p.dtype)

                # STEP 3.2 - DEFINIZIONE DI a_{i,k}
                x_abs = p.abs()
                lower_bound = self._compute_lower_bound(
                    variant,
                    x_abs,
                    relevant_norm,
                    signed_irrelevant_norm,
                    step_num,
                    eps,
                )

                w_optim = state["w_optim"]
                ratio = abs_grad / (w_optim + eps)

                # STEP 3.3 - CLASSIFICAZIONE DI A_k
                if variant in {"v3", "v4"}:
                    acceptable_mask = signed_irrelevant_mask & (ratio >= lower_bound) & (ratio <= x_abs)
                else:
                    acceptable_mask = signed_irrelevant_mask & (ratio >= lower_bound)

                # STEP 3.4 - COSTRUZIONE DI O_k E D_k
                optimisable_mask = relevant_mask | acceptable_mask
                decreasable_mask = ~optimisable_mask

                # STEP 2 - AGGIORNAMENTO DEI PESI w_i^O (solo parametri ottimizzabili)
                if optimisable_mask.any():                                                              
                    w_optim[optimisable_mask] = torch.sqrt( w_optim[optimisable_mask].pow(2) + grad[optimisable_mask].pow(2) )

                                                                                                                    #----------------------------------------------
                # STEP 4 - AGGIORNAMENTO DEI PARAMETRI OTTIMIZZABILI
                s_optim = torch.zeros_like(p, memory_format=torch.preserve_format)                                  # Calcolo del passo di incremento
                if optimisable_mask.any():
                    s_optim[optimisable_mask] = -grad[optimisable_mask] / (w_optim[optimisable_mask] + eps)         # Aggiornamento dei parametri ottimizzabili

                                                                                                                    #----------------------------------------------
                # STEP 5.1 - AGGIORNAMENTO DEI PESI w_i^D
                w_decr = state["w_decr"]
                if decreasable_mask.any():
                    w_decr[decreasable_mask] = torch.sqrt(
                        w_decr[decreasable_mask].pow(2) + p[decreasable_mask].pow(2)
                    )
                s_decr = torch.zeros_like(p, memory_format=torch.preserve_format)
                if decreasable_mask.any():
                                                                                                                    #----------------------------------------------
                    # STEP 5.2 - AGGIORNAMENTO DEI PARAMETRI DECREMENTABILI
                    s_l = -p[decreasable_mask] / (w_decr[decreasable_mask] + eps)                                   # Calcolo del passo di decremento
                    local_a = lower_bound[decreasable_mask]
                    mag = torch.minimum(local_a, s_l.abs())

                    local_sign_match = sign_match_mask[decreasable_mask]
                    local_p = p[decreasable_mask]

                    local_step = torch.zeros_like(local_p)
                    local_step[local_sign_match] = -torch.sign(local_p[local_sign_match]) * mag[local_sign_match]   # Aggiornamento dei parametri decrementabili
                    s_decr[decreasable_mask] = local_step
                                                                                                                    #----------------------------------------------
                # STEP 6 - COSTRUZIONE DEL NUOVO ITERATO
                p.add_(lr * (s_optim + s_decr))

        return loss
