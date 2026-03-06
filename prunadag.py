import torch
from torch.optim import Optimizer

class PrunAdag(Optimizer):


    def __init__(self, params, lr=1e-2, top_k_ratio=0.5, weight_decay_irrelevant=1e-3, eps=1e-10):
        # Validazione degli iperparametri
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate non valido: {lr}")
        if not 0.0 <= top_k_ratio <= 1.0:
            raise ValueError(f"top_k_ratio non valido: {top_k_ratio}")
        if not 0.0 <= weight_decay_irrelevant:
            raise ValueError(f"weight_decay_irrelevant non valido: {weight_decay_irrelevant}")
        # Impostazione dei valori di default per ogni gruppo di parametri
        defaults = dict(lr=lr, top_k_ratio=top_k_ratio, 
                        weight_decay_irrelevant=weight_decay_irrelevant, eps=eps)
        super(PrunAdag, self).__init__(params, defaults)



    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        # Se è fornita una closure, la usiamo per calcolare la loss
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Iterazione per ogni gruppo di parametri
        for group in self.param_groups:
            lr = group['lr']                                        # Velocità di apprendimento
            top_k_ratio = group['top_k_ratio']                      # Percentuale di parametri considerati rilevanti
            wd_irrelevant = group['weight_decay_irrelevant']        # Fattore di penalizzazione per i parametri irrilevanti
            eps = group['eps']                                      # Piccolo valore per stabilizzare la divisione 

            # Iterazione per ogni parametro nel gruppo
            for p in group['params']:
                # Se il gradiente è None, saltiamo questo parametro
                if p.grad is None:
                    continue
                
                grad = p.grad                                       # Gradiente del parametro                                    
                state = self.state[p]                               # Stato associato al parametro

                # Inizializzazione dello stato Adagrad
                if len(state) == 0:
                    state['step'] = 0                                                               # Contatore dei passi
                    state['sum'] = torch.zeros_like(p, memory_format=torch.preserve_format)         # Somma dei quadrati dei gradienti

                state['step'] += 1                                                                  # Incremento contatore dei passi    
                
                # 1. CLASSIFICAZIONE: Soglia per separare Rilevanti da Irrilevanti
                k = max(1, int(p.numel() * top_k_ratio))            # Numero di parametri considerati rilevanti
                abs_grad = torch.abs(grad)                          # Valori assoluti dei gradienti
                
                # Se il numero di parametri è maggiore di k, troviamo la soglia per isolare i Top-K parametri
                if k < p.numel():
                    threshold = torch.kthvalue(abs_grad.flatten(), p.numel() - k + 1).values        # Soglia gradiente per i Top-K parametri
                    relevant_mask = abs_grad >= threshold                                           # Maschera booleana per i parametri rilevanti    
                else:
                    # Altrimenti consideriamo tutti i parametri come rilevanti
                    relevant_mask = torch.ones_like(p, dtype=torch.bool)                            # Maschera booleana per i parametri rilevanti (tutti True)
                
                irrelevant_mask = ~relevant_mask                                                    # Maschera booleana per i parametri irrilevanti        

                # 2. AGGIORNAMENTO PARAMETRI RILEVANTI (Strategia Adattiva)
                if relevant_mask.any():
                    rel_grad = grad[relevant_mask]                                                  # Gradiente dei parametri rilevanti                
                    state['sum'][relevant_mask] += rel_grad ** 2                                    # Aggiornamento della somma dei quadrati dei gradienti per i parametri rilevanti
                    std = state['sum'][relevant_mask].sqrt().add_(eps)                              # Calcolo della radice quadrata della somma dei quadrati dei gradienti (con stabilizzazione)     
                    p[relevant_mask] -= lr * rel_grad / std                                         # Aggiornamento dei parametri rilevanti con la regola di Adagrad
                
                # 3. AGGIORNAMENTO PARAMETRI IRRILEVANTI (Strategia di Penalizzazione)
                if irrelevant_mask.any():
                    p[irrelevant_mask] -= lr * wd_irrelevant * p[irrelevant_mask]                   # Penalizzazione dei parametri irrilevanti proporzionale al loro valore attuale

        return loss