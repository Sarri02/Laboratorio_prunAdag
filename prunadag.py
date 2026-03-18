import torch
from torch.optim import Optimizer

class PrunAdag(Optimizer):

    def __init__(self, params, lr=1e-2, top_k_ratio=0.5, weight_decay=1e-3, eps=1e-10):
        # Validazione degli iperparametri
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate non valido: {lr}")
        if not 0.0 <= top_k_ratio <= 1.0:
            raise ValueError(f"top_k_ratio non valido: {top_k_ratio}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"weight_decay non valido: {weight_decay}")
            
        # Impostazione dei valori di default per ogni gruppo di parametri
        defaults = dict(lr=lr, top_k_ratio=top_k_ratio, 
                        weight_decay=weight_decay, eps=eps)
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
            lr = group['lr']                                  # Velocità di apprendimento
            top_k_ratio = group['top_k_ratio']                # Percentuale di parametri considerati rilevanti
            weight_decay = group['weight_decay']              # Fattore di penalizzazione
            eps = group['eps']                                # Piccolo valore per stabilizzare la divisione 

            # Iterazione per ogni parametro nel gruppo
            for p in group['params']:
                # Se il gradiente è None, saltiamo questo parametro
                if p.grad is None:
                    continue
                
                grad = p.grad                                 # Gradiente del parametro                                    
                state = self.state[p]                         # Stato associato al parametro

                # Inizializzazione dello stato Adagrad
                if len(state) == 0:
                    state['step'] = 0                                                       # Contatore dei passi
                    state['sum'] = torch.zeros_like(p, memory_format=torch.preserve_format) # Somma dei quadrati dei gradienti

                state['step'] += 1                                                          # Incremento contatore dei passi    
                
                # 1. CLASSIFICAZIONE: Soglia per isolare i parametri Rilevanti (in base al gradiente)
                k = max(1, int(p.numel() * top_k_ratio))            # Numero di parametri considerati rilevanti
                abs_grad = torch.abs(grad)                          # Valori assoluti dei gradienti
                
                # Se il numero di parametri è maggiore di k, troviamo la soglia per isolare i Top-K
                if k < p.numel():
                    threshold = torch.kthvalue(abs_grad.flatten(), p.numel() - k + 1).values # Soglia
                    relevant_mask = abs_grad >= threshold                                    # Maschera parametri rilevanti    
                else:
                    # Altrimenti consideriamo tutti i parametri come rilevanti
                    relevant_mask = torch.ones_like(p, dtype=torch.bool)
                
                # 2. WEIGHT DECAY GLOBALE
                # Applichiamo l'operazione di weight decay a tutte le variabili
                if weight_decay > 0.0:
                    p.sub_(p * lr * weight_decay)

                # 3. AGGIORNAMENTO PARAMETRI RILEVANTI (Strategia Adagrad)
                # Eseguiamo l'aggiornamento della discesa del gradiente solo sul sottoinsieme rilevante
                if relevant_mask.any():
                    rel_grad = grad[relevant_mask]                                           # Gradiente dei parametri rilevanti                
                    state['sum'][relevant_mask] += rel_grad ** 2                             # Aggiornamento somma dei quadrati
                    std = state['sum'][relevant_mask].sqrt().add_(eps)                       # Radice quadrata (con stabilizzazione)     
                    p[relevant_mask] -= lr * rel_grad / std                                  # Step di ottimizzazione Adagrad

        return loss