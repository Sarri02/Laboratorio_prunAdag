# Stochastic PrunAdag for Pruning-Aware Training of Neural Networks

Il **pruning delle reti neurali** è una tecnica utilizzata per ridurre le dimensioni e la complessità dei modelli di deep learning, rimuovendo i parametri che contribuiscono in misura minima alle prestazioni del modello.  

Durante l'addestramento o dopo che un modello è stato addestrato, vengono identificati ed eliminati pesi, neuroni o interi filtri con un impatto minimo sulle previsioni.

Questa operazione può essere eseguita, ad esempio, eliminando i pesi con il valore assoluto più basso. Ciò si traduce in una rete più sparsa che richiede meno memoria e calcolo, mantenendo, si spera, una precisione simile.  

Il pruning è ampiamente utilizzato per rendere le reti neurali più efficienti per l'implementazione su dispositivi con risorse limitate.

Recentemente è stata proposta una variante dell'algoritmo classico AdaGrad, denominata **PrunAdag**, che esegue passaggi di ottimizzazione tenendo conto dello scenario di pruning e cercando di produrre una rete robusta al pruning post-addestramento.  

A tal fine:

- gli aggiornamenti per discesa del gradiente vengono eseguiti solo rispetto a un sottoinsieme di variabili  
- mentre l'insieme completo delle variabili subisce un'operazione di decadimento dei pesi ad ogni passo  

Il metodo è stato tuttavia studiato e testato solo in contesti **full-batch** e con problemi convessi.  

L'analisi della variante **minibatch** sulle reti neurali profonde sarebbe particolarmente rilevante per le applicazioni.

---



## Obiettivi del Progetto

* **Implementazione:** Sviluppare un ottimizzatore in **PyTorch** che implementi l'algoritmo PrunAdag per l'utilizzo in modalità minibatch.
* **Validazione:** Testare il metodo su un set di problemi standard:
    * Rete fully connected (MLP) con due layer nascosti su dataset **MNIST**.
    * Rete convoluzionale semplice (CNN) su dataset **FashionMNIST**.
* **Analisi Comparativa:** Confrontare le prestazioni della rete addestrata tramite PruneAdag con quelle della rete ottenuta utilizzando l'ottimizzatore Adam, come segue:
    * Verificare se PruneAdag è in grado di addestrare correttamente la rete, accertandosi che la perdita sia effettivamente ridotta e che la qualità dell'adattamento sia paragonabile a quella della rete addestrata con Adam.
    * Mostrare il grafico della diminuzione della perdita nel corso delle epoche, per ciascun problema e per entrambi gli algoritmi.
    * Confrontare le prestazioni di test (in termini di accuratezza di test) dei due modelli.
    * Verificare la qualità della rete (sia in termini di perdita di addestramento che di accuratezza di test) dopo che l'operazione di pruning è stata applicata alle due reti. Il pruning dovrebbe essere effettuato impostando a zero tutti i pesi tranne quello con il valore assoluto maggiore. Il test dovrebbe essere ripetuto impostando la percentuale dei pesi sopravvissuti (cioè non pari a zero) al 10%, 20% e 50%. 

