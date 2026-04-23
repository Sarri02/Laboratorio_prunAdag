# Visualizzazione dei dati in /outputs

import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv('outputs/results_mnist_mlp_seed42_ep1_varv1.csv')
plt.figure(figsize=(10, 6))

optimizers = ['adam', 'prunadag']
colors = {'adam': 'blue', 'prunadag': 'red'}
markers = {'adam': 'o', 'prunadag': 's'}

for opt in optimizers:
    opt_data = df[df['optimizer'] == opt].sort_values('keep_ratio')
    plt.plot(opt_data['keep_ratio'], opt_data['test_accuracy'], marker=markers[opt], label=opt.capitalize(), color=colors[opt])

# titoli ed etichette
plt.title('Andamento dell\'Accuracy per Adam e PrunADAG')
plt.xlabel('Keep Ratio')
plt.ylabel('Test Accuracy (%)')
plt.legend()
plt.grid(True)

os.makedirs('grafici', exist_ok=True)


plt.savefig('grafici/accuracy_comparison.png', dpi=300)
# plt.show()
