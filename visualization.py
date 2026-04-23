# Visualizzazione dei dati in /outputs

import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv('outputs/results_mnist_mlp_seed42_ep1_varv1.csv')
df_post = df[df['phase'] == 'post_pruning']
plt.figure(figsize=(10, 6))

# adam
adam_data = df_post[df_post['optimizer'] == 'adam']
plt.plot(adam_data['keep_ratio'], adam_data['test_accuracy'], marker='o', label='Adam', color='blue')

# prunadag
prunadag_data = df_post[df_post['optimizer'] == 'prunadag']
plt.plot(prunadag_data['keep_ratio'], prunadag_data['test_accuracy'], marker='s', label='PrunADAG', color='red')

# titoli ed etichette
plt.title('Andamento dell\'Accuracy per Adam e PrunADAG (Post-Pruning)')
plt.xlabel('Keep Ratio')
plt.ylabel('Test Accuracy (%)')
plt.legend()
plt.grid(True)

os.makedirs('grafici', exist_ok=True)


plt.savefig('grafici/accuracy_comparison.png', dpi=300)
# plt.show()
