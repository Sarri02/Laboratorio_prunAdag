import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# percorsi
input_dir = 'outputs'
output_dir = 'grafici'

os.makedirs(output_dir, exist_ok=True)

# Trova tutti i file csv dei risultati
csv_files = glob.glob(os.path.join(input_dir, 'results_*.csv'))

if not csv_files:
    raise FileNotFoundError(f"Nessun file CSV trovato in {input_dir}/")

df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

optimizers = ['adam', 'prunadag']
colors = {'adam': 'blue', 'prunadag': 'red'}
markers = {'adam': 'o', 'prunadag': 's'}

# Raggruppa per dataset e modello per generare grafici dedicati
for (dataset, model), group_df in df.groupby(['dataset', 'model']):
    plt.figure(figsize=(10, 6))
    
    for opt in optimizers:
        opt_data = group_df[group_df['optimizer'] == opt]
        
        # Se ci sono più seed, aggrega calcolando media e deviazione standard
        agg_data = opt_data.groupby('keep_ratio')['test_accuracy'].agg(['mean', 'std']).reset_index()
        agg_data = agg_data.sort_values('keep_ratio')
        
        # Plot della media
        plt.plot(agg_data['keep_ratio'], agg_data['mean'], 
                 marker=markers[opt], label=opt.capitalize(), color=colors[opt])
        
        # Plot della deviazione standard (se presente più di un seed)
        if agg_data['std'].notna().any():
            plt.fill_between(agg_data['keep_ratio'], 
                             agg_data['mean'] - agg_data['std'], 
                             agg_data['mean'] + agg_data['std'], 
                             color=colors[opt], alpha=0.2)

    # Titoli ed etichette
    plt.title(f"Accuracy vs Keep Ratio\nDataset: {dataset} | Model: {model}")
    plt.xlabel('Keep Ratio')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Opzione per invertire l'asse x (da 1.0 a scendere)
    # plt.gca().invert_xaxis()
  
    out_filename = f'accuracy_{dataset}_{model}.png'
    plt.savefig(os.path.join(output_dir, out_filename), dpi=300)
    plt.close() 