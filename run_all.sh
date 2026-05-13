#!/bin/bash
# Uso: bash run_all.sh

set -e  # Esce al primo errore

echo "=========================================="
echo "LABORATORIO PRUNADAG - PIPELINE COMPLETO"
echo "=========================================="

# Verifica che la venv sia stata attivata
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Attivando la virtual environment..."
    source .venv-1/bin/activate
fi

echo ""
echo "1. Avvio TRAINING E PRUNING"
echo "=========================================="
python main.py

echo ""
echo "2. Generazione GRAFICI E TABELLE"
echo "=========================================="
python plots.py

echo ""
echo "=========================================="
echo "✓ PIPELINE COMPLETATO CON SUCCESSO"
echo "=========================================="
echo ""
echo "Risultati salvati in:"
echo "  - JSON risultati: results/"
echo "  - Grafici PDF: results/figures/"
echo "  - Tabella riassuntiva: results/figures/summary_table.txt"
echo ""
