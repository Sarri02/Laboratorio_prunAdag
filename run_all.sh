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

variants=$(python3 -c "from config import PRUNADAG_VARIANTS; print(' '.join(PRUNADAG_VARIANTS))")
seeds=$(python3 -c "from config import PRUNADAG_SEEDS; print(' '.join(map(str, PRUNADAG_SEEDS)))")

echo ""
echo "1. Avvio TRAINING E PRUNING"
echo "=========================================="

for variant in $variants; do
    for seed in $seeds; do
        echo ""
        echo "Eseguendo esperimenti con variante PrunAdag ${variant} e seed ${seed}..."
        python main.py "${variant}" "${seed}"
    done
done

echo ""
echo "=========================================="
echo "✓ PIPELINE COMPLETATO CON SUCCESSO"
echo "=========================================="