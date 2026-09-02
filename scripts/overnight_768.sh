#!/usr/bin/env bash
# Overnight: wait for the current run, evaluate it, then train at 768 px.
#
# 768 px needs 5.42 GB peak at batch 4 -- more than has been free tonight.
# Rather than gamble (an earlier over-allocation exhausted system memory and
# forced applications to be killed), this picks the configuration that fits the
# RAM actually available:
#
#   >= 7.5 GB free  ->  batch 4            5.42 GB peak
#   >= 5.0 GB free  ->  batch 2, accum 2   3.25 GB peak, identical effective batch
#   otherwise       ->  abort with a message rather than swap the machine
#
# Batch 2 + accum 2 is exactly equivalent to batch 4 here because BatchNorm is
# frozen, so its statistics do not depend on micro-batch size.
set -uo pipefail
cd "/Users/adarsh/Desktop/Projects/Diabetic Retinopathy Detection"

free_gb() {
    vm_stat | awk '/page size of/{p=$8} /Pages free/{f=$3} /Pages inactive/{i=$3} \
        END{gsub(/\./,"",f); gsub(/\./,"",i); printf "%.1f", (f+i)*p/1073741824}'
}

echo "[1/3] waiting for the current training run to finish..."
for _ in $(seq 1 240); do
    pgrep -f "scripts/train.py" >/dev/null || break
    sleep 60
done
if pgrep -f "scripts/train.py" >/dev/null; then
    echo "ABORT: a training run is still active after 4 hours"; exit 1
fi
echo "  clear"

echo
echo "[2/3] evaluating the macro-recall run..."
CKPT=models/checkpoints/sweep_corn_macro_fold0/best.ckpt
if [ -f "$CKPT" ]; then
    ./.venv/bin/python scripts/evaluate.py --checkpoint "$CKPT" --size 512 --loss corn \
        --bootstrap 2000 --out runs/eval_corn_macro.json 2>&1 | tail -22
else
    echo "  no checkpoint at $CKPT -- skipping"
fi

echo
echo "[3/3] 768 px run"
sleep 30                      # let memory settle after the previous process exits
FREE=$(free_gb)
echo "  free RAM: ${FREE} GB"

if (( $(echo "$FREE >= 7.5" | bc) )); then
    BATCH=4; ACCUM=1; echo "  -> batch 4 (5.42 GB peak)"
elif (( $(echo "$FREE >= 5.0" | bc) )); then
    BATCH=2; ACCUM=2; echo "  -> batch 2 x accum 2 (3.25 GB peak, effective batch 4)"
else
    cat <<MSG
  ABORT: only ${FREE} GB free; 768 px needs 3.25 GB even at batch 2, and running
  it this tight is what exhausted system memory before. Close some applications
  and re-run:
      bash scripts/overnight_768.sh
MSG
    exit 1
fi

./.venv/bin/python scripts/train.py \
    --size 768 --manifest data/manifests/aptos_768.csv \
    --batch-size "$BATCH" --accum "$ACCUM" \
    --epochs 40 --workers 2 --folds 0 --lr 1e-4 --grad-clip 1.0 --loss ce \
    --run-name sweep_768_ce 2>&1

echo
echo "evaluating 768..."
./.venv/bin/python scripts/evaluate.py \
    --checkpoint models/checkpoints/sweep_768_ce_fold0/best.ckpt \
    --size 768 --manifest data/manifests/aptos_768.csv --loss ce \
    --bootstrap 2000 --out runs/eval_768.json 2>&1 | tail -24

echo
echo "paired comparison against the 512 baseline..."
./.venv/bin/python scripts/compare.py runs/eval_512.json runs/eval_768.json \
    --label-a "512px" --label-b "768px" 2>&1 | tail -22
echo "OVERNIGHT_768_DONE"
