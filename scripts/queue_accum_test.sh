#!/usr/bin/env bash
# Wait for the running control, then run the accum-4 step-vs-epoch test.
#
# QUESTION: the epoch-3 CE collapse is locked to training progress, but "progress"
# is still ambiguous between optimiser STEPS and data-pass EPOCHS. Both prior runs
# used --accum 1, where epoch 3 == 2,928 steps, so the two coincide.
#
# At --accum 4 there are 183 optimiser steps/epoch, which separates them:
#     step-locked  (~2,928 steps) -> collapse at epoch 16
#     epoch-locked (3 data passes) -> collapse at epoch 3
#
# Warmup stays at 8 (matching the control) so warmup-end sits at epoch 8, between
# the two predictions and away from both. This run therefore differs from the
# control by exactly one flag: --accum 4.
set -uo pipefail
cd "/Users/adarsh/Desktop/Projects/Diabetic Retinopathy Detection"

# Match the EXECUTABLE, not the command text: pgrep -f matches any process whose
# command line merely mentions the script, including a monitoring shell.
training_running() {
    local pid
    for pid in $(pgrep -f "scripts/train\.py" 2>/dev/null); do
        case "$(ps -o comm= -p "$pid" 2>/dev/null)" in
            *[Pp]ython*) return 0 ;;
        esac
    done
    return 1
}

echo "[1/3] waiting for the control run to finish..."
for _ in $(seq 1 240); do
    training_running || break
    sleep 60
done
if training_running; then echo "ABORT: still training after 4 h"; exit 1; fi
echo "  clear"

echo
echo "[2/3] control-run summary"
./.venv/bin/python - <<'PY'
import csv, os
p = "runs/warmup8_accum1_lr1e-4/fold0/metrics.csv"
if not os.path.exists(p):
    print("  no control metrics found"); raise SystemExit
rows = [r for r in csv.DictReader(open(p)) if r.get("val/qwk")]
def collapsed(x):
    try:
        return (float(x["val/recall_2_Moderate"]) >= 0.99
                and float(x["val/recall_1_Mild"]) <= 0.02
                and float(x["val/recall_3_Severe"]) == 0.0)
    except (TypeError, ValueError):
        return False
sig = [i for i, x in enumerate(rows) if collapsed(x)]
best = max(rows, key=lambda r: float(r["val/qwk"]))
print(f"  {len(rows)} epochs, best qwk={float(best['val/qwk']):.4f} at epoch {best['epoch']}")
print(f"  collapse signature at epoch(s): {sig if sig else 'none'}")
PY

echo
echo "[3/3] accum-4 step-vs-epoch test (183 steps/epoch)"
echo "  epoch-locked -> collapse at epoch 3 | step-locked -> collapse at epoch 16"
FREE=$(vm_stat | awk '/page size of/{p=$8} /Pages free/{f=$3} /Pages inactive/{i=$3} \
    END{gsub(/\./,"",f); gsub(/\./,"",i); printf "%.1f", (f+i)*p/1073741824}')
echo "  free RAM: ${FREE} GB (needs ~2.3 GB; micro-batch is still 4)"
if (( $(echo "$FREE < 3.0" | bc) )); then
    echo "  ABORT: under 3 GB free"; exit 1
fi

./.venv/bin/python scripts/train.py \
    --size 512 --batch-size 4 --accum 4 --lr 1e-4 \
    --warmup-epochs 8 --epochs 40 --patience 40 \
    --grad-clip 1.0 --loss ce --monitor val/qwk \
    --folds 0 --n-splits 5 --seed 42 --workers 2 \
    --run-name accum4_warmup8_steptest 2>&1

echo
echo "=== collapse epoch in the accum-4 run ==="
./.venv/bin/python - <<'PY'
import csv, os
p = "runs/accum4_warmup8_steptest/fold0/metrics.csv"
if not os.path.exists(p):
    print("  no metrics"); raise SystemExit
rows = [r for r in csv.DictReader(open(p)) if r.get("val/qwk")]
def collapsed(x):
    try:
        return (float(x["val/recall_2_Moderate"]) >= 0.99
                and float(x["val/recall_1_Mild"]) <= 0.02
                and float(x["val/recall_3_Severe"]) == 0.0)
    except (TypeError, ValueError):
        return False
sig = [i for i, x in enumerate(rows) if collapsed(x)]
print(f"  {len(rows)} epochs; collapse signature at epoch(s): {sig if sig else 'NONE'}")
if 3 in sig and 16 not in sig:
    print("  -> EPOCH-LOCKED (data passes), not optimiser steps")
elif 16 in sig or any(14 <= s <= 18 for s in sig):
    print("  -> STEP-LOCKED (~2,928 optimiser steps)")
elif not sig:
    print("  -> NO COLLAPSE: accumulation prevents it outright")
else:
    print(f"  -> neither prediction; signature at {sig}")
for i in sorted(set(sig) | {3, 16}):
    if i < len(rows):
        r = rows[i]
        print(f"    ep {i:>2}: g1={float(r['val/recall_1_Mild']):.3f} "
              f"g2={float(r['val/recall_2_Moderate']):.3f} "
              f"g3={float(r['val/recall_3_Severe']):.3f} "
              f"qwk={float(r['val/qwk']):.4f}")
PY
echo "ACCUM4_STEPTEST_DONE"
