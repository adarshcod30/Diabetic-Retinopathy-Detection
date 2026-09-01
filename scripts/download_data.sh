#!/usr/bin/env bash
# Fetch the datasets that fit on a 28 GB-free development machine.
#
# Deliberately REFUSES to download EyePACS (~90 GB). Pretrain against it on
# Kaggle, where it is already mounted -- see docs/05_PROTOTYPE_SCOPE.md.
set -euo pipefail

DATASETS="aptos,idrid,drive"
DATA_DIR="data/raw"

usage() {
    cat <<'USAGE'
Usage: bash scripts/download_data.sh [--datasets aptos,idrid,drive] [--data-dir data/raw]

Datasets:
  aptos      APTOS 2019 (Kaggle API)         ~10 GB   automatic
  idrid      IDRiD                            ~2.5 GB  MANUAL (IEEE DataPort login)
  drive      DRIVE vessel segmentation        ~100 MB  MANUAL (grand-challenge login)
  messidor2  Messidor-2 external test set     ~5 GB    MANUAL (ADCIS terms)

Kaggle credentials: create a token at https://www.kaggle.com/settings/account
("Create New Token"), then:
  mkdir -p ~/.config/kaggle && mv ~/Downloads/kaggle.json ~/.config/kaggle/
  chmod 600 ~/.config/kaggle/kaggle.json
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets) DATASETS="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        -h|--help)  usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

have() { [[ ",$DATASETS," == *",$1,"* ]]; }

check_space() {
    local need_gb=$1 name=$2
    local free_gb
    free_gb=$(df -g . | awk 'NR==2 {print $4}')
    if (( free_gb < need_gb )); then
        echo "ERROR: $name needs ~${need_gb} GB but only ${free_gb} GB is free." >&2
        echo "Free space, or run this dataset's training on Kaggle instead." >&2
        exit 1
    fi
    echo "  disk check: ${free_gb} GB free, need ~${need_gb} GB  [ok]"
}

find_kaggle_json() {
    for p in "$HOME/.config/kaggle/kaggle.json" "$HOME/.kaggle/kaggle.json"; do
        [[ -f "$p" ]] && { echo "$p"; return 0; }
    done
    return 1
}

if have eyepacs; then
    cat >&2 <<'REFUSE'
ERROR: refusing to download EyePACS locally.

It is ~90 GB and will not fit alongside the other datasets on this machine.
It is PRETRAINING data, not task data -- run it on Kaggle where it is already
mounted, cache a 12-15k subset at 512 px there, and download only that.

See docs/05_PROTOTYPE_SCOPE.md section 3.2.
REFUSE
    exit 1
fi

mkdir -p "$DATA_DIR"

if have aptos; then
    echo "==> APTOS 2019"
    if ! kj=$(find_kaggle_json); then
        cat >&2 <<'NOCREDS'
  ERROR: no Kaggle credentials found.

  1. Sign in at https://www.kaggle.com/settings/account
  2. "Create New Token" -> downloads kaggle.json
  3. mkdir -p ~/.config/kaggle && mv ~/Downloads/kaggle.json ~/.config/kaggle/
     chmod 600 ~/.config/kaggle/kaggle.json
  4. Accept the competition rules at
     https://www.kaggle.com/c/aptos2019-blindness-detection/rules
     (downloads 403 until the rules are accepted, even with a valid token)
NOCREDS
        exit 1
    fi
    echo "  credentials: $kj"
    check_space 25 "APTOS"

    dest="$DATA_DIR/aptos"
    if [[ -d "$dest/train_images" ]]; then
        echo "  already present at $dest -- skipping"
    else
        mkdir -p "$dest"
        kaggle competitions download -c aptos2019-blindness-detection -p "$dest"
        echo "  unzipping..."
        unzip -q -o "$dest/aptos2019-blindness-detection.zip" -d "$dest"
        rm -f "$dest/aptos2019-blindness-detection.zip"
        echo "  done: $(find "$dest/train_images" -name '*.png' | wc -l | tr -d ' ') training images"
    fi
fi

if have idrid; then
    echo "==> IDRiD  (manual download required)"
    cat <<'IDRID'
  IEEE DataPort requires a login, so this cannot be scripted.
    1. https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
    2. Download all three parts (segmentation, grading, localisation)
    3. Extract into data/raw/idrid/ preserving the archive's folder structure
  Licence: CC BY 4.0 -- cite Porwal et al., Medical Image Analysis 2020.
IDRID
fi

if have drive; then
    echo "==> DRIVE  (manual download required)"
    cat <<'DRIVE'
    1. Register at https://drive.grand-challenge.org/
    2. Download the training and test archives
    3. Extract into data/raw/drive/
DRIVE
fi

if have messidor2; then
    echo "==> Messidor-2  (manual, and it is the LOCKED TEST SET)"
    cat <<'MESSIDOR'
    1. Accept terms at https://www.adcis.net/en/third-party/messidor2/
    2. Extract into data/external/messidor2/
  This is the locked external test set. Do not train on it, do not tune
  thresholds on it, and evaluate against it exactly once (Phase 8).
MESSIDOR
fi

echo
echo "Next: python scripts/preprocess.py --dataset aptos --size 512"
