#!/usr/bin/env python3
"""Train Phase 5's fusion head: concat(CNN embedding, lesion features) -> ordinal head.

Reuses `GradingModule` unchanged -- it already loss-agnostic (works with
"ce", "corn", "regression", "distance_ce" via drdetect.grading.losses) and
doesn't care about the architecture of `self.model`, only that it maps the
right input to the right number of outputs. The fusion head IS a grading
model; only its input (a 1289-d feature vector, not an image) and its
architecture (a small MLP, not a CNN) differ from the baseline.

Requires `scripts/extract_lesion_features.py` to have already produced
`data/manifests/aptos_lesion_features.csv`.

Single train/val split (`--n-splits 5 --fold 0`, not full 5-fold CV), per
the scoping decision applied to every Phase 4/5/6 item since soft exudates.

Reports the roadmap's own exit criterion directly: fusion vs. grading-alone
QWK on the same held-out images, plus the McNemar test on paired
correctness (drdetect.eval-style: exact two-sided binomial on discordant
pairs, matching scripts/compare.py's own `mcnemar`).

Usage:
    python scripts/train_fusion.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def pick_accelerator() -> str:
    import torch

    if torch.cuda.is_available():
        return "gpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def mcnemar(correct_a, correct_b) -> tuple[int, int, float]:
    """Exact McNemar on paired correctness. Identical to scripts/compare.py's
    own `mcnemar` -- inlined rather than imported since scripts/ isn't a
    package other scripts import from elsewhere in this project."""
    from scipy import stats

    b_only = int((~correct_a & correct_b).sum())
    a_only = int((correct_a & ~correct_b).sum())
    n = a_only + b_only
    if n == 0:
        return b_only, a_only, 1.0
    p = float(min(1.0, 2 * stats.binom.cdf(min(a_only, b_only), n, 0.5)))
    return b_only, a_only, p


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--features-csv", default="data/manifests/aptos_lesion_features.csv")
    p.add_argument("--manifest", default="data/manifests/aptos_512.csv")
    p.add_argument("--data-root", default="data/processed")
    p.add_argument(
        "--grading-checkpoint",
        default="models/checkpoints/baseline_effb0_512_fold0/best.ckpt",
    )
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--out-dir", default="models/checkpoints")
    p.add_argument("--run-name", default="fusion_head")
    args = p.parse_args()

    import lightning as L
    import numpy as np
    import pandas as pd
    import torch
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from torch.utils.data import DataLoader, TensorDataset

    from drdetect.data.splits import assert_no_group_leakage, stratified_group_split
    from drdetect.fusion.embedding import extract_cnn_embedding_and_logits, load_grading_backbone
    from drdetect.fusion.features import FEATURE_NAMES
    from drdetect.fusion.model import FusionHead
    from drdetect.grading.losses import corn_task_pos_weights, decode_output, outputs_for_loss
    from drdetect.grading.module import GradingModule
    from drdetect.utils.seed import seed_everything

    seed_everything(args.seed)
    accelerator = pick_accelerator()
    device = torch.device("mps" if accelerator == "mps" else accelerator)
    print(f"accelerator: {accelerator}")

    features_df = pd.read_csv(args.features_csv)
    manifest = pd.read_csv(args.manifest)[["image_id", "path", "group_id"]]
    df = features_df.merge(manifest, on="image_id", how="left", validate="one_to_one")
    if df["group_id"].isna().any():
        missing = df[df["group_id"].isna()]["image_id"].tolist()
        print(
            f"ERROR: {len(missing)} image_ids in {args.features_csv} not found in "
            f"{args.manifest}: {missing[:5]}...",
            file=sys.stderr,
        )
        return 1
    print(f"{len(df)} images with lesion features, grades {sorted(df['grade'].unique())}")

    grading_model = load_grading_backbone(args.grading_checkpoint, device=device)
    embeddings, baseline_logits = [], []
    for _i, row in df.iterrows():
        image_path = Path(args.data_root) / row["path"]
        emb, logits = extract_cnn_embedding_and_logits(image_path, grading_model, device)
        embeddings.append(emb.numpy())
        baseline_logits.append(logits.numpy())
    embeddings = np.stack(embeddings)
    baseline_logits = np.stack(baseline_logits)
    lesion_features = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    lesion_features = np.nan_to_num(
        lesion_features, nan=0.0
    )  # distance_to_fovea_mean when no lesions found
    labels = df["grade"].to_numpy()
    groups = df["group_id"].to_numpy()
    print(f"embeddings: {embeddings.shape}  lesion features: {lesion_features.shape}")

    folds, strategy = stratified_group_split(
        labels.tolist(), groups.tolist(), n_splits=args.n_splits, seed=args.seed
    )
    assert_no_group_leakage(folds, groups.tolist())
    val_idx = folds[args.fold]
    train_idx = np.setdiff1d(np.arange(len(df)), val_idx)
    print(f"split strategy: {strategy}  train: {len(train_idx)}  val: {len(val_idx)}")

    # Lesion features standardised on TRAIN statistics only, applied to both --
    # they range from ~0-0.2 (mean probabilities) to 0-80+ (MA count), and the
    # embedding is already implicitly normalised by the backbone's own pooling.
    lf_mean = lesion_features[train_idx].mean(axis=0)
    lf_std = lesion_features[train_idx].std(axis=0)
    lf_std[lf_std < 1e-6] = 1.0
    lesion_features_norm = (lesion_features - lf_mean) / lf_std

    fused = np.concatenate([embeddings, lesion_features_norm], axis=1).astype(np.float32)

    def make_loader(idx, shuffle):
        x = torch.from_numpy(fused[idx])
        y = torch.from_numpy(labels[idx]).long()
        return DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=shuffle)

    train_dl = make_loader(train_idx, shuffle=True)
    val_dl = make_loader(val_idx, shuffle=False)

    train_counts = np.bincount(labels[train_idx], minlength=5)
    print("train class counts:", train_counts.tolist())
    task_pos_weights = corn_task_pos_weights(labels[train_idx].tolist())
    print(f"CORN task weights: {[round(w, 3) for w in task_pos_weights]}")

    head = FusionHead(
        embedding_dim=embeddings.shape[1],
        lesion_dim=lesion_features.shape[1],
        hidden_dim=args.hidden_dim,
        num_outputs=outputs_for_loss("corn"),
        dropout=args.dropout,
    )
    module = GradingModule(
        head,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss_name="corn",
        task_pos_weights=task_pos_weights,
        max_epochs=args.epochs,
        warmup_epochs=0,
    )

    out_dir = Path(args.out_dir) / f"{args.run_name}_fold{args.fold}"
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        precision="32-true",
        deterministic=False,
        logger=CSVLogger(save_dir="runs", name=args.run_name, flush_logs_every_n_steps=10),
        callbacks=[
            ModelCheckpoint(
                dirpath=out_dir, filename="best", monitor="val/qwk", mode="max", save_top_k=1
            ),
            EarlyStopping(monitor="val/qwk", mode="max", patience=args.patience, min_delta=1e-3),
        ],
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    trainer.fit(module, train_dl, val_dl)

    ckpt_cb = trainer.checkpoint_callback
    best_qwk = (
        float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score is not None else float("nan")
    )
    print(f"\nfusion best val/qwk: {best_qwk:.4f}  ({trainer.current_epoch + 1} epochs)")

    best_module = GradingModule.load_from_checkpoint(
        ckpt_cb.best_model_path, model=head, map_location=device
    )
    best_module.eval()
    with torch.no_grad():
        val_x = torch.from_numpy(fused[val_idx]).to(device)
        fusion_logits = best_module(val_x).cpu()
    fusion_pred, _ = decode_output(fusion_logits, "corn")
    baseline_pred, _ = decode_output(torch.from_numpy(baseline_logits[val_idx]), "ce")

    y_val = labels[val_idx]
    from drdetect.eval.metrics import quadratic_weighted_kappa

    qwk_fusion = quadratic_weighted_kappa(y_val, fusion_pred)
    qwk_baseline = quadratic_weighted_kappa(y_val, baseline_pred)
    correct_fusion = fusion_pred == y_val
    correct_baseline = baseline_pred == y_val
    b_only, a_only, mcnemar_p = mcnemar(correct_baseline, correct_fusion)

    print(f"\n{'=' * 60}")
    print(f"grading-alone (baseline checkpoint) val QWK : {qwk_baseline:.4f}")
    print(f"fusion head val QWK                          : {qwk_fusion:.4f}")
    print(f"grading-alone accuracy                       : {correct_baseline.mean():.4f}")
    print(f"fusion accuracy                               : {correct_fusion.mean():.4f}")
    print(
        f"McNemar: fusion-only-correct={b_only}  baseline-only-correct={a_only}  p={mcnemar_p:.4f}"
    )

    result = {
        "n_images": len(df),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "split_strategy": strategy,
        "fusion_best_val_qwk_during_training": best_qwk,
        "qwk_baseline_grading_alone": qwk_baseline,
        "qwk_fusion": qwk_fusion,
        "acc_baseline": float(correct_baseline.mean()),
        "acc_fusion": float(correct_fusion.mean()),
        "mcnemar_fusion_only_correct": b_only,
        "mcnemar_baseline_only_correct": a_only,
        "mcnemar_p": mcnemar_p,
        "checkpoint": str(ckpt_cb.best_model_path),
    }
    out_path = Path("runs") / args.run_name / "fusion_evaluation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
