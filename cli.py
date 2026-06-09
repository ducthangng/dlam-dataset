"""
TFT Time Series Forecasting — CLI training script.

Usage:
    python cli.py train \
        --data-dir ./data \
        --output-dir ./runs \
        --run-name baseline

    python cli.py train --help     # full options
    python cli.py predict --help   # inference only on existing checkpoint

Outputs per run (./runs/{run_name}_{timestamp}/):
    config.json                 — full hyperparameter config
    train.log                   — training log
    checkpoints/                — best model checkpoint (.ckpt)
    metrics/
        overall.json            — aggregate val metrics + baselines
        per_series_mae.csv      — MAE per series, sorted
        per_horizon_mae.csv     — MAE per forecast step
    plots/
        lr_finder.png           — LR finder curve
        per_horizon_mae.png     — error vs horizon
        per_series_mae_hist.png — distribution of series-level errors
        vsn_importance.png      — Variable Selection Network feature importance
        sample_forecasts.png    — 6 sample forecasts (best/worst/random)
    predictions/
        val_predictions.parquet — full predictions on validation holdout

Author: Nick (TU Darmstadt DLAM project)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

COVARIATE_COLS = [
    "demand_forecast", "staffing_forecast",
    "upstream_quality_forecast", "shock_risk",
    "unit_reliability_forecast", "queue_pressure_forecast",
    "network_pressure_forecast", "event_load_forecast",
    "service_irregularity_risk_forecast",
    "throughput_disruption_risk_forecast",
]

KNOWN_FUTURE = [
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "is_weekend",
    "trend",
    "demand_forecast", "staffing_forecast",
    "upstream_quality_forecast",
    "maintenance_known",
    "unit_reliability_forecast",
    "queue_pressure_forecast",
    "network_pressure_forecast",
    "event_load_forecast",
    "service_irregularity_risk_forecast",
    "throughput_disruption_risk_forecast",
]

PAST_ONLY = [
    "workload_intensity",
    "promotion_intensity",
    "shock_risk",
    "nominal_capacity",
]

STATIC_CATEGORICALS = ["series_id"]
STATIC_REALS = ["zone_sin", "zone_cos"]
TARGET = "target"
QUANTILES = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
MEDIAN_IDX = QUANTILES.index(0.5)


@dataclass
class Config:
    # Paths
    data_dir: str
    output_dir: str
    run_name: str

    # Data
    train_csv: str = "train.csv"
    forecast_val_csv: str = "forecast_index_validation.csv"

    # Windowing
    encoder_length: int = 336
    prediction_length: int = 336

    # Model
    hidden_size: int = 128
    attention_head_size: int = 4
    dropout: float = 0.2
    hidden_continuous_size: int = 64

    # Training
    batch_size: int = 128
    max_epochs: int = 40
    learning_rate: float = 3e-3
    weight_decay: float = 0.0
    gradient_clip_val: float = 0.1
    early_stop_patience: int = 6
    reduce_lr_patience: int = 4

    # Hardware
    num_workers: int = 4
    precision: str = "bf16-mixed"   # Blackwell: bf16; older GPUs: 16-mixed
    accelerator: str = "auto"
    devices: int = 1

    # Misc
    seed: int = 42
    skip_lr_find: bool = False
    log_interval: int = 100

    # Set at runtime
    run_dir: str = ""
    timestamp: str = ""


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

def setup_logging(run_dir: Path) -> logging.Logger:
    log_file = run_dir / "train.log"
    logger = logging.getLogger("tft")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def log_environment(log: logging.Logger):
    import torch
    log.info(f"Python: {sys.version.split()[0]}")
    log.info(f"PyTorch: {torch.__version__}")
    log.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        log.info(f"CUDA: {torch.version.cuda}")


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

def load_and_prep_data(cfg: Config, log: logging.Logger):
    import pandas as pd

    train_path = Path(cfg.data_dir) / cfg.train_csv
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")

    log.info(f"Loading {train_path}")
    train = pd.read_csv(train_path, parse_dates=["timestamp"])
    log.info(f"  shape={train.shape}, series={train['series_id'].nunique()}, "
             f"range={train['timestamp'].min()} → {train['timestamp'].max()}")

    # Fill missing covariates per-series (ffill then bfill)
    log.info("Imputing missing covariates per series (ffill+bfill)")
    train = train.sort_values(["series_id", "timestamp"])
    nan_before = train[COVARIATE_COLS].isna().sum().sum()
    train[COVARIATE_COLS] = (
        train.groupby("series_id")[COVARIATE_COLS]
        .transform(lambda x: x.ffill().bfill())
    )
    nan_after = train[COVARIATE_COLS].isna().sum().sum()
    log.info(f"  NaN: {nan_before} → {nan_after}")
    if nan_after > 0:
        raise ValueError(f"{nan_after} NaN values remain in covariates after imputation")

    # Build time_idx
    df = train.sort_values(["series_id", "timestamp"]).reset_index(drop=True)
    min_time = df["timestamp"].min()
    df["time_idx"] = ((df["timestamp"] - min_time).dt.total_seconds() / 3600).astype(int)

    # Gap check
    for sid, group in df.groupby("series_id"):
        expected = group["time_idx"].max() - group["time_idx"].min() + 1
        if expected != len(group):
            raise ValueError(f"Gap detected in series {sid}: expected {expected}, got {len(group)}")

    log.info(f"  time_idx range: {df['time_idx'].min()} → {df['time_idx'].max()}, total rows={len(df):,}")

    # Sanity on known_future
    nan_kf = df[KNOWN_FUTURE].isna().sum()
    if nan_kf.sum() > 0:
        log.error(f"NaN in known_future cols:\n{nan_kf[nan_kf > 0]}")
        raise ValueError("known_future columns must not contain NaN")

    return df


def build_datasets(df, cfg: Config, log: logging.Logger):
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer

    max_time_idx = df["time_idx"].max()
    training_cutoff = max_time_idx - cfg.prediction_length

    train_df = df[df["time_idx"] <= training_cutoff].copy()
    # Validation needs encoder context + holdout window
    val_df = df[df["time_idx"] > training_cutoff - cfg.encoder_length].copy()

    log.info(f"Split: training_cutoff time_idx={training_cutoff}")
    log.info(f"  train rows={len(train_df):,} | val rows={len(val_df):,}")

    training_dataset = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=TARGET,
        group_ids=STATIC_CATEGORICALS,
        min_encoder_length=cfg.encoder_length,
        max_encoder_length=cfg.encoder_length,
        min_prediction_length=cfg.prediction_length,
        max_prediction_length=cfg.prediction_length,
        static_categoricals=STATIC_CATEGORICALS,
        static_reals=STATIC_REALS,
        time_varying_known_reals=KNOWN_FUTURE,
        time_varying_unknown_reals=PAST_ONLY + [TARGET],
        target_normalizer=GroupNormalizer(
            groups=STATIC_CATEGORICALS, transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=False,
        allow_missing_timesteps=False,
    )

    # predict=False so we sample multiple windows in the holdout block
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        val_df,
        predict=False,
        stop_randomization=True,
        min_prediction_idx=training_cutoff + 1,
    )

    log.info(f"Train samples: {len(training_dataset):,} | Val samples: {len(validation_dataset):,}")
    return training_dataset, validation_dataset, training_cutoff


def build_dataloaders(training_ds, validation_ds, cfg: Config):
    train_loader = training_ds.to_dataloader(
        train=True,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=True,
    )
    val_loader = validation_ds.to_dataloader(
        train=False,
        batch_size=cfg.batch_size * 2,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=True,
    )
    return train_loader, val_loader


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

def build_model(training_ds, cfg: Config, log: logging.Logger):
    import lightning.pytorch as pl
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss

    pl.seed_everything(cfg.seed, workers=True)

    tft = TemporalFusionTransformer.from_dataset(
        training_ds,
        hidden_size=cfg.hidden_size,
        attention_head_size=cfg.attention_head_size,
        dropout=cfg.dropout,
        hidden_continuous_size=cfg.hidden_continuous_size,
        learning_rate=cfg.learning_rate,
        loss=QuantileLoss(quantiles=QUANTILES),
        output_size=len(QUANTILES),
        log_interval=cfg.log_interval,
        log_val_interval=1,
        reduce_on_plateau_patience=cfg.reduce_lr_patience,
        optimizer="adamw",
        weight_decay=cfg.weight_decay,
    )

    n_params = sum(p.numel() for p in tft.parameters())
    log.info(f"Model parameters: {n_params:,}")
    return tft


def run_lr_finder(tft, train_loader, val_loader, cfg: Config, run_dir: Path, log: logging.Logger):
    import lightning.pytorch as pl
    from lightning.pytorch.tuner import Tuner
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log.info("Running LR finder...")
    tuner_trainer = pl.Trainer(
        accelerator=cfg.accelerator, devices=cfg.devices,
        gradient_clip_val=cfg.gradient_clip_val,
        max_epochs=1, enable_progress_bar=False, logger=False,
        precision=cfg.precision,
    )
    tuner = Tuner(tuner_trainer)
    lr_finder = tuner.lr_find(
        tft, train_dataloaders=train_loader, val_dataloaders=val_loader,
        min_lr=1e-5, max_lr=1e-1, num_training=100,
    )
    suggested = lr_finder.suggestion()
    log.info(f"  Suggested LR: {suggested:.2e}")

    fig = lr_finder.plot(suggest=True)
    fig.savefig(run_dir / "plots" / "lr_finder.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return suggested


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train_model(tft, train_loader, val_loader, cfg: Config, run_dir: Path, log: logging.Logger):
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import (
        EarlyStopping, ModelCheckpoint, LearningRateMonitor
    )
    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

    ckpt_dir = run_dir / "checkpoints"
    early_stop = EarlyStopping(
        monitor="val_loss", patience=cfg.early_stop_patience,
        mode="min", verbose=True, min_delta=1e-3,
    )
    checkpoint = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss", mode="min", save_top_k=2, save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    loggers = [
        CSVLogger(save_dir=str(run_dir), name="lightning_logs"),
        TensorBoardLogger(save_dir=str(run_dir), name="tb_logs"),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator, devices=cfg.devices,
        precision=cfg.precision,
        callbacks=[early_stop, checkpoint, lr_monitor],
        gradient_clip_val=cfg.gradient_clip_val,
        log_every_n_steps=50,
        logger=loggers,
        enable_progress_bar=True,
        deterministic=False,
    )

    log.info(f"Starting training: max_epochs={cfg.max_epochs}, "
             f"patience={cfg.early_stop_patience}, precision={cfg.precision}")
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    log.info(f"Training complete.")
    log.info(f"  best val_loss: {checkpoint.best_model_score:.4f}")
    log.info(f"  best ckpt: {checkpoint.best_model_path}")
    return checkpoint.best_model_path, float(checkpoint.best_model_score)


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def evaluate(best_ckpt: str, val_loader, training_ds, cfg: Config,
             run_dir: Path, log: logging.Logger):
    import numpy as np
    import pandas as pd
    import torch
    from pytorch_forecasting import TemporalFusionTransformer
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log.info("Loading best checkpoint for evaluation...")
    tft = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)
    tft.eval()

    log.info("Running inference on validation set...")
    raw = tft.predict(
        val_loader, mode="raw",
        return_x=True, return_index=True,
        trainer_kwargs={
            "accelerator": cfg.accelerator,
            "devices": cfg.devices,
            "precision": cfg.precision,
            "enable_progress_bar": False,
            "logger": False,
        },
    )

    # Predictions: [n_samples, horizon, n_quantiles]
    y_pred_q = raw.output.prediction.cpu().numpy()
    y_pred = y_pred_q[..., MEDIAN_IDX]                      # median

    # Decoder target (ground truth)
    y_true = raw.x["decoder_target"].cpu().numpy()
    idx_df = raw.index.reset_index(drop=True)

    log.info(f"  predictions shape: {y_pred.shape}, targets: {y_true.shape}")

    # ---- Overall MAE
    mae_overall = float(np.abs(y_pred - y_true).mean())
    log.info(f"Overall val MAE (median): {mae_overall:.4f}")

    # ---- Per-series MAE
    mae_per_sample = np.abs(y_pred - y_true).mean(axis=1)
    series_df = idx_df.copy()
    series_df["mae"] = mae_per_sample
    per_series = (
        series_df.groupby("series_id")["mae"].mean()
        .reset_index().sort_values("mae")
    )
    per_series.to_csv(run_dir / "metrics" / "per_series_mae.csv", index=False)
    log.info(f"  per-series MAE: min={per_series['mae'].min():.3f} "
             f"median={per_series['mae'].median():.3f} "
             f"max={per_series['mae'].max():.3f}")

    # ---- Per-horizon MAE
    mae_per_step = np.abs(y_pred - y_true).mean(axis=0)
    pd.DataFrame({
        "step": np.arange(1, len(mae_per_step) + 1),
        "mae": mae_per_step,
    }).to_csv(run_dir / "metrics" / "per_horizon_mae.csv", index=False)

    # ---- Baseline: Seasonal Naive (168h)
    log.info("Computing SeasonalNaive(168) baseline...")
    sn_errors = []
    for x, y in iter(val_loader):
        enc_t = x["encoder_target"].cpu().numpy()
        dec_t = y[0].cpu().numpy() if isinstance(y, (tuple, list)) else y.cpu().numpy()
        for i in range(enc_t.shape[0]):
            history = enc_t[i]
            season = 168
            n_repeats = cfg.prediction_length // season + 1
            pred = np.tile(history[-season:], n_repeats)[:cfg.prediction_length]
            sn_errors.append(float(np.abs(pred - dec_t[i]).mean()))
    sn_mae = float(np.mean(sn_errors))
    improvement = (1 - mae_overall / sn_mae) * 100 if sn_mae > 0 else float("nan")
    log.info(f"  SeasonalNaive MAE: {sn_mae:.4f} | TFT improvement: {improvement:.1f}%")

    # ---- Save overall metrics JSON
    overall = {
        "val_mae_median": mae_overall,
        "val_mae_seasonal_naive_168": sn_mae,
        "improvement_pct_over_seasonal_naive": improvement,
        "n_val_samples": int(len(y_pred)),
        "horizon": int(cfg.prediction_length),
        "encoder_length": int(cfg.encoder_length),
        "per_series_mae_summary": {
            "min": float(per_series["mae"].min()),
            "p25": float(per_series["mae"].quantile(0.25)),
            "median": float(per_series["mae"].median()),
            "p75": float(per_series["mae"].quantile(0.75)),
            "max": float(per_series["mae"].max()),
        },
    }
    with open(run_dir / "metrics" / "overall.json", "w") as f:
        json.dump(overall, f, indent=2)

    # ---- Plots
    plot_per_horizon_mae(mae_per_step, run_dir / "plots" / "per_horizon_mae.png", cfg)
    plot_per_series_distribution(per_series, run_dir / "plots" / "per_series_mae_hist.png")
    plot_sample_forecasts(y_pred_q, y_true, idx_df, per_series,
                          run_dir / "plots" / "sample_forecasts.png", cfg)

    # ---- VSN interpretation
    try:
        log.info("Computing Variable Selection Network interpretation...")
        interpretation = tft.interpret_output(raw.output, reduction="sum")
        fig = tft.plot_interpretation(interpretation)
        if isinstance(fig, dict):
            for key, f in fig.items():
                f.savefig(run_dir / "plots" / f"vsn_{key}.png", dpi=120, bbox_inches="tight")
                plt.close(f)
        else:
            fig.savefig(run_dir / "plots" / "vsn_importance.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

        # Save feature importance as CSV too
        save_vsn_csv(tft, interpretation, training_ds, run_dir / "metrics")
    except Exception as e:
        log.warning(f"VSN interpretation failed: {e}")

    # ---- Save predictions
    log.info("Saving predictions to parquet...")
    save_predictions(y_pred_q, y_true, idx_df, run_dir / "predictions" / "val_predictions.parquet")

    return overall


def plot_per_horizon_mae(mae_per_step, path, cfg):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(range(1, len(mae_per_step) + 1), mae_per_step, linewidth=1.2)
    ax.axhline(mae_per_step.mean(), color="red", linestyle="--", alpha=0.6,
               label=f"mean MAE = {mae_per_step.mean():.3f}")
    # Mark daily/weekly grids
    for d in range(1, cfg.prediction_length // 24 + 1):
        ax.axvline(d * 24, color="gray", alpha=0.15, linewidth=0.5)
    ax.set_xlabel("Forecast step (hours ahead)")
    ax.set_ylabel("MAE")
    ax.set_title(f"Validation MAE vs forecast horizon ({cfg.prediction_length}h)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_per_series_distribution(per_series, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(per_series["mae"], bins=30, edgecolor="black", alpha=0.75)
    ax.axvline(per_series["mae"].median(), color="red", linestyle="--",
               label=f"median = {per_series['mae'].median():.3f}")
    ax.set_xlabel("MAE")
    ax.set_ylabel("Number of series")
    ax.set_title(f"Per-series MAE distribution (n={len(per_series)})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_sample_forecasts(y_pred_q, y_true, idx_df, per_series, path, cfg):
    import matplotlib.pyplot as plt
    import numpy as np

    # Pick 2 best, 2 worst, 2 median series
    best_ids = per_series.head(2)["series_id"].tolist()
    worst_ids = per_series.tail(2)["series_id"].tolist()
    mid = len(per_series) // 2
    median_ids = per_series.iloc[mid - 1:mid + 1]["series_id"].tolist()
    selected = best_ids + median_ids + worst_ids

    fig, axes = plt.subplots(3, 2, figsize=(14, 9))
    axes = axes.flatten()

    for ax, sid in zip(axes, selected):
        mask = (idx_df["series_id"] == sid).values
        if not mask.any():
            continue
        # Pick first window for this series
        i = np.where(mask)[0][0]
        steps = np.arange(1, cfg.prediction_length + 1)
        ax.plot(steps, y_true[i], label="actual", color="black", linewidth=1.2)
        ax.plot(steps, y_pred_q[i, :, MEDIAN_IDX], label="median", color="C0", linewidth=1)
        # 50% interval (q25-q75)
        ax.fill_between(steps, y_pred_q[i, :, 2], y_pred_q[i, :, 4],
                        alpha=0.3, color="C0", label="50% PI")
        # 80% interval (q10-q90)
        ax.fill_between(steps, y_pred_q[i, :, 1], y_pred_q[i, :, 5],
                        alpha=0.15, color="C0", label="80% PI")
        series_mae = float(np.abs(y_pred_q[i, :, MEDIAN_IDX] - y_true[i]).mean())
        ax.set_title(f"series={sid} | MAE={series_mae:.3f}", fontsize=10)
        ax.set_xlabel("step ahead")
        ax.grid(alpha=0.3)
        if ax is axes[0]:
            ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Sample forecasts: top=best, middle=median, bottom=worst", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_vsn_csv(tft, interpretation, training_ds, metrics_dir: Path):
    import pandas as pd

    for key in ["encoder_variables", "decoder_variables", "static_variables"]:
        if key not in interpretation:
            continue
        scores = interpretation[key].cpu().numpy()
        # Get matching variable names
        if key == "encoder_variables":
            names = training_ds.reals  # may be longer; we slice
        elif key == "decoder_variables":
            names = training_ds.reals
        else:
            names = training_ds.static_categoricals + training_ds.static_reals
        names = list(names)[:len(scores)]
        df = pd.DataFrame({"feature": names, "importance": scores})
        df = df.sort_values("importance", ascending=False)
        df.to_csv(metrics_dir / f"vsn_{key}.csv", index=False)


def save_predictions(y_pred_q, y_true, idx_df, path: Path):
    import numpy as np
    import pandas as pd

    n_samples, horizon, n_quantiles = y_pred_q.shape
    rows = []
    for i in range(n_samples):
        sid = idx_df.iloc[i]["series_id"]
        t0 = int(idx_df.iloc[i]["time_idx"])
        for h in range(horizon):
            row = {
                "series_id": sid,
                "time_idx": t0 + h,
                "step": h + 1,
                "actual": float(y_true[i, h]),
            }
            for qi, q in enumerate(QUANTILES):
                row[f"q{int(q * 100):02d}"] = float(y_pred_q[i, h, qi])
            rows.append(row)
    pd.DataFrame(rows).to_parquet(path, index=False)


# -----------------------------------------------------------------------------
# Run orchestration
# -----------------------------------------------------------------------------

def make_run_dir(cfg: Config) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.timestamp = timestamp
    run_dir = Path(cfg.output_dir) / f"{cfg.run_name}_{timestamp}"
    cfg.run_dir = str(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(cfg: Config, run_dir: Path):
    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)


def cmd_train(args):
    cfg = Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        encoder_length=args.encoder_length,
        prediction_length=args.prediction_length,
        hidden_size=args.hidden_size,
        attention_head_size=args.attention_head_size,
        dropout=args.dropout,
        hidden_continuous_size=args.hidden_continuous_size,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_val=args.gradient_clip_val,
        early_stop_patience=args.early_stop_patience,
        reduce_lr_patience=args.reduce_lr_patience,
        num_workers=args.num_workers,
        precision=args.precision,
        accelerator=args.accelerator,
        devices=args.devices,
        seed=args.seed,
        skip_lr_find=args.skip_lr_find,
    )

    run_dir = make_run_dir(cfg)
    log = setup_logging(run_dir)
    log.info("=" * 70)
    log.info(f"TFT training — run: {cfg.run_name}_{cfg.timestamp}")
    log.info(f"Output: {run_dir}")
    log.info("=" * 70)
    log_environment(log)
    save_config(cfg, run_dir)

    # Pipeline
    df = load_and_prep_data(cfg, log)
    training_ds, validation_ds, _ = build_datasets(df, cfg, log)
    train_loader, val_loader = build_dataloaders(training_ds, validation_ds, cfg)
    tft = build_model(training_ds, cfg, log)

    if not cfg.skip_lr_find:
        suggested = run_lr_finder(tft, train_loader, val_loader, cfg, run_dir, log)
        tft.hparams.learning_rate = suggested
        cfg.learning_rate = float(suggested)
        save_config(cfg, run_dir)  # re-save with updated LR

    best_ckpt, best_val_loss = train_model(tft, train_loader, val_loader, cfg, run_dir, log)

    # Re-build val_loader fresh for evaluation (some loaders consumed)
    _, val_loader = build_dataloaders(training_ds, validation_ds, cfg)
    metrics = evaluate(best_ckpt, val_loader, training_ds, cfg, run_dir, log)

    log.info("=" * 70)
    log.info("FINAL RESULTS")
    log.info(f"  best val_loss (QuantileLoss): {best_val_loss:.4f}")
    log.info(f"  val MAE (median):             {metrics['val_mae_median']:.4f}")
    log.info(f"  SeasonalNaive baseline MAE:   {metrics['val_mae_seasonal_naive_168']:.4f}")
    log.info(f"  improvement over baseline:    {metrics['improvement_pct_over_seasonal_naive']:.1f}%")
    log.info(f"  artifacts: {run_dir}")
    log.info("=" * 70)


def cmd_predict(args):
    """Run inference on existing checkpoint, regenerate metrics + plots."""
    import json

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    with open(run_dir / "config.json") as f:
        cfg_dict = json.load(f)
    cfg = Config(**cfg_dict)

    log = setup_logging(run_dir)
    log.info(f"Predict mode on existing run: {run_dir}")

    df = load_and_prep_data(cfg, log)
    training_ds, validation_ds, _ = build_datasets(df, cfg, log)
    _, val_loader = build_dataloaders(training_ds, validation_ds, cfg)

    # Find best checkpoint
    ckpts = list((run_dir / "checkpoints").glob("tft-*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {run_dir / 'checkpoints'}")
    # Pick lowest val_loss from filename
    best = min(ckpts, key=lambda p: float(p.stem.split("val_loss=")[-1]))
    log.info(f"Using checkpoint: {best}")

    evaluate(str(best), val_loader, training_ds, cfg, run_dir, log)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog="cli.py",
        description="TFT time series forecasting — training & evaluation CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    pt = sub.add_parser("train", help="Train a new model end-to-end")
    pt.add_argument("--data-dir", required=True, help="Directory containing train.csv etc.")
    pt.add_argument("--output-dir", default="./runs", help="Where run dirs are created")
    pt.add_argument("--run-name", default="baseline", help="Run name prefix")

    # Windowing
    pt.add_argument("--encoder-length", type=int, default=336)
    pt.add_argument("--prediction-length", type=int, default=336)

    # Model
    pt.add_argument("--hidden-size", type=int, default=128)
    pt.add_argument("--attention-head-size", type=int, default=4)
    pt.add_argument("--dropout", type=float, default=0.2)
    pt.add_argument("--hidden-continuous-size", type=int, default=64)

    # Training
    pt.add_argument("--batch-size", type=int, default=128)
    pt.add_argument("--max-epochs", type=int, default=40)
    pt.add_argument("--learning-rate", type=float, default=3e-3)
    pt.add_argument("--weight-decay", type=float, default=0.0)
    pt.add_argument("--gradient-clip-val", type=float, default=0.1)
    pt.add_argument("--early-stop-patience", type=int, default=6)
    pt.add_argument("--reduce-lr-patience", type=int, default=4)

    # Hardware
    pt.add_argument("--num-workers", type=int, default=4)
    pt.add_argument("--precision", default="bf16-mixed",
                    choices=["32", "16-mixed", "bf16-mixed"],
                    help="bf16-mixed for Blackwell/A100/L4, 16-mixed for older")
    pt.add_argument("--accelerator", default="auto",
                    choices=["auto", "gpu", "cpu"])
    pt.add_argument("--devices", type=int, default=1)

    # Misc
    pt.add_argument("--seed", type=int, default=42)
    pt.add_argument("--skip-lr-find", action="store_true",
                    help="Skip LR finder, use --learning-rate as-is")

    pt.set_defaults(func=cmd_train)

    # predict
    pp = sub.add_parser("predict",
                        help="Re-run evaluation on existing checkpoint")
    pp.add_argument("--run-dir", required=True,
                    help="Existing run dir (must contain config.json + checkpoints/)")
    pp.set_defaults(func=cmd_predict)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()