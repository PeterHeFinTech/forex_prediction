"""Evaluate a trained Perceiver on Up predictions and plot versus baseline."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.solver import (
    compounded_annualized_return,
    compounded_annualized_volatility,
    compounded_cumulative_return,
    compounded_max_drawdown,
    compounded_sharpe_ratio,
)
from models.Perceiver import Perceiver


def parse_args():
    parser = argparse.ArgumentParser(description="Perceiver Up-only test evaluation")
    parser.add_argument("--data_path", type=Path,
                        default=Path("/home/corelabtq/Desktop/Research/a_stock.npz"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/home/corelabtq/Desktop/Research/forex/checkpoints/best_model_a_stock.pt"))
    parser.add_argument("--output_plot", type=Path,
                        default=Path("/home/corelabtq/Desktop/Research/forex/logs/a_stock_up_vs_baseline.png"))
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--use_amp", action="store_true")
    return parser.parse_args()


def binary_metrics(predicted_up, true_up):
    tp = int(np.sum(predicted_up & true_up))
    fp = int(np.sum(predicted_up & ~true_up))
    fn = int(np.sum(~predicted_up & true_up))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return tp, fp, fn, precision, recall, f1


def daily_equal_weight(returns, timestamps, selected=None):
    returns = np.asarray(returns, dtype=np.float64)
    dates = np.asarray(timestamps).astype("datetime64[D]")
    selected = np.ones(len(returns), dtype=bool) if selected is None else np.asarray(selected, dtype=bool)
    valid = np.isfinite(returns) & ~np.isnat(dates)
    returns, dates, selected = returns[valid], dates[valid], selected[valid]
    order = np.argsort(dates, kind="mergesort")
    returns, dates, selected = returns[order], dates[order], selected[order]
    unique_dates, inverse = np.unique(dates, return_inverse=True)
    sums = np.bincount(inverse, weights=np.where(selected, returns, 0.0))
    counts = np.bincount(inverse, weights=selected.astype(np.float64))
    daily = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    trades = np.bincount(inverse, weights=selected.astype(np.int64)).astype(np.int64)
    return unique_dates, daily, trades


def summarize(name, daily_returns):
    return {
        "name": name,
        "cumulative": compounded_cumulative_return(daily_returns),
        "annualized": compounded_annualized_return(daily_returns),
        "volatility": compounded_annualized_volatility(daily_returns),
        "sharpe": compounded_sharpe_ratio(daily_returns, risk_free_rate=0.02),
        "max_drawdown": compounded_max_drawdown(daily_returns, aggregate_by_time=False),
    }


def print_summary(metrics):
    print(f"\n{metrics['name']} (daily equal weight, compounded)")
    print(f"  Compounded cumulative return: {metrics['cumulative'] * 100:.4f}%")
    print(f"  Annualized return:            {metrics['annualized'] * 100:.4f}%")
    print(f"  Annualized volatility:        {metrics['volatility'] * 100:.4f}%")
    print(f"  Sharpe ratio:                 {metrics['sharpe']:.4f}")
    print(f"  Maximum drawdown:             {metrics['max_drawdown'] * 100:.4f}%")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full test evaluation")
    device = torch.device("cuda:0")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = Perceiver(
        hidden_dim=256, num_layers=3, seq_size=96, num_features=12,
        num_heads=16, num_classes=3, dropout=0.2,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    print(
        f"Checkpoint epoch={checkpoint.get('epoch')}, val_loss={checkpoint.get('val_loss'):.6f}, "
        f"val_f1={checkpoint.get('val_f1'):.4f}%",
        flush=True,
    )

    print(f"Loading test split: {args.data_path}", flush=True)
    data = np.load(args.data_path, allow_pickle=False)
    x_test = data["X_test"]
    y_test = np.asarray(data["y_test"], dtype=np.int64)
    returns = np.asarray(data["target_returns_test"], dtype=np.float64)
    timestamps = np.asarray(data["target_timestamp_test"]).astype("datetime64[D]")
    n_samples = len(y_test)
    if not (len(x_test) == len(returns) == len(timestamps) == n_samples):
        raise ValueError("Test arrays are not sample-aligned")
    print(f"Test samples={n_samples:,}; input shape={x_test.shape[1:]}", flush=True)

    preds = np.empty(n_samples, dtype=np.int8)
    up_probs = np.empty(n_samples, dtype=np.float32)
    started = time.time()
    with torch.inference_mode():
        for start in range(0, n_samples, args.batch_size):
            stop = min(start + args.batch_size, n_samples)
            inputs = torch.from_numpy(x_test[start:stop]).to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.use_amp):
                logits = model(inputs)
            probabilities = logits.float().softmax(dim=1)
            preds[start:stop] = probabilities.argmax(dim=1).cpu().numpy()
            up_probs[start:stop] = probabilities[:, 2].cpu().numpy()
            batch_number = (start // args.batch_size) + 1
            if batch_number % 250 == 0 or stop == n_samples:
                print(f"Test {stop:,}/{n_samples:,} ({time.time() - started:.1f}s)", flush=True)

    true_up = y_test == 2
    argmax_up = preds == 2
    gated_up = argmax_up & (up_probs > args.confidence)

    print("\nUp class (one-vs-rest, argmax prediction)")
    tp, fp, fn, precision, recall, f1 = binary_metrics(argmax_up, true_up)
    print(f"  Support (true Up): {true_up.sum():,}")
    print(f"  Predicted Up:      {argmax_up.sum():,}")
    print(f"  TP / FP / FN:      {tp:,} / {fp:,} / {fn:,}")
    print(f"  Precision:         {precision * 100:.4f}%")
    print(f"  Recall:            {recall * 100:.4f}%")
    print(f"  F1:                {f1 * 100:.4f}%")

    print(f"\nUp class (argmax=Up and P(Up) > {args.confidence:.2f})")
    tp, fp, fn, precision, recall, f1 = binary_metrics(gated_up, true_up)
    print(f"  Selected trades:   {gated_up.sum():,} ({gated_up.mean() * 100:.4f}% of test)")
    print(f"  TP / FP / FN:      {tp:,} / {fp:,} / {fn:,}")
    print(f"  Precision:         {precision * 100:.4f}%")
    print(f"  Recall:            {recall * 100:.4f}%")
    print(f"  F1:                {f1 * 100:.4f}%")

    # The dataset was capped before float32 serialization. Do not reapply an
    # exact 0.10 comparison here: float32(0.10) is 0.10000000149.
    valid = np.isfinite(returns) & ~np.isnat(timestamps)
    if not np.all(valid):
        print(f"  Return/date filter removed {(~valid).sum():,} invalid samples", flush=True)
    gated_valid = gated_up & valid
    executed_returns = returns[gated_valid]
    print(f"  Mean realized return/trade: {executed_returns.mean() * 100 if len(executed_returns) else 0.0:.6f}%")
    print(f"  Positive-return rate:       {(executed_returns > 0).mean() * 100 if len(executed_returns) else 0.0:.4f}%")

    dates, baseline_daily, baseline_count = daily_equal_weight(returns[valid], timestamps[valid])
    model_dates, model_daily, model_count = daily_equal_weight(
        returns[valid], timestamps[valid], gated_up[valid]
    )
    if not np.array_equal(dates, model_dates):
        raise RuntimeError("Baseline and model date grids differ")

    baseline_metrics = summarize("Baseline: all eligible stocks", baseline_daily)
    model_metrics = summarize(f"Model: Up with P(Up)>{args.confidence:.2f}", model_daily)
    print_summary(baseline_metrics)
    print_summary(model_metrics)
    print(f"\nTrading days: {len(dates):,}")
    print(f"Model active days: {np.sum(model_count > 0):,} ({np.mean(model_count > 0) * 100:.2f}%)")
    print(f"Median selected stocks on active days: "
          f"{np.median(model_count[model_count > 0]) if np.any(model_count > 0) else 0:.1f}")

    baseline_equity = np.cumprod(1.0 + baseline_daily)
    model_equity = np.cumprod(1.0 + model_daily)
    args.output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(dates, baseline_equity, label="Baseline: all eligible stocks", linewidth=2.0)
    ax.plot(dates, model_equity, label=f"Model: Up, P(Up) > {args.confidence:.2f}", linewidth=2.0)
    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
    if np.all(baseline_equity > 0) and np.all(model_equity > 0):
        ax.set_yscale("log")
        ax.set_ylabel("Portfolio wealth (log scale, initial = 1)")
    else:
        ax.set_ylabel("Portfolio wealth (initial = 1)")
    ax.set_title("A-share Test Set: Up-only Model vs Equal-weight Baseline")
    ax.set_xlabel("Target date")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_plot, dpi=180)
    plt.close(fig)
    print(f"\nSaved plot: {args.output_plot}", flush=True)


if __name__ == "__main__":
    main()
