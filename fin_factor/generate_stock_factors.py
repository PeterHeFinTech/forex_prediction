"""
generate_stock_factors.py

Reads X_train / X_val / X_test from a source .npz (OHLCV, shape [N, 128, 5]).
Generates 6 technical factors ONE AT A TIME, split by split, chunk by chunk.
Each factor column is accumulated into a disk-backed memmap so RAM stays low.
Final output: X shape [N, 96, 11]  (OHLCV + 6 factors, last-96 crop).

Factor order (appended after OHLCV channels 0-4):
  5  macd
  6  rsi
  7  bb_width
  8  log_ret
  9  log_vol_ret
  10 vol_ma_ratio

Usage
-----
python3 generate_stock_factors.py                          # defaults
python3 generate_stock_factors.py --chunk-size 10000       # lower RAM
python3 generate_stock_factors.py --tmp-dir /fast/ssd      # custom temp dir
"""

import os
import sys
import time
import argparse
import shutil
import tempfile
import zipfile

import numpy as np
from numpy.lib.format import write_array


FACTOR_NAMES = ["macd", "rsi", "bb_width", "log_ret", "log_vol_ret", "vol_ma_ratio"]


# ─────────────────────────────────────────────────────────────
# Progress bar
# ─────────────────────────────────────────────────────────────

class Progress:
    def __init__(self, total: int, prefix: str = ""):
        self.total  = total
        self.prefix = prefix
        self.t0     = time.time()
        self._last  = -1

    def update(self, done: int) -> None:
        pct = int(100.0 * done / self.total)
        if pct == self._last:
            return
        self._last = pct
        elapsed = time.time() - self.t0
        eta_str = ""
        if done > 0:
            eta     = elapsed * (self.total - done) / done
            eta_str = f"  ETA {eta:5.0f}s"
        bar = "#" * (pct // 2) + "." * (50 - pct // 2)
        print(f"\r  {self.prefix}  [{bar}] {pct:3d}%  "
              f"{done:>12,}/{self.total:,}{eta_str}",
              end="", flush=True)

    def done(self) -> None:
        elapsed = time.time() - self.t0
        bar = "#" * 50
        print(f"\r  {self.prefix}  [{bar}] 100%  "
              f"{self.total:>12,}/{self.total:,}  {elapsed:.1f}s",
              flush=True)


# ─────────────────────────────────────────────────────────────
# Math helpers  (all operate on [B, T] float32 arrays)
# ─────────────────────────────────────────────────────────────

def ema_2d(x: np.ndarray, span: int = None, alpha: float = None) -> np.ndarray:
    if alpha is None:
        alpha = 2.0 / (span + 1.0)
    y = np.empty_like(x, dtype=np.float32)
    y[:, 0] = x[:, 0]
    a, b = float(alpha), float(1.0 - alpha)
    for t in range(1, x.shape[1]):
        y[:, t] = a * x[:, t] + b * y[:, t - 1]
    return y


def rolling_mean_std(x: np.ndarray, window: int):
    n, l = x.shape
    mean = np.zeros((n, l), dtype=np.float32)
    std  = np.zeros((n, l), dtype=np.float32)
    if l < window:
        return mean, std
    x64 = x.astype(np.float64, copy=False)
    cs  = np.cumsum(x64, axis=1)
    cs2 = np.cumsum(x64 * x64, axis=1)
    pad = np.zeros((n, 1), dtype=np.float64)
    s   = cs[:, window - 1:] - np.concatenate([pad, cs[:, :l - window]], axis=1)
    s2  = cs2[:, window - 1:] - np.concatenate([pad, cs2[:, :l - window]], axis=1)
    m   = s / window
    var = np.maximum(s2 / window - m * m, 0.0)
    mean[:, window - 1:] = m.astype(np.float32)
    std[:, window - 1:]  = np.sqrt(var).astype(np.float32)
    return mean, std


# ─────────────────────────────────────────────────────────────
# Single factor, single chunk  ->  [B, T]
# ─────────────────────────────────────────────────────────────

def compute_one_factor(factor: str, x_chunk: np.ndarray) -> np.ndarray:
    eps    = 1e-8
    close  = np.ascontiguousarray(x_chunk[:, :, 3], dtype=np.float32)
    volume = (np.ascontiguousarray(x_chunk[:, :, 4], dtype=np.float32)
              if x_chunk.shape[2] >= 5 else np.ones_like(close))

    if factor == "macd":
        result = ema_2d(close, span=12) - ema_2d(close, span=26)

    elif factor == "rsi":
        delta = np.zeros_like(close)
        delta[:, 1:] = close[:, 1:] - close[:, :-1]
        gain     = np.maximum(delta, 0.0)
        loss     = np.abs(np.minimum(delta, 0.0))
        avg_gain = ema_2d(gain, alpha=1.0 / 14.0)
        avg_loss = ema_2d(loss, alpha=1.0 / 14.0)
        rs       = avg_gain / (avg_loss + 1e-10)
        result   = (100.0 - (100.0 / (1.0 + rs))) / 100.0

    elif factor == "bb_width":
        bb_mean, bb_std = rolling_mean_std(close, window=20)
        result = (4.0 * bb_std) / (bb_mean + eps)
        result[:, :19] = 0.0

    elif factor == "log_ret":
        result = np.zeros_like(close)
        prev   = np.where(close[:, :-1] > 0, close[:, :-1], eps)
        result[:, 1:] = np.log(np.maximum(close[:, 1:], eps) / prev)

    elif factor == "log_vol_ret":
        safe_vol = np.where(volume > 0, volume, 1.0).astype(np.float32)
        result   = np.zeros_like(close)
        result[:, 1:] = np.log(safe_vol[:, 1:] / (safe_vol[:, :-1] + eps))

    elif factor == "vol_ma_ratio":
        safe_vol = np.where(volume > 0, volume, 1.0).astype(np.float32)
        vol_mean, _ = rolling_mean_std(safe_vol, window=20)
        result = np.ones_like(close)
        result[:, 19:] = safe_vol[:, 19:] / (vol_mean[:, 19:] + eps)

    else:
        raise ValueError(f"Unknown factor: {factor}")

    np.nan_to_num(result, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return result.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Fill one factor column into an existing memmap
# ─────────────────────────────────────────────────────────────

def fill_factor_column(
    factor: str,
    src_X: np.ndarray,      # [N, T, C] read-only (lazy)
    out_mm: np.ndarray,     # [N, T, C+F] writable memmap
    col_idx: int,
    chunk_size: int,
) -> None:
    n    = src_X.shape[0]
    prog = Progress(n, prefix=f"{factor:<14}")
    for start in range(0, n, chunk_size):
        end   = min(start + chunk_size, n)
        chunk = src_X[start:end].astype(np.float32, copy=False)
        col   = compute_one_factor(factor, chunk)   # [B, T]
        out_mm[start:end, :, col_idx] = col
        out_mm.flush()
        prog.update(end)
    prog.done()


# ─────────────────────────────────────────────────────────────
# NPZ write helper
# ─────────────────────────────────────────────────────────────

def write_npz_key(zf: zipfile.ZipFile, key: str, arr: np.ndarray) -> None:
    with zf.open(f"{key}.npy", mode="w", force_zip64=True) as f:
        write_array(f, np.asanyarray(arr), allow_pickle=False)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate stock technical factors one-by-one with low RAM usage."
    )
    parser.add_argument(
        "--input",
        default="/home/corelabtq/Desktop/Research/forex/fin_factor/stock_atr_by_time.npz",
    )
    parser.add_argument(
        "--output",
        default="/home/corelabtq/Desktop/Research/forex/fin_factor/stock_atr_by_time_factors96.npz",
    )
    parser.add_argument("--crop",       type=int, default=96,
                        help="Keep last N timesteps (default: 96)")
    parser.add_argument("--chunk-size", type=int, default=20_000,
                        help="Samples per processing chunk (lower = less RAM, default: 20000)")
    parser.add_argument("--tmp-dir",    default=None,
                        help="Directory for temporary memmaps (needs ~(N*T*11*4) bytes free per split)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"[ERROR] Input not found: {args.input}")
    if not zipfile.is_zipfile(args.input):
        sys.exit(
            "[ERROR] Input NPZ is corrupted or incomplete: "
            f"{args.input}\n"
            "        It is missing a valid ZIP central directory, so numpy cannot open it.\n"
            "        Please regenerate a clean stock_atr_by_time.npz first, then rerun this script."
        )
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        sys.exit("[ERROR] --output must be different from --input")

    tmp_dir = tempfile.mkdtemp(dir=args.tmp_dir)
    print("=" * 65)
    print(f"  Input      : {args.input}")
    print(f"  Output     : {args.output}")
    print(f"  Crop       : last {args.crop} timesteps")
    print(f"  Chunk size : {args.chunk_size:,}")
    print(f"  Tmp dir    : {tmp_dir}")
    print("=" * 65)

    splits        = ["train", "val", "test"]
    full_mmmaps   = {}   # split -> writable memmap [N, T, 11]
    mm_paths      = {}

    try:
        src = np.load(args.input, allow_pickle=True)

        # ── Step 1: create full memmaps, copy OHLCV ───────────────
        print(f"\n{'─'*65}")
        print("[Step 1/3]  Allocate memmaps and copy OHLCV channels")
        print(f"{'─'*65}")
        for split in splits:
            key = f"X_{split}"
            if key not in src:
                print(f"  skip: {key} not in source")
                continue
            src_X   = src[key]
            n, T, C = src_X.shape
            out_C   = C + len(FACTOR_NAMES)
            gb      = n * T * out_C * 4 / 1024 ** 3
            print(f"\n  {key}  {src_X.shape}  ->  [{n}, {T}, {out_C}]  ({gb:.2f} GB temp)")
            mm_path = os.path.join(tmp_dir, f"{split}.mmap")
            mm_paths[split] = mm_path
            mm = np.memmap(mm_path, dtype=np.float32, mode='w+', shape=(n, T, out_C))
            prog = Progress(n, prefix="copy OHLCV    ")
            for start in range(0, n, args.chunk_size):
                end = min(start + args.chunk_size, n)
                mm[start:end, :, :C] = src_X[start:end].astype(np.float32, copy=False)
                mm.flush()
                prog.update(end)
            prog.done()
            full_mmmaps[split] = mm

        # ── Step 2: generate factors one at a time ────────────────
        total = len(FACTOR_NAMES) * len(full_mmmaps)
        done  = 0
        print(f"\n{'─'*65}")
        print(f"[Step 2/3]  Generate {len(FACTOR_NAMES)} factors x {len(full_mmmaps)} splits = {total} jobs")
        print(f"{'─'*65}")
        for fi, factor in enumerate(FACTOR_NAMES):
            col_idx = 5 + fi
            for split in splits:
                if split not in full_mmmaps:
                    continue
                done += 1
                src_X = src[f"X_{split}"]
                print(f"\n  [{done:>2}/{total}]  factor='{factor}'  split={split}  col={col_idx}")
                fill_factor_column(
                    factor     = factor,
                    src_X      = src_X,
                    out_mm     = full_mmmaps[split],
                    col_idx    = col_idx,
                    chunk_size = args.chunk_size,
                )

        # ── Step 3: write output NPZ (streaming from memmaps) ─────
        print(f"\n{'─'*65}")
        print(f"[Step 3/3]  Write output NPZ")
        print(f"{'─'*65}")
        skip_x_keys   = {f"X_{s}" for s in splits}
        feature_names = np.array(
            ["open", "high", "low", "close", "volume"] + FACTOR_NAMES,
            dtype=object,
        )

        with zipfile.ZipFile(
            args.output, mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=4,
            allowZip64=True,
        ) as zf:

            for split in splits:
                if split not in full_mmmaps:
                    continue
                key  = f"X_{split}"
                mm   = full_mmmaps[split]
                crp  = mm[:, -args.crop:, :]      # [N, crop, 11]  (view, not copy)
                print(f"\n  [{key}]  {crp.shape}  writing ...", flush=True)
                t0 = time.time()
                write_npz_key(zf, key, crp)
                print(f"    -> done in {time.time()-t0:.1f}s", flush=True)

            for k in src.files:
                if k in skip_x_keys:
                    continue
                print(f"  [{k}]  copying ...", flush=True)
                write_npz_key(zf, k, src[k])

            write_npz_key(zf, "feature_names", feature_names)
            write_npz_key(zf, "crop_len",      np.int32(args.crop))

        out_gb = os.path.getsize(args.output) / 1024 ** 3
        print(f"\n{'='*65}")
        print(f"  Finished!")
        print(f"  Output : {args.output}")
        print(f"  Size   : {out_gb:.2f} GB")
        print(f"{'='*65}")

    finally:
        try:
            src.close()
        except Exception:
            pass
        print(f"\nCleaning up temp dir: {tmp_dir}")
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()