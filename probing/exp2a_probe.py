import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from probing.probing_common import (
    ensure_out_dir,
    extract_embeddings,
    get_default_parser,
    load_model,
    load_split,
)


def hurst_exponent(series: np.ndarray, max_lag: int = 20) -> float:
    x = np.asarray(series, dtype=np.float64)
    lags = range(2, max_lag + 1)
    tau = [np.std(x[lag:] - x[:-lag]) for lag in lags]
    tau = np.array(tau, dtype=np.float64)
    tau = np.where(tau <= 1e-12, 1e-12, tau)
    poly = np.polyfit(np.log(np.array(list(lags))), np.log(tau), 1)
    return float(poly[0] * 2.0)


def build_probe_labels(X: np.ndarray):
    close = X[:, :, 3]
    high = X[:, :, 1]
    low = X[:, :, 2]

    # realized volatility on last 20 bars
    ret = np.diff(np.log(np.clip(close, 1e-12, None)), axis=1)
    rv = np.std(ret[:, -20:], axis=1)
    rv_th = np.percentile(rv, 75)
    y_vol = (rv > rv_th).astype(int)

    # trending via Hurst exponent
    hurst_vals = np.array([hurst_exponent(c) for c in close], dtype=np.float64)
    y_trend = (hurst_vals > 0.6).astype(int)

    # spread day using last bar (H-L)/C
    spread = (high[:, -1] - low[:, -1]) / np.clip(close[:, -1], 1e-12, None)
    spread_th = np.median(spread)
    y_spread = (spread > spread_th).astype(int)

    return {
        "high_volatility": y_vol,
        "trending": y_trend,
        "high_spread_day": y_spread,
    }


def run_probe_auc(emb: np.ndarray, y: np.ndarray, seed: int) -> float:
    X_train, X_test, y_train, y_test = train_test_split(
        emb, y, test_size=0.3, random_state=seed, stratify=y
    )
    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(X_train, y_train)
    p = clf.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, p))


def main():
    parser = get_default_parser("EXP-2a Financial Concept Probing")
    args = parser.parse_args()

    out_dir = ensure_out_dir(args.out_dir)

    X, _, _, _, _ = load_split(args.data, split=args.split, max_samples=args.max_samples, seed=args.seed)

    model_r1 = load_model(args.r1, device=args.device)
    model_r3 = load_model(args.r3, device=args.device)

    emb_r1 = extract_embeddings(model_r1, X, batch_size=args.batch_size, device=args.device)
    emb_r3 = extract_embeddings(model_r3, X, batch_size=args.batch_size, device=args.device)

    labels = build_probe_labels(X)

    result = {}
    for concept, y in labels.items():
        auc_r1 = run_probe_auc(emb_r1, y, args.seed)
        auc_r3 = run_probe_auc(emb_r3, y, args.seed)
        result[concept] = {
            "auc_r1": auc_r1,
            "auc_r3": auc_r3,
            "delta_auc": auc_r1 - auc_r3,
            "positive_ratio": float(np.mean(y)),
        }

    out_file = Path(out_dir) / "exp2a_probe_results.json"
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("Saved:", out_file)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
