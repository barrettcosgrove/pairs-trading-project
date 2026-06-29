"""Inspect silhouette-driven K selection used by ``working_model`` clustering.

Run from the repo root (recommended):

    uv run python working_model/test_silhouette_scoring.py

Or from ``working_model/``:

    cd working_model
    uv run python test_silhouette_scoring.py

Synthetic data mimics correlated return structure so silhouette curves are
non-trivial. Use ``--quiet`` on ``cluster_stocks`` to suppress duplicate banners
when comparing scan vs full clustering.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Resolve imports whether launched from repo root or working_model/.
_WM_ROOT = Path(__file__).resolve().parent
if str(_WM_ROOT) not in sys.path:
    sys.path.insert(0, str(_WM_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from configuration import CONFIG  # noqa: E402
from pairs_strategy import calculate_returns, cluster_stocks, silhouette_scan_kmeans  # noqa: E402


def synthetic_returns_panel(
    n_stocks: int,
    n_days: int,
    *,
    seed: int,
    factor_dim: int = 3,
) -> pd.DataFrame:
    """Build a correlated simple-return panel with no trend (research demo).

    Args:
        n_stocks: Number of synthetic tickers.
        n_days: Number of business rows.
        seed: RNG seed for reproducibility.
        factor_dim: Number of orthogonal factors mixing into each name.

    Returns:
        Wide DataFrame of synthetic returns indexed by date.
    """
    rng = np.random.default_rng(seed)
    factors = rng.standard_normal((n_days, factor_dim))
    loadings = rng.standard_normal((n_stocks, factor_dim))
    idiosync = rng.standard_normal((n_days, n_stocks)) * 0.5
    rets_mx = factors @ loadings.T + idiosync
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    cols = [f"SYN{i:03d}" for i in range(n_stocks)]
    return pd.DataFrame(rets_mx, index=idx, columns=cols)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print silhouette sweep and run cluster_stocks on synthetic or CSV prices."
    )
    parser.add_argument(
        "--k-min",
        type=int,
        default=None,
        help=f"Minimum k for sweep (default: CONFIG.cluster_k_min={CONFIG.cluster_k_min}).",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=None,
        help=f"Maximum k for sweep (default: CONFIG.cluster_k_max={CONFIG.cluster_k_max}).",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=None,
        help=f"KMeans n_init (default: CONFIG.kmeans_n_init={CONFIG.kmeans_n_init}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=CONFIG.kmeans_random_seed,
        help="Random seed for synthetic data and KMeans.",
    )
    parser.add_argument(
        "--stocks",
        type=int,
        default=80,
        help="Synthetic universe size (ignored if --csv is set).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=252,
        help="Synthetic history length (ignored if --csv is set).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional CSV path with DatetimeIndex in first column and tickers as "
        "other columns (prices → returns computed via calculate_returns).",
    )
    parser.add_argument(
        "--no-cluster-stocks-demo",
        action="store_true",
        help="Only print silhouette_scan table; skip cluster_stocks().",
    )
    args = parser.parse_args()

    k_min = int(CONFIG.cluster_k_min if args.k_min is None else args.k_min)
    k_max = int(CONFIG.cluster_k_max if args.k_max is None else args.k_max)
    n_init = int(CONFIG.kmeans_n_init if args.n_init is None else args.n_init)

    if args.csv:
        raw = pd.read_csv(args.csv, index_col=0, parse_dates=True)
        panel = calculate_returns(raw)
    else:
        panel = synthetic_returns_panel(args.stocks, args.days, seed=args.seed)

    print(
        f"Panel: {panel.shape[1]} tickers × {panel.shape[0]} return rows "
        f"({panel.index.min().date()} .. {panel.index.max().date()})"
    )
    print("\n--- Silhouette sweep (standalone KMeans fits per k) ---")
    scan = silhouette_scan_kmeans(
        panel,
        k_min=k_min,
        k_max=k_max,
        n_init=n_init,
        random_state=args.seed,
    )
    if scan.empty:
        print("No valid k range (too few stocks for k_min / k_max).")
        return
    print(scan.to_string(index=False))

    ok = scan.loc[~scan["degenerate"] & scan["silhouette"].notna()]
    if not ok.empty:
        best_row = ok.loc[ok["silhouette"].idxmax()]
        print(
            f"\nBest k by silhouette (non-degenerate): k={int(best_row['k'])} "
            f"score={float(best_row['silhouette']):.4f}"
        )

    if args.no_cluster_stocks_demo:
        return

    print("\n--- cluster_stocks (silhouette k-selection on same panel) ---")
    clusters = cluster_stocks(
        panel,
        n_clusters=CONFIG.n_clusters,
        use_silhouette_k_selection=True,
        k_min=k_min,
        k_max=k_max,
        kmeans_n_init=n_init,
        random_state=args.seed,
        verbose=True,
    )
    vc = clusters.value_counts().sort_index()
    print("Cluster sizes:\n", vc.to_string())


if __name__ == "__main__":
    main()
