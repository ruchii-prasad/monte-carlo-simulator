"""Command-line entry point.

Examples
--------
    python simulate.py --s0 100 --mu 0.08 --sigma 0.2 --T 1 --paths 10000
    python simulate.py --csv prices.csv --T 0.5 --plot out.png
    python simulate.py --s0 100 --sigma 0.2 --r 0.05 --T 1 --strike 105
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from mcsim import black_scholes, estimate_params, mc_price, simulate_gbm, summarize


def load_prices(path: str) -> np.ndarray:
    """Read the last numeric column of a CSV (e.g. Close / Adj Close)."""
    prices = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            for cell in reversed(row):
                try:
                    prices.append(float(cell))
                    break
                except ValueError:
                    continue
    return np.array(prices)


def plot(paths: np.ndarray, T: float, out: str, n_show: int = 100) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.linspace(0, T, paths.shape[1])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2, 1]})
    ax1.plot(t, paths[:n_show].T, lw=0.6, alpha=0.5)
    ax1.plot(t, paths.mean(axis=0), color="black", lw=2, label="mean path")
    ax1.set(xlabel="Time (years)", ylabel="Price", title=f"GBM paths ({n_show} of {len(paths)})")
    ax1.legend()
    ax2.hist(paths[:, -1], bins=60, color="steelblue", edgecolor="white")
    ax2.axvline(paths[0, 0], color="red", ls="--", label="S0")
    ax2.set(xlabel="Terminal price", title="Terminal distribution")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Plot saved to {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Monte Carlo GBM stock price simulator")
    p.add_argument("--s0", type=float, help="initial price (defaults to last CSV price)")
    p.add_argument("--mu", type=float, default=0.08, help="annual drift")
    p.add_argument("--sigma", type=float, default=0.20, help="annual volatility")
    p.add_argument("--csv", help="historical prices CSV; estimates s0, mu, sigma")
    p.add_argument("--T", type=float, default=1.0, help="horizon in years")
    p.add_argument("--steps", type=int, default=252)
    p.add_argument("--paths", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--antithetic", action="store_true")
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--plot", help="save chart to this PNG path")
    p.add_argument("--strike", type=float, help="also price a European call and put")
    p.add_argument("--r", type=float, default=0.05, help="risk-free rate for option pricing")
    a = p.parse_args()

    mu, sigma, s0 = a.mu, a.sigma, a.s0
    if a.csv:
        prices = load_prices(a.csv)
        mu, sigma = estimate_params(prices)
        s0 = s0 or float(prices[-1])
        print(f"Estimated from {len(prices)} prices: mu={mu:.4f}, sigma={sigma:.4f}")
    if s0 is None:
        s0 = 100.0

    paths = simulate_gbm(s0, mu, sigma, a.T, a.steps, a.paths, a.seed, a.antithetic)
    stats = summarize(paths, a.confidence)

    print(f"\nS0={s0:.2f}  mu={mu:.4f}  sigma={sigma:.4f}  T={a.T}  paths={a.paths}")
    print(f"Theoretical E[S_T]   : {s0 * np.exp(mu * a.T):.4f}")
    for k, v in stats.items():
        label = f"{k} ({a.confidence:.0%})" if k in ("VaR", "CVaR") else k
        fmt = f"{v:.2%}" if k in ("prob_profit", "VaR", "CVaR") else f"{v:.4f}"
        print(f"{label:<21}: {fmt}")

    if a.strike:
        print(f"\nEuropean options  K={a.strike}  r={a.r}")
        for kind in ("call", "put"):
            price, se = mc_price(s0, a.strike, a.r, sigma, a.T, kind, a.paths * 10, a.seed)
            bs = black_scholes(s0, a.strike, a.r, sigma, a.T, kind)
            print(f"  {kind:<4}: MC {price:.4f} ± {1.96 * se:.4f} (95% CI) | Black-Scholes {bs:.4f}")

    if a.plot:
        plot(paths, a.T, a.plot)


if __name__ == "__main__":
    main()
