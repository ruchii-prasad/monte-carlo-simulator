"""Geometric Brownian Motion path simulation.

Under GBM, dS = mu*S*dt + sigma*S*dW, which has the exact solution

    S(t+dt) = S(t) * exp((mu - sigma^2/2) dt + sigma * sqrt(dt) * Z),  Z ~ N(0, 1)

We simulate in log space with this exact step, so there is no discretisation
error regardless of the number of time steps.
"""

from __future__ import annotations

import numpy as np


def simulate_gbm(
    s0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
    antithetic: bool = False,
) -> np.ndarray:
    """Simulate GBM price paths.

    Parameters
    ----------
    s0 : initial price (> 0)
    mu : annualised drift
    sigma : annualised volatility (>= 0)
    T : horizon in years (> 0)
    n_steps : number of time steps (>= 1)
    n_paths : number of paths (>= 1; must be even if antithetic)
    seed : RNG seed for reproducibility
    antithetic : use antithetic variates (pairs Z with -Z) for variance reduction

    Returns
    -------
    Array of shape (n_paths, n_steps + 1); column 0 is s0.
    """
    if s0 <= 0:
        raise ValueError("s0 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if T <= 0:
        raise ValueError("T must be positive")
    if n_steps < 1 or n_paths < 1:
        raise ValueError("n_steps and n_paths must be >= 1")
    if antithetic and n_paths % 2:
        raise ValueError("n_paths must be even when antithetic=True")

    rng = np.random.default_rng(seed)
    dt = T / n_steps

    if antithetic:
        half = rng.standard_normal((n_paths // 2, n_steps))
        z = np.vstack([half, -half])
    else:
        z = rng.standard_normal((n_paths, n_steps))

    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)], axis=1
    )
    return s0 * np.exp(log_paths)


def summarize(paths: np.ndarray, confidence: float = 0.95) -> dict:
    """Summary statistics of the terminal price distribution, plus VaR/CVaR of
    the return over the horizon (reported as positive loss fractions)."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    s0 = paths[0, 0]
    terminal = paths[:, -1]
    returns = terminal / s0 - 1.0

    var_threshold = np.quantile(returns, 1 - confidence)
    tail = returns[returns <= var_threshold]

    return {
        "mean": float(terminal.mean()),
        "median": float(np.median(terminal)),
        "std": float(terminal.std(ddof=1)) if len(terminal) > 1 else 0.0,
        "p5": float(np.quantile(terminal, 0.05)),
        "p95": float(np.quantile(terminal, 0.95)),
        "prob_profit": float((terminal > s0).mean()),
        "VaR": float(-var_threshold),
        "CVaR": float(-tail.mean()),
    }


def estimate_params(prices: np.ndarray, periods_per_year: int = 252) -> tuple[float, float]:
    """Estimate annualised (mu, sigma) from a historical price series using
    log returns. mu is the arithmetic drift of GBM, i.e. mean_log/dt + sigma^2/2."""
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1 or len(prices) < 3:
        raise ValueError("need a 1-D series of at least 3 prices")
    if np.any(prices <= 0):
        raise ValueError("prices must be positive")
    log_ret = np.diff(np.log(prices))
    dt = 1.0 / periods_per_year
    sigma = log_ret.std(ddof=1) / np.sqrt(dt)
    mu = log_ret.mean() / dt + 0.5 * sigma**2
    return float(mu), float(sigma)
