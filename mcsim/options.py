"""European option pricing: Monte Carlo vs. the Black-Scholes closed form."""

from __future__ import annotations

from math import erf, exp, log, sqrt

import numpy as np


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def black_scholes(s0: float, K: float, r: float, sigma: float, T: float, kind: str = "call") -> float:
    """Black-Scholes price of a European call or put (no dividends)."""
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    if sigma == 0:
        fwd_payoff = max(s0 - K * exp(-r * T), 0.0)
        return fwd_payoff if kind == "call" else max(K * exp(-r * T) - s0, 0.0)
    d1 = (log(s0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if kind == "call":
        return s0 * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)
    return K * exp(-r * T) * _norm_cdf(-d2) - s0 * _norm_cdf(-d1)


def mc_price(
    s0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    kind: str = "call",
    n_paths: int = 100_000,
    seed: int | None = None,
    antithetic: bool = True,
) -> tuple[float, float]:
    """Monte Carlo price of a European option under the risk-neutral measure
    (drift = r). Only the terminal price matters, so we sample it in one step.

    Returns (price, standard_error)."""
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    rng = np.random.default_rng(seed)

    if antithetic:
        n_pairs = (n_paths + 1) // 2
        z = rng.standard_normal(n_pairs)
        z = np.concatenate([z, -z])
    else:
        z = rng.standard_normal(n_paths)

    sT = s0 * np.exp((r - 0.5 * sigma**2) * T + sigma * sqrt(T) * z)
    payoff = np.maximum(sT - K, 0.0) if kind == "call" else np.maximum(K - sT, 0.0)
    disc = exp(-r * T) * payoff

    if antithetic:
        # Antithetic pairs are not independent; average each pair first so the
        # standard error is computed from i.i.d. samples.
        pair_means = 0.5 * (disc[:n_pairs] + disc[n_pairs:])
        return float(pair_means.mean()), float(pair_means.std(ddof=1) / sqrt(n_pairs))
    return float(disc.mean()), float(disc.std(ddof=1) / sqrt(len(disc)))
