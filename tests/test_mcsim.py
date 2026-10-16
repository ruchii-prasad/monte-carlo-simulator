import numpy as np
import pytest

from mcsim import black_scholes, estimate_params, mc_price, simulate_gbm, summarize


def test_shape_and_start():
    paths = simulate_gbm(100, 0.05, 0.2, 1.0, 50, 200, seed=0)
    assert paths.shape == (200, 51)
    assert np.all(paths[:, 0] == 100)
    assert np.all(paths > 0)


def test_reproducible_with_seed():
    a = simulate_gbm(100, 0.05, 0.2, 1.0, 10, 100, seed=7)
    b = simulate_gbm(100, 0.05, 0.2, 1.0, 10, 100, seed=7)
    assert np.array_equal(a, b)


def test_mean_converges_to_theory():
    # E[S_T] = S0 * exp(mu * T); fails if the -sigma^2/2 drift correction is wrong
    s0, mu, sigma, T = 100, 0.1, 0.3, 2.0
    paths = simulate_gbm(s0, mu, sigma, T, 1, 400_000, seed=1)
    expected = s0 * np.exp(mu * T)
    se = paths[:, -1].std() / np.sqrt(len(paths))
    assert abs(paths[:, -1].mean() - expected) < 4 * se


def test_log_variance_matches_theory():
    s0, mu, sigma, T = 100, 0.05, 0.25, 1.5
    paths = simulate_gbm(s0, mu, sigma, T, 100, 100_000, seed=2)
    log_ret = np.log(paths[:, -1] / s0)
    assert log_ret.var() == pytest.approx(sigma**2 * T, rel=0.02)
    assert log_ret.mean() == pytest.approx((mu - 0.5 * sigma**2) * T, abs=0.01)


def test_zero_vol_is_deterministic():
    paths = simulate_gbm(100, 0.05, 0.0, 1.0, 12, 5, seed=0)
    assert np.allclose(paths[:, -1], 100 * np.exp(0.05))


def test_antithetic():
    paths = simulate_gbm(100, 0.05, 0.2, 1.0, 10, 100, seed=0, antithetic=True)
    log_ret = np.log(paths[:, -1] / 100)
    drift = (0.05 - 0.02) * 1.0
    assert np.allclose(log_ret[:50] - drift, -(log_ret[50:] - drift))
    with pytest.raises(ValueError):
        simulate_gbm(100, 0.05, 0.2, 1.0, 10, 99, antithetic=True)


@pytest.mark.parametrize("kwargs", [dict(s0=0), dict(sigma=-0.1), dict(T=0), dict(n_steps=0)])
def test_input_validation(kwargs):
    args = dict(s0=100, mu=0.05, sigma=0.2, T=1.0, n_steps=10, n_paths=10)
    args.update(kwargs)
    with pytest.raises(ValueError):
        simulate_gbm(**args)


def test_summary_risk_metrics():
    paths = simulate_gbm(100, 0.05, 0.3, 1.0, 1, 50_000, seed=3)
    s = summarize(paths, 0.95)
    assert s["CVaR"] >= s["VaR"]
    assert s["p5"] < s["median"] < s["p95"]
    assert 0 < s["prob_profit"] < 1


def test_black_scholes_known_value_and_parity():
    # Hull textbook example: S=42, K=40, r=0.1, sigma=0.2, T=0.5 -> call 4.76, put 0.81
    assert black_scholes(42, 40, 0.1, 0.2, 0.5, "call") == pytest.approx(4.76, abs=0.01)
    assert black_scholes(42, 40, 0.1, 0.2, 0.5, "put") == pytest.approx(0.81, abs=0.01)
    c = black_scholes(100, 105, 0.05, 0.2, 1, "call")
    p = black_scholes(100, 105, 0.05, 0.2, 1, "put")
    assert c - p == pytest.approx(100 - 105 * np.exp(-0.05), abs=1e-10)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("antithetic", [True, False])
def test_mc_matches_black_scholes(kind, antithetic):
    price, se = mc_price(100, 105, 0.05, 0.2, 1.0, kind, 400_000, seed=4, antithetic=antithetic)
    bs = black_scholes(100, 105, 0.05, 0.2, 1.0, kind)
    assert abs(price - bs) < 4 * se


def test_estimate_params_recovers_inputs():
    mu, sigma = 0.12, 0.3
    prices = simulate_gbm(100, mu, sigma, 40.0, 40 * 252, 1, seed=5)[0]
    mu_hat, sigma_hat = estimate_params(prices)
    assert sigma_hat == pytest.approx(sigma, rel=0.03)
    # drift is notoriously noisy: SE of mu_hat ~ sigma/sqrt(T) ~ 0.047 here
    assert abs(mu_hat - mu) < 0.15
