import numpy as np
import pytest

from mcsim import black_scholes, estimate_params, mc_price, simulate_gbm, summarize


def test_shape_and_start():
    paths = simulate_gbm(100, 0.05, 0.2, 1.0, 50, 200, seed=0)

    assert paths.shape == (200, 51)
    assert np.all(paths[:, 0] == 100)
    assert np.all(paths > 0)


def test_same_seed_gives_same_result():
    paths1 = simulate_gbm(100, 0.05, 0.2, 1.0, 10, 100, seed=7)
    paths2 = simulate_gbm(100, 0.05, 0.2, 1.0, 10, 100, seed=7)

    assert np.array_equal(paths1, paths2)


def test_average_price():
    s0, mu, sigma, T = 100, 0.1, 0.3, 2.0

    paths = simulate_gbm(s0, mu, sigma, T, 1, 400_000, seed=1)

    # The expected final price is S0 * exp(mu * T)
    expected = s0 * np.exp(mu * T)

    final_prices = paths[:, -1]
    se = final_prices.std() / np.sqrt(len(final_prices))

    assert abs(final_prices.mean() - expected) < 4 * se


def test_log_return():
    s0, mu, sigma, T = 100, 0.05, 0.25, 1.5

    paths = simulate_gbm(s0, mu, sigma, T, 100, 100_000, seed=2)

    log_returns = np.log(paths[:, -1] / s0)

    # Check if the variance and mean are close to the theoretical values
    assert log_returns.var() == pytest.approx(sigma**2 * T, rel=0.02)
    assert log_returns.mean() == pytest.approx(
        (mu - 0.5 * sigma**2) * T, abs=0.01
    )


def test_zero_volatility():
    paths = simulate_gbm(100, 0.05, 0.0, 1.0, 12, 5, seed=0)

    expected = 100 * np.exp(0.05)

    assert np.allclose(paths[:, -1], expected)


def test_antithetic_paths():
    paths = simulate_gbm(
        100, 0.05, 0.2, 1.0, 10, 100, seed=0, antithetic=True
    )

    log_returns = np.log(paths[:, -1] / 100)
    drift = (0.05 - 0.5 * 0.2**2) * 1.0

    # The second half should use the opposite random numbers
    assert np.allclose(
        log_returns[:50] - drift,
        -(log_returns[50:] - drift)
    )

    # An odd number of paths should not be allowed
    with pytest.raises(ValueError):
        simulate_gbm(
            100, 0.05, 0.2, 1.0, 10, 99, antithetic=True
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"s0": 0},
        {"sigma": -0.1},
        {"T": 0},
        {"n_steps": 0},
    ],
)
def test_invalid_inputs(kwargs):
    args = {
        "s0": 100,
        "mu": 0.05,
        "sigma": 0.2,
        "T": 1.0,
        "n_steps": 10,
        "n_paths": 10,
    }

    args.update(kwargs)

    with pytest.raises(ValueError):
        simulate_gbm(**args)


def test_risk_values():
    paths = simulate_gbm(100, 0.05, 0.3, 1.0, 1, 50_000, seed=3)

    result = summarize(paths, 0.95)

    assert result["CVaR"] >= result["VaR"]
    assert result["p5"] < result["median"] < result["p95"]
    assert 0 < result["prob_profit"] < 1


def test_black_scholes_prices():
    # Check the call and put prices using a known example
    call = black_scholes(42, 40, 0.1, 0.2, 0.5, "call")
    put = black_scholes(42, 40, 0.1, 0.2, 0.5, "put")

    assert call == pytest.approx(4.76, abs=0.01)
    assert put == pytest.approx(0.81, abs=0.01)

    # Check put-call parity
    call = black_scholes(100, 105, 0.05, 0.2, 1, "call")
    put = black_scholes(100, 105, 0.05, 0.2, 1, "put")

    assert call - put == pytest.approx(
        100 - 105 * np.exp(-0.05), abs=1e-10
    )


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("antithetic", [True, False])
def test_mc_price(kind, antithetic):
    price, se = mc_price(
        100, 105, 0.05, 0.2, 1.0, kind,
        400_000, seed=4, antithetic=antithetic
    )

    expected = black_scholes(100, 105, 0.05, 0.2, 1.0, kind)

    # Monte Carlo price should be close to the Black-Scholes price
    assert abs(price - expected) < 4 * se


def test_estimated_parameters():
    mu, sigma = 0.12, 0.3

    prices = simulate_gbm(
        100, mu, sigma, 40.0, 40 * 252, 1, seed=5
    )[0]

    mu_est, sigma_est = estimate_params(prices)

    assert sigma_est == pytest.approx(sigma, rel=0.03)

    # Estimating mu is less accurate, so allow a bigger difference
    assert abs(mu_est - mu) < 0.15
