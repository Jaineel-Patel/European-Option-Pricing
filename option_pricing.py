import math
import numpy as np
from scipy.stats import norm


def black_scholes_call(S0, K, r, sigma, T):
    """
    Price a European call option using the Black-Scholes formula.

    Parameters:
        S0    : initial stock price
        K     : strike price
        r     : continuously compounded risk-free rate
        sigma : volatility
        T     : time to maturity
    """

    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be positive.")

    if sigma <= 0 or T <= 0:
        raise ValueError("sigma and T must be positive.")

    d1 = (
        math.log(S0 / K)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * math.sqrt(T))

    d2 = d1 - sigma * math.sqrt(T)

    call_price = (
        S0 * norm.cdf(d1)
        - K * math.exp(-r * T) * norm.cdf(d2)
    )

    return call_price


def monte_carlo_call(
    S0,
    K,
    r,
    sigma,
    T,
    n_simulations,
    seed=7
):
    """
    Price a European call option using standard Monte Carlo simulation.

    Returns:
        price
        standard error
        95% confidence interval
    """

    if n_simulations <= 0:
        raise ValueError("Number of simulations must be positive.")

    rng = np.random.default_rng(seed)

    # Generate standard normal random variables
    Z = rng.standard_normal(n_simulations)

    # Simulate stock price at maturity
    ST = S0 * np.exp(
        (r - 0.5 * sigma**2) * T
        + sigma * math.sqrt(T) * Z
    )

    # Calculate option payoff
    payoffs = np.maximum(ST - K, 0)

    # Discount payoffs back to the present
    discounted_payoffs = (
        math.exp(-r * T) * payoffs
    )

    # Monte Carlo estimate
    price = np.mean(discounted_payoffs)

    # Standard error of the estimate
    standard_error = (
        np.std(discounted_payoffs, ddof=1)
        / math.sqrt(n_simulations)
    )

    # Approximate 95% confidence interval
    margin = 1.96 * standard_error

    confidence_interval = (
        price - margin,
        price + margin
    )

    return price, standard_error, confidence_interval


def monte_carlo_antithetic_call(
    S0,
    K,
    r,
    sigma,
    T,
    n_pairs,
    seed=7
):
    """
    Price a European call using Monte Carlo with
    antithetic variates.

    For every Z simulated, -Z is also used.
    """

    if n_pairs <= 0:
        raise ValueError("Number of pairs must be positive.")

    rng = np.random.default_rng(seed)

    # Generate random values
    Z = rng.standard_normal(n_pairs)

    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * math.sqrt(T)

    # Stock prices using Z
    ST_plus = S0 * np.exp(
        drift + diffusion * Z
    )

    # Stock prices using -Z
    ST_minus = S0 * np.exp(
        drift - diffusion * Z
    )

    # Option payoffs
    payoff_plus = np.maximum(ST_plus - K, 0)
    payoff_minus = np.maximum(ST_minus - K, 0)

    # Average each antithetic pair
    paired_payoffs = (
        payoff_plus + payoff_minus
    ) / 2

    # Discount to present value
    discounted_payoffs = (
        math.exp(-r * T) * paired_payoffs
    )

    # Monte Carlo estimate
    price = np.mean(discounted_payoffs)

    # Standard error
    standard_error = (
        np.std(discounted_payoffs, ddof=1)
        / math.sqrt(n_pairs)
    )

    # 95% confidence interval
    margin = 1.96 * standard_error

    confidence_interval = (
        price - margin,
        price + margin
    )

    return price, standard_error, confidence_interval


def crr_call(
    S0,
    K,
    r,
    sigma,
    T,
    n_steps
):
    """
    Price a European call option using
    the Cox-Ross-Rubinstein binomial model.
    """

    if n_steps <= 0:
        raise ValueError("Number of steps must be positive.")

    # Length of each time step
    dt = T / n_steps

    # Up and down factors
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u

    # Risk-neutral probability
    p = (
        math.exp(r * dt) - d
    ) / (u - d)

    if not 0 <= p <= 1:
        raise ValueError(
            "Risk-neutral probability is outside [0, 1]."
        )

    # Possible stock prices at maturity
    j = np.arange(n_steps + 1)

    stock_prices = (
        S0
        * (u ** j)
        * (d ** (n_steps - j))
    )

    # European call payoff at maturity
    option_values = np.maximum(
        stock_prices - K,
        0
    )

    # Discount backwards through the tree
    discount = math.exp(-r * dt)

    for _ in range(n_steps):

        option_values = discount * (
            p * option_values[1:]
            + (1 - p) * option_values[:-1]
        )

    return float(option_values[0])
