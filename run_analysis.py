import time

import pandas as pd
import matplotlib.pyplot as plt

from option_pricing import (
    black_scholes_call,
    monte_carlo_call,
    monte_carlo_antithetic_call,
    crr_call,
)


# ============================================================
# MODEL PARAMETERS
# ============================================================

S0 = 100       # Initial stock price
K = 100        # Strike price
r = 0.05       # Risk-free interest rate
sigma = 0.20   # Volatility
T = 1.0        # Time to maturity in years


# ============================================================
# 1. BLACK-SCHOLES BENCHMARK
# ============================================================

def run_black_scholes():

    price = black_scholes_call(
        S0,
        K,
        r,
        sigma,
        T
    )

    print("=" * 70)
    print("BLACK-SCHOLES")
    print("=" * 70)

    print(f"European call price: {price:.6f}")

    return price


# ============================================================
# 2. CRR BINOMIAL MODEL
# ============================================================

def run_crr_analysis(black_scholes_price):

    print()
    print("=" * 70)
    print("CRR BINOMIAL MODEL CONVERGENCE")
    print("=" * 70)

    steps = [
        10,
        25,
        50,
        100,
        250,
        500,
        1000
    ]

    results = []

    for n_steps in steps:

        start_time = time.perf_counter()

        price = crr_call(
            S0,
            K,
            r,
            sigma,
            T,
            n_steps
        )

        runtime = time.perf_counter() - start_time

        error = abs(
            price - black_scholes_price
        )

        results.append({
            "Steps": n_steps,
            "CRR Price": price,
            "Absolute Error": error,
            "Runtime (s)": runtime
        })

    df = pd.DataFrame(results)

    print(
        df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    return df


# ============================================================
# 3. MONTE CARLO CONVERGENCE
# ============================================================

def run_monte_carlo_convergence(black_scholes_price):

    print()
    print("=" * 70)
    print("MONTE CARLO CONVERGENCE")
    print("=" * 70)

    simulation_sizes = [
        100,
        1_000,
        10_000,
        100_000,
        1_000_000
    ]

    results = []

    for n_simulations in simulation_sizes:

        start_time = time.perf_counter()

        price, standard_error, confidence_interval = (
            monte_carlo_call(
                S0,
                K,
                r,
                sigma,
                T,
                n_simulations
            )
        )

        runtime = time.perf_counter() - start_time

        error = abs(
            price - black_scholes_price
        )

        results.append({
            "Simulations": n_simulations,
            "Price": price,
            "Absolute Error": error,
            "Standard Error": standard_error,
            "CI Lower": confidence_interval[0],
            "CI Upper": confidence_interval[1],
            "Runtime (s)": runtime
        })

    df = pd.DataFrame(results)

    print(
        df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    return df


# ============================================================
# 4. ANTITHETIC VARIATES
# ============================================================

def run_antithetic_analysis():

    print()
    print("=" * 70)
    print("STANDARD MONTE CARLO VS ANTITHETIC VARIATES")
    print("=" * 70)

    n_simulations = 100_000

    # -------------------------------
    # Standard Monte Carlo
    # -------------------------------

    start_time = time.perf_counter()

    standard_price, standard_se, standard_ci = (
        monte_carlo_call(
            S0,
            K,
            r,
            sigma,
            T,
            n_simulations
        )
    )

    standard_runtime = (
        time.perf_counter() - start_time
    )

    # -------------------------------
    # Antithetic Monte Carlo
    # -------------------------------

    start_time = time.perf_counter()

    antithetic_price, antithetic_se, antithetic_ci = (
        monte_carlo_antithetic_call(
            S0,
            K,
            r,
            sigma,
            T,
            n_simulations
        )
    )

    antithetic_runtime = (
        time.perf_counter() - start_time
    )

    # -------------------------------
    # Calculate improvements
    # -------------------------------

    variance_reduction = (
        1
        - (antithetic_se ** 2)
        / (standard_se ** 2)
    )

    standard_error_reduction = (
        1
        - antithetic_se / standard_se
    )

    # -------------------------------
    # Print results
    # -------------------------------

    print()
    print("Standard Monte Carlo")
    print(f"Price:          {standard_price:.6f}")
    print(f"Standard error: {standard_se:.6f}")
    print(
        f"95% CI:         "
        f"({standard_ci[0]:.6f}, "
        f"{standard_ci[1]:.6f})"
    )
    print(
        f"Runtime:        "
        f"{standard_runtime:.6f} seconds"
    )

    print()
    print("Antithetic Monte Carlo")
    print(f"Price:          {antithetic_price:.6f}")
    print(f"Standard error: {antithetic_se:.6f}")
    print(
        f"95% CI:         "
        f"({antithetic_ci[0]:.6f}, "
        f"{antithetic_ci[1]:.6f})"
    )
    print(
        f"Runtime:        "
        f"{antithetic_runtime:.6f} seconds"
    )

    print()
    print(
        f"Standard error reduction: "
        f"{standard_error_reduction:.2%}"
    )

    print(
        f"Variance reduction: "
        f"{variance_reduction:.2%}"
    )

    return {
        "standard_price": standard_price,
        "standard_se": standard_se,
        "antithetic_price": antithetic_price,
        "antithetic_se": antithetic_se,
        "standard_error_reduction": standard_error_reduction,
        "variance_reduction": variance_reduction
    }


# ============================================================
# 5. MONTE CARLO CONVERGENCE PLOT
# ============================================================

def plot_monte_carlo_convergence(
    monte_carlo_results,
    black_scholes_price
):

    plt.figure(figsize=(8, 5))

    plt.plot(
        monte_carlo_results["Simulations"],
        monte_carlo_results["Price"],
        marker="o",
        label="Monte Carlo"
    )

    plt.axhline(
        black_scholes_price,
        linestyle="--",
        label="Black-Scholes"
    )

    plt.xscale("log")

    plt.xlabel("Number of simulations")
    plt.ylabel("Option price")

    plt.title(
        "Monte Carlo Convergence"
    )

    plt.legend()
    plt.tight_layout()

    plt.show()


# ============================================================
# 6. CRR CONVERGENCE PLOT
# ============================================================

def plot_crr_convergence(
    crr_results,
    black_scholes_price
):

    plt.figure(figsize=(8, 5))

    plt.plot(
        crr_results["Steps"],
        crr_results["CRR Price"],
        marker="o",
        label="CRR"
    )

    plt.axhline(
        black_scholes_price,
        linestyle="--",
        label="Black-Scholes"
    )

    plt.xlabel("Number of tree steps")
    plt.ylabel("Option price")

    plt.title(
        "CRR Convergence to Black-Scholes"
    )

    plt.legend()
    plt.tight_layout()

    plt.show()


# ============================================================
# 7. MAIN PROGRAM
# ============================================================

def main():

    print()
    print("EUROPEAN CALL OPTION PRICING")
    print()

    # Black-Scholes benchmark
    black_scholes_price = run_black_scholes()

    # CRR analysis
    crr_results = run_crr_analysis(
        black_scholes_price
    )

    # Monte Carlo convergence
    monte_carlo_results = (
        run_monte_carlo_convergence(
            black_scholes_price
        )
    )

    # Antithetic variates
    antithetic_results = (
        run_antithetic_analysis()
    )

    # Plots
    plot_monte_carlo_convergence(
        monte_carlo_results,
        black_scholes_price
    )

    plot_crr_convergence(
        crr_results,
        black_scholes_price
    )


if __name__ == "__main__":
    main()
