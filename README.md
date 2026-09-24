# European Option Pricing

Python project looking at different ways of pricing a European call option. I implemented the Black-Scholes formula, Monte Carlo simulation and the Cox-Ross-Rubinstein (CRR) binomial model.

I mainly used the project to look at how Monte Carlo pricing converges as the number of simulations increases and whether antithetic variates can reduce the variance of the estimate.

## Models

### Black-Scholes

The Black-Scholes formula gives an analytical price for a European call option. I use this as the benchmark for the numerical methods.

The parameters used in the example are:

| Parameter | Value |
| --- | ---: |
| Initial stock price | 100 |
| Strike price | 100 |
| Risk-free rate | 5% |
| Volatility | 20% |
| Time to maturity | 1 year |

### Monte Carlo

For each simulation, I generate a random value from a standard normal distribution and use it to simulate the stock price at maturity.

The call payoff is then calculated as:

`max(ST - K, 0)`

The option price is the average discounted payoff across all simulations.

I test different numbers of simulations to see how the estimate changes and how the standard error decreases.

### Antithetic Variates

I also implemented antithetic variates as a simple variance reduction method.

For every random value `Z`, I also use `-Z`. The two resulting payoffs are averaged before calculating the option price.

I compare standard Monte Carlo and antithetic Monte Carlo using standard error, confidence intervals and computational time.

### Cox-Ross-Rubinstein

The CRR model uses a binomial tree to approximate the stock price process.

I increase the number of steps in the tree and compare the resulting prices with the Black-Scholes benchmark.

## Project Structure

```text
option-pricing/
│
├── README.md
├── requirements.txt
├── option_pricing.py
└── run_analysis.py
