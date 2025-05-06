# Understanding GARCH(1,1) Models for Volatility Analysis

## GARCH Model Theory

### The Fundamental Problem
Financial asset returns often exhibit:
- **Volatility clustering**: Periods of high volatility tend to cluster together
- **Mean-reverting volatility**: Volatility eventually returns to a long-term average
- **Leptokurtic distribution**: Returns have "fat tails" compared to normal distributions

Standard models with constant volatility fail to capture these realities, which is why GARCH models were developed.

### GARCH(1,1) Mathematical Framework

The GARCH(1,1) model is defined by three key equations:

1. **Return equation**: 
   r_t = h_t × ε_t

2. **Volatility evolution**:
   h_t² = α₀ + α₁r_{t-1}² + β₁h_{t-1}²

3. **Innovation distribution**:
   ε_t ~ N(0,1)

Where:
- r_t: Return at time t
- h_t: Conditional volatility at time t
- α₀: Baseline volatility constant (> 0)
- α₁: Weight given to recent squared returns (≥ 0)
- β₁: Persistence of volatility (≥ 0)
- ε_t: Random innovation from standard normal distribution

The constraints α₁ + β₁ < 1 ensure stationarity of the process.

### Key Interpretations

#### Equilibrium Volatility
The long-term volatility level that the system converges to is:
- h² = α₀/(1-β₁) when α₁ = 0
- h² = α₀/(1-α₁-β₁) for the complete model

#### Parameter Meanings
- α₀: Minimum level of volatility regardless of market conditions
- α₁: Reaction to new market information/shocks
- β₁: Memory of the model (persistence of past volatility)

#### Rewriting for Intuition
The volatility equation can be rewritten as:
h_t² - h² = β₁(h_{t-1}² - h²) + α₁(r_{t-1}² - h²)

Which shows that:
1. Volatility tends to revert to h² (equilibrium)
2. The reversion speed depends on β₁
3. High returns shock volatility away from equilibrium

### Special Cases
- When α₁ = 0: Volatility follows a simple autoregressive process
- When β₁ = 0: Volatility depends only on the most recent return
- When α₁ + β₁ approaches 1: Shocks to volatility are very persistent (long memory)

## Implementation in Our Script

### The Estimation Process
1. **Maximum Likelihood Estimation**: The script finds optimal parameters by maximizing the likelihood of observing the Bitcoin returns given the GARCH model
2. **Log-likelihood function**: Uses the probability density of returns under conditional normality
3. **Optimization**: Uses L-BFGS-B algorithm with parameter constraints

### Volatility Computation
Once parameters are estimated, volatility is computed using equation (2):
1. Start with an initial variance estimate
2. Iteratively compute each day's volatility using previous day's information

### Volatility Forecasting
For future periods where returns aren't yet known:
1. Start with the most recent volatility estimate
2. For longer horizons, the equation simplifies to:
   h_t+k² = α₀ + (α₁+β₁)h_t+k-1²
3. As k increases, volatility approaches the equilibrium value

### Simulation Insights
The script simulates multiple potential paths by:
1. Generating random ε_t from N(0,1)
2. Computing future returns and volatilities based on the model equations
3. Showing how random shocks affect the volatility path

## Mathematical Properties of GARCH

### Volatility Memory
The impact of a shock decays at rate β₁:
- High β₁ (>0.9): Very persistent volatility (common in cryptocurrency)
- Low β₁ (<0.7): Quick mean reversion

### Volatility Clustering
The α₁ parameter controls how strongly recent return shocks affect future volatility:
- High α₁: Strong reaction to recent market events
- Low α₁: More stable volatility even after large price moves

### Half-Life of Volatility Shocks
The time it takes for a volatility shock to decay by half:
- Half-life = log(0.5)/log(β₁)

## Practical Applications

### Risk Management
- VaR (Value at Risk) calculation using time-varying volatility
- Dynamic position sizing based on forecasted volatility

### Option Pricing
- More accurate pricing by using GARCH volatility forecasts
- Volatility smile modeling

### Trading Strategies
- Volatility breakout systems
- Mean-reversion during high volatility regimes

## Limitations and Extensions

### GARCH(1,1) Limitations
- Assumes symmetrical response to positive and negative returns
- Normal distribution assumption may underestimate extreme events
- Single time scale (daily in our implementation)

### Extensions
- EGARCH: Captures asymmetric volatility response
- GJR-GARCH: Models leverage effects
- GARCH-M: Includes risk premium in return equation
- t-GARCH: Uses Student's t-distribution for fat tails