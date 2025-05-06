# Understanding EWMA and Mean-Reverting EWMA Volatility Models

This README explains the concepts and implementation of two volatility models:
1. Standard Exponentially Weighted Moving Average (EWMA)
2. Mean-Reverting EWMA

## Basic Concepts

### What is Volatility?
In finance, volatility measures how much an asset's price fluctuates over time. High volatility means large price swings (more risk), while low volatility means smoother price movements (less risk).

### Why Estimate Volatility?
- **Risk Management**: Measure potential losses
- **Option Pricing**: Options are more valuable when volatility is high
- **Trading Strategies**: Some strategies specifically target volatility patterns
- **Portfolio Construction**: Balancing risk across different assets

## The EWMA Model

### The Basic Idea
The standard EWMA model estimates volatility by taking a weighted average of past squared returns, giving more weight to recent observations and less weight to older ones.

### The Formula
```
σ²t = (1-K)r²t-1 + Kσ²t-1
```
Where:
- σ²t is the variance estimate at time t
- r²t-1 is the squared return at time t-1
- K is the decay factor (typically 0.94 for daily data)

### Key Properties
- **Simplicity**: Easy to implement and understand
- **Responsiveness**: Quickly adapts to new market conditions
- **No Mean Reversion**: Doesn't assume volatility returns to a long-term average
- **Memory Decay**: Older observations have exponentially less influence

### Limitations
- The model assumes volatility shocks persist indefinitely
- Can sometimes overreact to large market moves
- Doesn't capture the tendency of volatility to revert to a long-run average

## The Mean-Reverting EWMA Model

### The Basic Idea
The Mean-Reverting EWMA extends the standard model by adding a mechanism that pulls volatility estimates back toward a long-run average level over time.

### The State Equation
```
xt+1 = xt - λ(xt - μ) + τeεt+1
```
Where:
- xt is the true variance state at time t
- λ is the mean reversion rate
- μ is the long-run mean variance
- τe is the innovation noise scale
- εt+1 is a standard normal random variable

### The Observation Equation
```
yt+1 = xt+1 + τuηt+1
```
Where:
- yt+1 is the observed squared return
- τu is the measurement noise scale
- ηt+1 is a standard normal random variable

### The Optimal Estimator
```
x̂t+1|t = (1-K)x̂t|t-1 + Kyt
```
Where:
- K is the Kalman gain, optimally balancing new information and prior estimates

### Key Advantages
- **Mean Reversion**: Captures the tendency of volatility to return to normal levels
- **Noise Filtering**: Separates true volatility from measurement noise
- **Theoretical Foundation**: Based on state-space modeling principles
- **Long-run Stability**: Prevents unrealistic long-term forecasts

## Understanding the Implementation

### Key Parameters

#### For Standard EWMA:
- **decay_factor (K)**: Controls how quickly the influence of past returns decays
  - Higher values (closer to 1) = smoother estimates, slower reactions
  - Lower values (closer to 0) = more responsive to recent returns

#### For Mean-Reverting EWMA:
- **lambda_**: The mean reversion speed
  - Higher values = faster reversion to the long-run mean
  - Lower values = slower reversion, more persistence
- **mu**: The long-run variance level (estimated from data if not provided)
- **K**: The Kalman filter gain (calculated optimally if not provided)

### Volatility Forecasting

The two models forecast future volatility differently:
- **Standard EWMA**: Forecasts are constant at the last estimated value
- **Mean-Reverting EWMA**: Forecasts gradually converge to the long-run mean

### Implementation Notes

The code is structured into three main parts:
1. **Model Classes**: `EWMA` and `MeanRevertingEWMA` implement the core models
2. **Helper Functions**: For data loading, visualization, and analysis
3. **Main Analysis**: Demonstrates how to use the models with Bitcoin data

## Using the Code

1. Run the full analysis:
   ```python
   df, ewma_model, mr_ewma_model = analyze_bitcoin_volatility(file_path)
   ```

2. View the comparison between models:
   ```python
   plot_volatility_comparison(df['Return'].values, ewma_model, mr_ewma_model)
   ```

3. Generate volatility forecasts:
   ```python
   plot_forecast_comparison(ewma_model, mr_ewma_model, forecast_steps=100)
   ```

## Interpreting the Results

- **Volatility Comparison Plot**: Shows how each model responds to return shocks
- **Forecast Comparison Plot**: Shows how forecasts differ between models
- **Model Parameters**: Reveal the characteristics of each model
  - Large λ = fast mean reversion
  - Small λ = slow mean reversion
  - Long-run volatility (√μ) = where volatility tends to converge

## Relation Between GARCH and EWMA

As shown in the mathematical derivation, GARCH(1,1) can be viewed as "EWMA with an offset":

### GARCH(1,1) Model
```
h²t = α₀ + α₁r²t-1 + β₁h²t-1
```

### EWMA Model
```
σ²t = (1-K)r²t-1 + Kσ²t-1
```

These models are identical when:
- α₀ = 0
- α₁ = 1-K
- β₁ = K

The constant term α₀ in GARCH creates the offset that introduces mean reversion, allowing volatility to converge to a long-run level:
```
Long-run variance = α₀/(1-α₁-β₁)
```

This relationship helps us understand that GARCH and the Mean-Reverting EWMA are addressing similar issues but through different mathematical frameworks.

## Conclusion

The Mean-Reverting EWMA addresses key limitations of the standard EWMA by incorporating the empirical fact that volatility tends to return to normal levels over time. This makes it particularly useful for longer-term forecasting and risk management.

Both models are valuable tools in quantitative finance, with different strengths depending on the application. The standard EWMA is simpler and may be better for short-term forecasting, while the Mean-Reverting EWMA typically provides more realistic long-term forecasts.