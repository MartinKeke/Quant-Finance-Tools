# Realized Volatility Calculation

This document explains the methodology used to calculate realized volatility from high-frequency price data.

## Overview

Realized volatility (RV) is a measure of the actual observed volatility of an asset over a specific time period, based on historical price data. Unlike implied volatility (derived from options prices), realized volatility is backward-looking and calculated directly from market prices.

## Calculation Methodology

The realized volatility calculation follows these steps:

### Step 1: Calculate Log Returns

For each time point t, the log return is calculated as:

```
r_t = ln(P_t / P_{t-1})
```

where:
- P_t is the price at time t
- P_{t-1} is the price at time t-1

In code:
```python
df['log_return'] = np.log(df['close'] / df['close'].shift(1))
```

Log returns are preferred over simple returns because they are additive over time and have better statistical properties for financial time series analysis.

### Step 2: Calculate Squared Returns

Each log return is squared:

```
r_t^2
```

In code:
```python
returns_df['squared_return'] = returns_df['log_return'] ** 2
```

Squared returns represent the variance contribution of each individual observation.

### Step 3: Sum Squared Returns Over a Window

The squared returns are summed over a specified window (e.g., 1440 minutes for daily RV):

```
σ² = Σ r_i^2
```

where the sum is taken over all observations i in the window.

In code:
```python
returns_df['rolling_variance'] = returns_df['squared_return'].rolling(window=window_minutes).sum()
```

This sum of squared returns provides an estimate of the variance for the period.

### Step 4: Annualize the Volatility

The final step converts the variance to standard deviation (by taking the square root) and annualizes the result:

```
σ_annual = σ × sqrt(N/T)
```

where:
- σ is the standard deviation (square root of variance)
- N is the number of periods in a year (525,600 minutes)
- T is the number of periods in your window (e.g., 1440 minutes)

In code:
```python
minutes_per_year = 365 * 24 * 60
returns_df['realized_vol'] = np.sqrt(returns_df['rolling_variance'] * (minutes_per_year / window_minutes))
```

## Window Periods

Common window periods for calculating realized volatility include:

- **Daily RV**: 1440 minutes (24 hours)
- **Weekly RV**: 10080 minutes (7 days)
- **Monthly RV**: 43200 minutes (30 days)

## Theoretical Foundation

This computation is based on the theory that the variance of returns over a time period can be estimated by summing the squared high-frequency returns within that period. This approach is grounded in stochastic calculus and the quadratic variation process.

The advantage of using high-frequency data (like 1-minute intervals) is that it provides a more accurate measurement of the true volatility compared to using just daily closing prices.

## Properties of Realized Volatility

- It's a backward-looking measure (historical)
- More data points (higher frequency) typically provide more accurate estimates
- It can capture intraday volatility patterns that daily measures miss
- It can be decomposed into continuous and jump components for deeper analysis
