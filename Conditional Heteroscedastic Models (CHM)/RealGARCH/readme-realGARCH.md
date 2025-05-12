# RealGARCH Model Implementation

## Overview
This project implements the RealGARCH(1,1) model introduced by Hansen et al. (2012), which combines the strengths of traditional GARCH models with realized volatility measures. The RealGARCH model improves volatility forecasting by incorporating high-frequency intraday return information.

## Model Specification

The RealGARCH(1,1) model is defined by three equations:

1. Return equation: 
   r_t = h_t^{1/2} ε_t

2. Conditional variance equation: 
   h_t^2 = α_0 + β_1 h_{t-1}^2 + γ x_{t-1}

3. Realized variance equation: 
   x_t = ξ + φ h_t^2 + u_t

Where:
- r_t is the daily return
- h_t^2 is the conditional variance
- x_t is the realized variance
- α_0, β_1, γ, ξ, φ are model parameters
- ε_t and u_t are random variables

## Features
- Processes high-frequency data to calculate daily realized volatility
- Estimates RealGARCH model parameters via maximum likelihood
- Forecasts future volatility 
- Generates visualizations of returns, volatility, and forecasts
- Handles data quality issues and provides fallback to simulated data

## Usage
Run the script with Python:
```
python realgarch.py
```

The script automatically:
1. Loads and processes high-frequency price data
2. Calculates daily returns and realized volatility
3. Fits the RealGARCH model
4. Generates forecasts
5. Creates plots and saves results