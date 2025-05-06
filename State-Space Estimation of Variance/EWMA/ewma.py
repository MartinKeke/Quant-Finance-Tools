import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class EWMA:
    """
    Standard Exponentially Weighted Moving Average (EWMA) for volatility estimation.
    
    This implements the model from Image 3 in the form:
    σ²ₜ = (1-K)r²ₜ₋₁ + Kσ²ₜ₋₁
    
    Where K is the decay factor (λ in some literature).
    """
    
    def __init__(self, decay_factor=0.94):
        """
        Initialize EWMA model.
        
        Args:
            decay_factor (float): The decay factor K, typically 0.94 for daily data
                                 (as used by RiskMetrics)
        """
        self.K = decay_factor
        self.variances = None
        self.volatilities = None
    
    def fit(self, returns):
        """
        Estimate volatility using EWMA.
        
        Args:
            returns (array-like): Time series of asset returns
            
        Returns:
            self: The fitted model
        """
        returns = np.asarray(returns)
        n = len(returns)
        
        # Initialize arrays for variance and volatility
        variances = np.zeros(n)
        
        # Initialize with sample variance
        variances[0] = np.var(returns)
        
        # Calculate EWMA variance recursively
        for t in range(1, n):
            # EWMA recursion: σ²ₜ = (1-K)r²ₜ₋₁ + Kσ²ₜ₋₁
            variances[t] = (1 - self.K) * returns[t-1]**2 + self.K * variances[t-1]
        
        self.variances = variances
        self.volatilities = np.sqrt(variances)
        return self
    
    def forecast(self, steps=1):
        """
        Forecast future volatility.
        
        For standard EWMA, the forecast is constant at the last estimated value.
        
        Args:
            steps (int): Number of steps ahead to forecast
            
        Returns:
            array: Forecasted volatilities
        """
        if self.volatilities is None:
            raise ValueError("Model must be fitted before forecasting")
        
        # For EWMA, multi-step forecasts are all the same
        return np.ones(steps) * self.volatilities[-1]


class MeanRevertingEWMA:
    """
    Mean-Reverting EWMA model as described in Image 5 and Image 6.
    
    This implements the extended state model:
    x_{t+1} = x_t - λ(x_t - μ) + τ_ε*ε_{t+1}
    
    Where:
    - x_t is the state (true variance)
    - λ is the mean reversion rate
    - μ is the long-run mean
    - τ_ε is the innovation noise scale
    
    With optimal estimation recursion:
    x̂_{t+1|t} = (1-K)x̂_{t|t-1} + Ky_t
    """
    
    def __init__(self, lambda_=0.05, mu=None, K=None, tau_epsilon=None, tau_eta=None):
        """
        Initialize Mean-Reverting EWMA model.
        
        Args:
            lambda_ (float): Mean reversion rate
            mu (float, optional): Long-run mean level, if None will be estimated from data
            K (float, optional): Kalman gain, if None will be calculated optimally
            tau_epsilon (float, optional): Innovation noise standard deviation
            tau_eta (float, optional): Measurement noise standard deviation
        """
        self.lambda_ = lambda_
        self.mu = mu
        self.K = K
        self.tau_epsilon = tau_epsilon
        self.tau_eta = tau_eta
        self.states = None
        self.volatilities = None
    
    def fit(self, returns):
        """
        Estimate volatility using the Mean-Reverting EWMA model.
        
        Args:
            returns (array-like): Time series of asset returns
            
        Returns:
            self: The fitted model
        """
        returns = np.asarray(returns)
        n = len(returns)
        
        # Squared returns are the observations in this model
        squared_returns = returns**2
        
        # If mu not provided, estimate from data
        if self.mu is None:
            self.mu = np.mean(squared_returns)
        
        # If noise parameters not provided, estimate them
        if self.tau_epsilon is None:
            # Simple heuristic based on data variability
            self.tau_epsilon = 0.1 * np.std(squared_returns)
        
        if self.tau_eta is None:
            # Measurement noise typically larger than innovation noise
            self.tau_eta = 2.0 * self.tau_epsilon
        
        # Calculate kappa (ratio of measurement to innovation noise)
        kappa = self.tau_eta / self.tau_epsilon
        
        # Calculate optimal Kalman gain if not provided
        if self.K is None:
            # From Image 5, for κ >> 1, use simplified formula
            if kappa > 5:  # Arbitrary threshold
                self.K = 1 / (1 + kappa)
            else:
                # Use the exact formula for stationary Kalman gain
                var_stationary = self.tau_epsilon**2 * (1 + np.sqrt((2*kappa)**2 + 1)) / 2
                self.K = var_stationary / (var_stationary + self.tau_eta**2)
        
        # Initialize arrays
        states = np.zeros(n)
        
        # Initialize with long-run mean
        states[0] = self.mu
        
        # Kalman filter recursive estimation
        for t in range(1, n):
            # Prediction step with mean reversion
            predicted_state = states[t-1] - self.lambda_ * (states[t-1] - self.mu)
            
            # Update step
            states[t] = (1 - self.K) * predicted_state + self.K * squared_returns[t]
        
        self.states = states
        self.volatilities = np.sqrt(np.maximum(0, states))  # Ensure non-negative
        return self
    
    def forecast(self, steps=1):
        """
        Forecast future volatility accounting for mean reversion.
        
        Args:
            steps (int): Number of steps ahead to forecast
            
        Returns:
            array: Forecasted volatilities
        """
        if self.states is None:
            raise ValueError("Model must be fitted before forecasting")
        
        # Initialize with last state
        forecasts = np.zeros(steps)
        current_state = self.states[-1]
        
        # Apply mean reversion for each forecast step
        for t in range(steps):
            current_state = current_state - self.lambda_ * (current_state - self.mu)
            forecasts[t] = current_state
        
        # Convert variance to volatility
        return np.sqrt(np.maximum(0, forecasts))


def load_and_prepare_data(file_path):
    """
    Load and prepare financial data for volatility modeling.
    
    Args:
        file_path (str): Path to CSV file with price data
        
    Returns:
        DataFrame: Processed data with returns
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Check if 'Date' column exists
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
    
    # Clean price column if needed
    if 'Price' in df.columns:
        if isinstance(df['Price'].iloc[0], str):
            df['Price'] = df['Price'].str.replace(',', '').astype(float)
    
    # Calculate returns
    if 'Return' not in df.columns:
        df['Return'] = df['Price'].pct_change()
        df = df.dropna(subset=['Return'])
    
    return df


def plot_volatility_comparison(returns, ewma_model, mr_ewma_model, title='Volatility Comparison', save_path=None):
    """
    Plot volatility estimates from both models for comparison.
    
    Args:
        returns (array-like): Original return series
        ewma_model (EWMA): Fitted standard EWMA model
        mr_ewma_model (MeanRevertingEWMA): Fitted mean-reverting EWMA model
        title (str): Plot title
        save_path (str, optional): Path to save the figure
    """
    # Create time index for plotting
    t = np.arange(len(returns))
    
    # Create plot
    fig = plt.figure(figsize=(12, 8))
    
    # Plot squared returns
    plt.subplot(3, 1, 1)
    plt.plot(t, returns**2, 'gray', alpha=0.5, label='Squared Returns')
    plt.title('Squared Returns', fontsize=12)
    plt.ylabel('r²')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot standard EWMA volatility
    plt.subplot(3, 1, 2)
    plt.plot(t, ewma_model.volatilities, 'b-', label=f'EWMA (K={ewma_model.K:.2f})')
    plt.title('Standard EWMA Volatility', fontsize=12)
    plt.ylabel('sigma')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot mean-reverting EWMA volatility
    plt.subplot(3, 1, 3)
    plt.plot(t, mr_ewma_model.volatilities, 'r-', 
             label=f'Mean-Reverting EWMA (lambda={mr_ewma_model.lambda_:.2f}, K={mr_ewma_model.K:.2f})')
    plt.axhline(y=np.sqrt(mr_ewma_model.mu), color='g', linestyle='--', 
                label=f'Long-run Mean (sqrt(mu)={np.sqrt(mr_ewma_model.mu):.4f})')
    plt.title('Mean-Reverting EWMA Volatility', fontsize=12)
    plt.xlabel('Time')
    plt.ylabel('sigma')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    plt.show()
    return fig


def plot_forecast_comparison(ewma_model, mr_ewma_model, forecast_steps=100, save_path=None):
    """
    Plot volatility forecasts from both models.
    
    Args:
        ewma_model (EWMA): Fitted standard EWMA model
        mr_ewma_model (MeanRevertingEWMA): Fitted mean-reverting EWMA model
        forecast_steps (int): Number of steps to forecast
        save_path (str, optional): Path to save the figure
    """
    # Generate forecasts
    ewma_forecast = ewma_model.forecast(forecast_steps)
    mr_ewma_forecast = mr_ewma_model.forecast(forecast_steps)
    
    # Time steps for plotting
    t = np.arange(forecast_steps)
    
    # Create plot
    fig = plt.figure(figsize=(10, 6))
    
    plt.plot(t, ewma_forecast, 'b-', label=f'EWMA (K={ewma_model.K:.2f})')
    plt.plot(t, mr_ewma_forecast, 'r-', 
             label=f'Mean-Reverting EWMA (lambda={mr_ewma_model.lambda_:.2f})')
    plt.axhline(y=np.sqrt(mr_ewma_model.mu), color='g', linestyle='--', 
                label=f'Long-run Volatility (sqrt(mu)={np.sqrt(mr_ewma_model.mu):.4f})')
    
    plt.title('Volatility Forecast Comparison', fontsize=14)
    plt.xlabel('Steps Ahead')
    plt.ylabel('Forecasted Volatility')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    plt.show()
    return fig


def analyze_bitcoin_volatility(file_path, save_results=True):
    """
    Analyze Bitcoin volatility using both EWMA models.
    
    Args:
        file_path (str): Path to Bitcoin price data
        save_results (bool): Whether to save results to CSV and figures
        
    Returns:
        tuple: (df, ewma_model, mr_ewma_model)
    """
    # Load and prepare data
    df = load_and_prepare_data(file_path)
    returns = df['Return'].values
    
    print(f"Loaded {len(df)} days of Bitcoin data")
    print(f"Mean return: {np.mean(returns):.4f}")
    print(f"Return volatility: {np.std(returns):.4f}")
    
    # Fit models
    print("\nFitting EWMA model...")
    ewma_model = EWMA(decay_factor=0.94)
    ewma_model.fit(returns)
    
    print("Fitting Mean-Reverting EWMA model...")
    mr_ewma_model = MeanRevertingEWMA(lambda_=0.05)
    mr_ewma_model.fit(returns)
    
    # Add volatility estimates to dataframe
    df['EWMA_Volatility'] = ewma_model.volatilities
    df['MR_EWMA_Volatility'] = mr_ewma_model.volatilities
    
    # Print model parameters
    print("\nModel Parameters:")
    print(f"EWMA decay factor (K): {ewma_model.K:.4f}")
    print(f"Mean-Reverting EWMA parameters:")
    print(f"  - Mean reversion rate (λ): {mr_ewma_model.lambda_:.4f}")
    print(f"  - Long-run variance (μ): {mr_ewma_model.mu:.6f}")
    print(f"  - Long-run volatility (√μ): {np.sqrt(mr_ewma_model.mu):.4f}")
    print(f"  - Kalman gain (K): {mr_ewma_model.K:.4f}")
    
    # Latest volatility estimates
    latest_ewma = ewma_model.volatilities[-1]
    latest_mr_ewma = mr_ewma_model.volatilities[-1]
    
    print(f"\nLatest Volatility Estimates:")
    print(f"EWMA: {latest_ewma:.4f}")
    print(f"Mean-Reverting EWMA: {latest_mr_ewma:.4f}")
    
    # Forecast volatility for next 30 days
    forecast_days = 30
    ewma_forecast = ewma_model.forecast(forecast_days)
    mr_ewma_forecast = mr_ewma_model.forecast(forecast_days)
    
    print(f"\n{forecast_days}-day Volatility Forecast:")
    print(f"EWMA: {ewma_forecast[-1]:.4f} (constant)")
    print(f"Mean-Reverting EWMA: {mr_ewma_forecast[-1]:.4f}")
    
    # Save results if requested
    if save_results:
        # Save volatility estimates to CSV
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create results directory if it doesn't exist
        results_dir = os.path.join(script_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save dataframe with volatility estimates
        csv_path = os.path.join(results_dir, "bitcoin_volatility_estimates.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved volatility estimates to: {csv_path}")
        
        # Save model parameters to text file
        params_path = os.path.join(results_dir, "model_parameters.txt")
        with open(params_path, 'w') as f:
            f.write("EWMA and Mean-Reverting EWMA Model Parameters\n")
            f.write("==============================================\n\n")
            f.write(f"EWMA decay factor (K): {ewma_model.K:.4f}\n\n")
            f.write("Mean-Reverting EWMA parameters:\n")
            f.write(f"  - Mean reversion rate (lambda): {mr_ewma_model.lambda_:.4f}\n")
            f.write(f"  - Long-run variance (mu): {mr_ewma_model.mu:.6f}\n")
            f.write(f"  - Long-run volatility (sqrt(mu)): {np.sqrt(mr_ewma_model.mu):.4f}\n")
            f.write(f"  - Kalman gain (K): {mr_ewma_model.K:.4f}\n\n")
            f.write(f"Latest Volatility Estimates (as of {df['Date'].iloc[-1].strftime('%Y-%m-%d')}):\n")
            f.write(f"  - EWMA: {latest_ewma:.4f}\n")
            f.write(f"  - Mean-Reverting EWMA: {latest_mr_ewma:.4f}\n\n")
            f.write(f"{forecast_days}-day Volatility Forecast:\n")
            f.write(f"  - EWMA: {ewma_forecast[-1]:.4f} (constant)\n")
            f.write(f"  - Mean-Reverting EWMA: {mr_ewma_forecast[-1]:.4f}\n")
        print(f"Saved model parameters to: {params_path}")
        
        # Save forecast data to CSV
        last_date = df['Date'].iloc[-1]
        forecast_dates = pd.date_range(start=last_date, periods=forecast_days + 1)[1:]
        
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'EWMA_Forecast': ewma_forecast,
            'MR_EWMA_Forecast': mr_ewma_forecast
        })
        
        forecast_path = os.path.join(results_dir, "volatility_forecast.csv")
        forecast_df.to_csv(forecast_path, index=False)
        print(f"Saved volatility forecast to: {forecast_path}")
        
        # Create and save plots
        # Plot volatility comparison and save
        fig_comparison = plot_volatility_comparison(df['Return'].values, 
                                                   ewma_model, 
                                                   mr_ewma_model, 
                                                   title='Bitcoin Volatility Comparison',
                                                   save_path=os.path.join(results_dir, "volatility_comparison.png"))
        
        # Plot forecast comparison and save
        fig_forecast = plot_forecast_comparison(ewma_model, 
                                               mr_ewma_model, 
                                               forecast_steps=forecast_days,
                                               save_path=os.path.join(results_dir, "volatility_forecast.png"))
    
    return df, ewma_model, mr_ewma_model


# Example usage
if __name__ == "__main__":
    file_path = r"C:\Users\Martin\Quant Finance Tools\State-Space Estimation of Variance\Bitcoin Historical Data.csv"
    
    # Analyze Bitcoin volatility
    df, ewma_model, mr_ewma_model = analyze_bitcoin_volatility(file_path)
    
    # Plot volatility comparison
    plot_volatility_comparison(df['Return'].values, ewma_model, mr_ewma_model, 
                               title='Bitcoin Volatility Comparison')
    
    # Plot forecast comparison
    plot_forecast_comparison(ewma_model, mr_ewma_model, forecast_steps=100)