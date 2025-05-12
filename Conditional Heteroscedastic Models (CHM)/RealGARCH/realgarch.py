import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sqlite3
import os
import sys
import importlib.util
import datetime
import json
import traceback

# Path to your realized volatility script and database
RV_SCRIPT_PATH = r"C:\Users\Martin\Quant Finance Tools\Conditional Heteroscedastic Models (CHM)\Realized Volatility\realized_volatility.py"
DB_PATH = r"C:\Users\Martin\Quant Finance Tools\Conditional Heteroscedastic Models (CHM)\Realized Volatility\bitcoin_prices_test.db"

# Create RealGARCH output directory in the same folder as this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "RealGARCH")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Import your realized volatility module dynamically
def import_rv_module():
    module_name = "realized_volatility"
    spec = importlib.util.spec_from_file_location(module_name, RV_SCRIPT_PATH)
    rv_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = rv_module
    spec.loader.exec_module(rv_module)
    return rv_module

# Load the realized volatility module
rv_module = import_rv_module()

# Function to process the high-frequency data directly
def process_bitcoin_data():
    """
    Process the Bitcoin 1-minute data directly to create returns and realized volatility
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Get the 1-minute Bitcoin data
        query = """
        SELECT timestamp, close, date_time
        FROM bitcoin_1m 
        ORDER BY timestamp ASC
        """
        
        hf_data = pd.read_sql_query(query, conn)
        conn.close()
        
        if hf_data.empty:
            print("No data retrieved from database")
            return None, None
        
        print(f"Loaded {len(hf_data)} high-frequency observations")
        
        # Convert timestamp to datetime
        hf_data['datetime'] = pd.to_datetime(hf_data['timestamp'], unit='ms')
        
        # Set datetime as index
        hf_data = hf_data.set_index('datetime')
        
        # Calculate log returns
        hf_data['log_return'] = np.log(hf_data['close'] / hf_data['close'].shift(1))
        hf_data = hf_data.dropna()
        
        # Calculate squared returns for realized volatility
        hf_data['squared_return'] = hf_data['log_return'] ** 2
        
        # Determine the frequency of the data by checking time differences
        time_diffs = (hf_data.index[1:] - hf_data.index[:-1]).total_seconds()
        median_diff = np.median(time_diffs)
        print(f"Median time difference between observations: {median_diff} seconds")
        
        # Determine if data is approximately minute-by-minute
        is_minute_data = 50 <= median_diff <= 70  # Around 60 seconds
        
        if is_minute_data:
            print("Detected 1-minute data")
        else:
            print(f"Data frequency: {median_diff} seconds (not 1-minute data)")
        
        # Create daily returns
        daily_returns = hf_data['log_return'].resample('D').sum().to_frame()
        
        # Calculate daily realized volatility (sum of squared intraday returns)
        # Annualize by multiplying by sqrt(252) (trading days in a year)
        daily_rv = hf_data['squared_return'].resample('D').sum().to_frame()
        daily_rv.columns = ['realized_var']
        daily_rv['realized_var'] = daily_rv['realized_var'] * 252
        
        # Filter out days with very few observations, which might have incomplete data
        obs_per_day = hf_data.resample('D').size()
        min_obs = 60  # Require at least 60 observations per day (1 hour of 1-minute data)
        valid_days = obs_per_day[obs_per_day >= min_obs].index
        
        daily_returns = daily_returns.loc[valid_days]
        daily_rv = daily_rv.loc[valid_days]
        
        # Remove days with zero or very small variance
        daily_rv = daily_rv[daily_rv['realized_var'] > 1e-10]
        
        # Match the dates
        common_dates = daily_returns.index.intersection(daily_rv.index)
        daily_returns = daily_returns.loc[common_dates]
        daily_rv = daily_rv.loc[common_dates]
        
        print(f"Created daily data from {common_dates[0]} to {common_dates[-1]}")
        print(f"Total days: {len(common_dates)}")
        
        if len(common_dates) < 30:
            print(f"Warning: Only {len(common_dates)} days with valid data. Model estimation may be unstable.")
            if len(common_dates) < 10:
                print("Error: Too few days with valid data. Cannot proceed.")
                # If we don't have enough data, simulate some for testing the model
                print("Generating simulated data for testing...")
                return generate_simulated_data(252)  # Generate 1 year of simulated data
        
        return daily_returns, daily_rv
        
    except Exception as e:
        print(f"Error processing Bitcoin data: {str(e)}")
        traceback.print_exc()
        return None, None

# Function to generate simulated data for testing
def generate_simulated_data(n_days):
    """
    Generate simulated data for testing the RealGARCH model
    
    Parameters:
    n_days: number of days to simulate
    
    Returns:
    daily_returns, daily_rv
    """
    print(f"Generating {n_days} days of simulated data for testing")
    
    # Parameters for the simulation
    omega = 0.05
    beta = 0.85
    gamma = 0.1
    xi = 0.0
    phi = 1.0
    
    # Initialize arrays
    h_t = np.zeros(n_days)
    x_t = np.zeros(n_days)
    r_t = np.zeros(n_days)
    
    # Initial values
    h_t[0] = omega / (1 - beta - gamma * phi)  # Unconditional variance
    x_t[0] = xi + phi * h_t[0] + np.random.normal(0, 0.1)
    r_t[0] = np.sqrt(h_t[0]) * np.random.normal(0, 1)
    
    # Simulate
    for t in range(1, n_days):
        # Volatility equation: h_t² = α₀ + β₁h_t-1² + γx_t-1
        h_t[t] = omega + beta * h_t[t-1] + gamma * x_t[t-1]
        
        # Return equation: r_t = h_t^0.5 * ε_t
        epsilon_t = np.random.normal(0, 1)
        r_t[t] = np.sqrt(h_t[t]) * epsilon_t
        
        # Realized volatility equation: x_t = ξ + φh_t² + u_t
        u_t = np.random.normal(0, 0.1)
        x_t[t] = xi + phi * h_t[t] + u_t
    
    # Create dataframes
    dates = pd.date_range(start='2025-01-01', periods=n_days)
    daily_returns = pd.DataFrame({'log_return': r_t}, index=dates)
    daily_rv = pd.DataFrame({'realized_var': x_t}, index=dates)
    
    return daily_returns, daily_rv

class RealGARCH:
    def __init__(self, debug=True):
        self.params = None
        self.param_names = ['omega', 'beta', 'gamma', 'xi', 'phi']
        self.h_t = None
        self.x_t = None
        self.debug = debug
        
    def _neg_log_likelihood(self, params, returns, rv):
        """
        Negative log-likelihood function for RealGARCH(1,1)
        
        Parameters:
        params: list [omega, beta, gamma, xi, phi]
        returns: array of daily returns
        rv: array of daily realized variance
        """
        omega, beta, gamma, xi, phi = params
        
        T = len(returns)
        h_t = np.zeros(T)
        x_t = np.zeros(T)
        
        # Initial values
        h_t[0] = np.var(returns)
        x_t[0] = rv[0]
        
        # Calculate h_t and log-likelihood
        log_likelihood = 0
        
        for t in range(1, T):
            # h_t² = α₀ + β₁h_t-1² + γx_t-1
            h_t[t] = omega + beta * h_t[t-1] + gamma * x_t[t-1]
            
            # Ensure h_t is positive
            if h_t[t] <= 0:
                return 1e10
            
            # For the return equation r_t = h_t^0.5 * ε_t
            log_likelihood += -0.5 * np.log(h_t[t]) - 0.5 * returns[t]**2 / h_t[t]
            
            # In the likelihood, x_t is observed (it's the realized variance)
            x_t[t] = rv[t]
        
        return -log_likelihood
    
    def fit(self, returns, rv, initial_params=None):
        """
        Fit the RealGARCH(1,1) model
        
        Parameters:
        returns: array of daily returns
        rv: array of daily realized variance
        initial_params: initial parameter values [omega, beta, gamma, xi, phi]
        
        Returns:
        estimated parameters
        """
        if self.debug:
            print(f"Fitting RealGARCH model to {len(returns)} observations")
            print(f"Returns mean: {np.mean(returns):.6f}, std: {np.std(returns):.6f}")
            print(f"RV mean: {np.mean(rv):.6f}, std: {np.std(rv):.6f}")
        
        if initial_params is None:
            # Use data-driven initial values
            omega = 0.01 * np.var(returns)
            beta = 0.8
            gamma = 0.1
            xi = 0.0
            phi = 1.0
            initial_params = [omega, beta, gamma, xi, phi]
            
            if self.debug:
                print(f"Initial parameters: {initial_params}")
        
        # Bounds for parameters
        bounds = [
            (1e-10, 1),     # omega > 0
            (0, 0.999),     # 0 < beta < 1
            (0, 0.999),     # 0 < gamma < 1
            (None, None),   # xi can be any value
            (0, None)       # phi > 0
        ]
        
        # Constraint: beta + gamma < 1 for stationarity
        def constraint(params):
            return 1 - params[1] - params[2]
        
        constraints = {'type': 'ineq', 'fun': constraint}
        
        try:
            result = minimize(
                self._neg_log_likelihood,
                initial_params,
                args=(returns, rv),
                bounds=bounds,
                constraints=constraints,
                method='SLSQP',
                options={'disp': self.debug, 'maxiter': 1000}
            )
            
            if not result.success:
                print(f"Warning: Optimization did not converge. Message: {result.message}")
                
            if self.debug:
                print(f"Optimization result: {result}")
                
            self.params = result.x
            
            # Store the fitted values
            T = len(returns)
            self.h_t = np.zeros(T)
            self.x_t = np.zeros(T)
            
            omega, beta, gamma, xi, phi = self.params
            
            # Initial values
            self.h_t[0] = np.var(returns)
            self.x_t[0] = rv[0]
            
            # Calculate h_t and x_t
            for t in range(1, T):
                self.h_t[t] = omega + beta * self.h_t[t-1] + gamma * self.x_t[t-1]
                self.x_t[t] = rv[t]
                
            if self.debug:
                print(f"Fitted h_t - min: {np.min(self.h_t):.6f}, max: {np.max(self.h_t):.6f}, mean: {np.mean(self.h_t):.6f}")
                print(f"Fitted x_t - min: {np.min(self.x_t):.6f}, max: {np.max(self.x_t):.6f}, mean: {np.mean(self.x_t):.6f}")
            
            return dict(zip(self.param_names, self.params))
            
        except Exception as e:
            print(f"Error in model fitting: {str(e)}")
            traceback.print_exc()
            # Return dummy parameters in case of error
            self.params = initial_params
            return dict(zip(self.param_names, self.params))
    
    def forecast(self, returns, rv, n_ahead=1):
        """
        Forecast volatility n days ahead
        
        Parameters:
        returns: array of daily returns
        rv: array of daily realized variance
        n_ahead: number of days to forecast ahead
        
        Returns:
        forecasted volatility
        """
        if self.params is None:
            raise ValueError("Model must be fit before forecasting")
        
        omega, beta, gamma, xi, phi = self.params
        
        T = len(returns)
        h_t = np.zeros(T + n_ahead)
        x_t = np.zeros(T + n_ahead)
        
        # Fill in the historical values for h_t
        if self.h_t is not None and self.x_t is not None:
            h_t[:T] = self.h_t
            x_t[:T] = self.x_t
        else:
            h_t[0] = np.var(returns)
            x_t[0] = rv[0]
            
            for t in range(1, T):
                h_t[t] = omega + beta * h_t[t-1] + gamma * x_t[t-1]
                x_t[t] = rv[t]
        
        # Forecast future values
        for t in range(T, T + n_ahead):
            h_t[t] = omega + beta * h_t[t-1] + gamma * x_t[t-1]
            # For forecasting, we use the model equation for x_t
            x_t[t] = xi + phi * h_t[t]
        
        if self.debug:
            print(f"Forecast h_t - min: {np.min(h_t[T:]):.6f}, max: {np.max(h_t[T:]):.6f}, mean: {np.mean(h_t[T:]):.6f}")
            print(f"Forecast x_t - min: {np.min(x_t[T:]):.6f}, max: {np.max(x_t[T:]):.6f}, mean: {np.mean(x_t[T:]):.6f}")
        
        return h_t[T:], x_t[T:]

def save_results(model, h_forecast, x_forecast, daily_returns, daily_rv, forecast_dates):
    """
    Save model results to the RealGARCH directory
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model parameters
    params_dict = {name: float(value) for name, value in zip(model.param_names, model.params)}
    params_dict['timestamp'] = timestamp
    params_dict['data_start_date'] = daily_returns.index[0].strftime("%Y-%m-%d")
    params_dict['data_end_date'] = daily_returns.index[-1].strftime("%Y-%m-%d")
    params_dict['n_observations'] = len(daily_returns)
    
    params_file = os.path.join(OUTPUT_DIR, f"realgarch_params_{timestamp}.json")
    with open(params_file, 'w') as f:
        json.dump(params_dict, f, indent=4)
    
    # Save historical fitted values
    historical_df = pd.DataFrame({
        'date': daily_returns.index.strftime("%Y-%m-%d"),
        'returns': daily_returns['log_return'].values,
        'realized_vol': np.sqrt(daily_rv['realized_var'].values),  # Convert back to volatility
        'fitted_cond_vol': np.sqrt(model.h_t),  # Convert variance to volatility
        'fitted_rv': np.sqrt(model.x_t)  # Convert variance to volatility
    })
    historical_file = os.path.join(OUTPUT_DIR, f"realgarch_historical_{timestamp}.csv")
    historical_df.to_csv(historical_file, index=False)
    
    # Save forecasts
    forecast_df = pd.DataFrame({
        'date': forecast_dates.strftime("%Y-%m-%d"),
        'forecasted_cond_vol': np.sqrt(h_forecast),  # Convert variance to volatility
        'forecasted_rv': np.sqrt(x_forecast)  # Convert variance to volatility
    })
    forecast_file = os.path.join(OUTPUT_DIR, f"realgarch_forecast_{timestamp}.csv")
    forecast_df.to_csv(forecast_file, index=False)
    
    # Also save the raw data
    raw_data_df = pd.DataFrame({
        'date': daily_returns.index.strftime("%Y-%m-%d"),
        'returns': daily_returns['log_return'].values,
        'realized_var': daily_rv['realized_var'].values
    })
    raw_data_file = os.path.join(OUTPUT_DIR, f"realgarch_rawdata_{timestamp}.csv")
    raw_data_df.to_csv(raw_data_file, index=False)
    
    print(f"Results saved in the RealGARCH directory:")
    print(f"  - Parameters: {os.path.basename(params_file)}")
    print(f"  - Historical fitted values: {os.path.basename(historical_file)}")
    print(f"  - Forecasts: {os.path.basename(forecast_file)}")
    print(f"  - Raw data: {os.path.basename(raw_data_file)}")
    
    return params_file, historical_file, forecast_file, raw_data_file

def plot_results(daily_returns, daily_rv, model, h_forecast, x_forecast, forecast_dates):
    """
    Plot the RealGARCH results and save the plots to the RealGARCH directory
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Check if we have valid data to plot
    if (daily_returns is None or daily_returns.empty or 
        daily_rv is None or daily_rv.empty or 
        model.h_t is None or model.x_t is None or 
        h_forecast is None or x_forecast is None):
        print("Warning: Some data is missing or empty, cannot create plots")
        return None
    
    # Check if there are any invalid values
    if (np.isnan(daily_returns['log_return']).any() or 
        np.isnan(daily_rv['realized_var']).any() or 
        np.isnan(model.h_t).any() or 
        np.isnan(model.x_t).any() or 
        np.isnan(h_forecast).any() or 
        np.isnan(x_forecast).any()):
        print("Warning: Data contains NaN values, plots may be incomplete")
    
    # Create figure for the plots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Returns
    plt.subplot(3, 1, 1)
    returns_line = plt.plot(daily_returns.index, daily_returns['log_return'], 'b-', linewidth=0.5)
    plt.title('Daily Log Returns')
    plt.ylabel('Log Return')
    plt.grid(True)
    
    # Print some diagnostic info
    print(f"Returns plot - line data points: {len(returns_line[0].get_xdata())}")
    
    # Plot 2: Historical volatility
    plt.subplot(3, 1, 2)
    h_t_line = plt.plot(daily_returns.index, np.sqrt(model.h_t), 'r-', 
                        label='Model Conditional Volatility')
    rv_line = plt.plot(daily_rv.index, np.sqrt(daily_rv['realized_var']), 'b-', alpha=0.7, 
                      label='Realized Volatility')
    plt.title('Historical Volatility Comparison')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)
    
    # Print some diagnostic info
    print(f"Historical vol plot - model line data points: {len(h_t_line[0].get_xdata())}")
    print(f"Historical vol plot - RV line data points: {len(rv_line[0].get_xdata())}")
    
    # Plot 3: Forecasted volatility
    plt.subplot(3, 1, 3)
    h_forecast_line = plt.plot(forecast_dates, np.sqrt(h_forecast), 'r-', 
                              label='Forecasted Conditional Volatility')
    x_forecast_line = plt.plot(forecast_dates, np.sqrt(x_forecast), 'b-', 
                              label='Forecasted Realized Volatility')
    plt.title('Volatility Forecasts')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)
    
    # Print some diagnostic info
    print(f"Forecast plot - cond vol line data points: {len(h_forecast_line[0].get_xdata())}")
    print(f"Forecast plot - RV line data points: {len(x_forecast_line[0].get_xdata())}")
    
    plt.tight_layout()
    
    # Save the combined plot
    combined_plot_file = os.path.join(OUTPUT_DIR, f"realgarch_combined_plot_{timestamp}.png")
    plt.savefig(combined_plot_file, dpi=300)
    plt.close()
    
    # Save individual plots for more detailed viewing
    
    # Returns plot
    plt.figure(figsize=(12, 6))
    plt.plot(daily_returns.index, daily_returns['log_return'], 'b-', linewidth=0.5)
    plt.title('Daily Log Returns')
    plt.ylabel('Log Return')
    plt.grid(True)
    returns_plot_file = os.path.join(OUTPUT_DIR, f"realgarch_returns_{timestamp}.png")
    plt.savefig(returns_plot_file, dpi=300)
    plt.close()
    
    # Historical volatility plot
    plt.figure(figsize=(12, 6))
    plt.plot(daily_returns.index, np.sqrt(model.h_t), 'r-', label='Model Conditional Volatility')
    plt.plot(daily_rv.index, np.sqrt(daily_rv['realized_var']), 'b-', alpha=0.7, 
             label='Realized Volatility')
    plt.title('Historical Volatility Comparison')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)
    hist_vol_plot_file = os.path.join(OUTPUT_DIR, f"realgarch_historical_vol_{timestamp}.png")
    plt.savefig(hist_vol_plot_file, dpi=300)
    plt.close()
    
    # Forecast plot
    plt.figure(figsize=(12, 6))
    plt.plot(forecast_dates, np.sqrt(h_forecast), 'r-', 
             label='Forecasted Conditional Volatility')
    plt.plot(forecast_dates, np.sqrt(x_forecast), 'b-', 
             label='Forecasted Realized Volatility')
    plt.title('Volatility Forecasts')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)
    forecast_plot_file = os.path.join(OUTPUT_DIR, f"realgarch_forecast_{timestamp}.png")
    plt.savefig(forecast_plot_file, dpi=300)
    plt.close()
    
    # Also save raw data plots to help diagnose issues
    plt.figure(figsize=(12, 6))
    plt.plot(daily_returns.index, daily_returns['log_return'], 'b-', linewidth=0.5)
    plt.title('Raw Daily Log Returns')
    plt.ylabel('Log Return')
    plt.grid(True)
    raw_returns_plot_file = os.path.join(OUTPUT_DIR, f"realgarch_raw_returns_{timestamp}.png")
    plt.savefig(raw_returns_plot_file, dpi=300)
    plt.close()
    
    plt.figure(figsize=(12, 6))
    plt.plot(daily_rv.index, daily_rv['realized_var'], 'b-', linewidth=0.5)
    plt.title('Raw Daily Realized Variance')
    plt.ylabel('Realized Variance')
    plt.grid(True)
    raw_rv_plot_file = os.path.join(OUTPUT_DIR, f"realgarch_raw_rv_{timestamp}.png")
    plt.savefig(raw_rv_plot_file, dpi=300)
    plt.close()
    
    print(f"Plots saved in the RealGARCH directory:")
    print(f"  - Combined plot: {os.path.basename(combined_plot_file)}")
    print(f"  - Returns plot: {os.path.basename(returns_plot_file)}")
    print(f"  - Historical volatility plot: {os.path.basename(hist_vol_plot_file)}")
    print(f"  - Forecast plot: {os.path.basename(forecast_plot_file)}")
    print(f"  - Raw data plots: {os.path.basename(raw_returns_plot_file)}, {os.path.basename(raw_rv_plot_file)}")
    
    # Display the combined plot
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(daily_returns.index, daily_returns['log_return'], 'b-', linewidth=0.5)
    plt.title('Daily Log Returns')
    plt.ylabel('Log Return')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(daily_returns.index, np.sqrt(model.h_t), 'r-', label='Model Conditional Volatility')
    plt.plot(daily_rv.index, np.sqrt(daily_rv['realized_var']), 'b-', alpha=0.7, 
             label='Realized Volatility')
    plt.title('Historical Volatility Comparison')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(forecast_dates, np.sqrt(h_forecast), 'r-', 
             label='Forecasted Conditional Volatility')
    plt.plot(forecast_dates, np.sqrt(x_forecast), 'b-', 
             label='Forecasted Realized Volatility')
    plt.title('Volatility Forecasts')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return combined_plot_file

def run_realgarch_analysis():
    """
    Main function to run RealGARCH analysis with your data
    """
    try:
        print(f"RealGARCH analysis output will be saved to: {OUTPUT_DIR}")
        
        # Process Bitcoin data directly
        print("Processing Bitcoin data...")
        daily_returns, daily_rv = process_bitcoin_data()
        
        if daily_returns is None or daily_rv is None or daily_returns.empty or daily_rv.empty:
            print("Error: Failed to process data, using simulated data instead")
            daily_returns, daily_rv = generate_simulated_data(252)  # Generate 1 year of simulated data
        
        print(f"Data prepared: {len(daily_returns)} daily observations")
        print(f"Date range: {daily_returns.index[0].date()} to {daily_returns.index[-1].date()}")
        
        # Save raw data to inspect
        raw_data_df = pd.DataFrame({
            'date': daily_returns.index.strftime("%Y-%m-%d"),
            'returns': daily_returns['log_return'].values,
            'realized_var': daily_rv['realized_var'].values
        })
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_data_file = os.path.join(OUTPUT_DIR, f"realgarch_rawdata_{timestamp}.csv")
        raw_data_df.to_csv(raw_data_file, index=False)
        print(f"Raw data saved to: {os.path.basename(raw_data_file)}")
        
        # Fit the RealGARCH model
        print("Fitting RealGARCH model...")
        model = RealGARCH(debug=True)
        # For the model, we use daily returns and realized variance (not volatility)
        params = model.fit(
            daily_returns['log_return'].values, 
            daily_rv['realized_var'].values
        )
        
        print("Estimated RealGARCH parameters:")
        for name, value in params.items():
            print(f"{name}: {value:.6f}")
        
        # Forecast volatility
        forecast_horizon = 10  # Number of days to forecast ahead
        print(f"Forecasting volatility {forecast_horizon} days ahead...")
        h_forecast, x_forecast = model.forecast(
            daily_returns['log_return'].values, 
            daily_rv['realized_var'].values,
            n_ahead=forecast_horizon
        )
        
        # Create forecast dates
        last_date = daily_returns.index[-1]
        forecast_dates = pd.date_range(start=last_date, periods=forecast_horizon+1)[1:]
        
        print("Volatility Forecasts:")
        for i, date in enumerate(forecast_dates):
            print(f"{date.date()}: Conditional Volatility = {np.sqrt(h_forecast[i]):.4f}, "
                  f"Realized Volatility = {np.sqrt(x_forecast[i]):.4f}")

# Save results to files
        params_file, historical_file, forecast_file, raw_data_file = save_results(
            model, h_forecast, x_forecast, daily_returns, daily_rv, forecast_dates
        )
        
        # Plot and save results
        plot_file = plot_results(daily_returns, daily_rv, model, h_forecast, x_forecast, forecast_dates)
        
        print(f"RealGARCH analysis completed successfully!")
        
        return model, h_forecast, x_forecast, daily_returns, daily_rv
    
    except Exception as e:
        print(f"Error in RealGARCH analysis: {str(e)}")
        traceback.print_exc()
        
        # Create error log file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        error_file = os.path.join(OUTPUT_DIR, f"realgarch_error_{timestamp}.txt")
        with open(error_file, 'w') as f:
            f.write(f"Error occurred at {datetime.datetime.now()}\n\n")
            f.write(f"Error message: {str(e)}\n\n")
            f.write("Traceback:\n")
            traceback.print_exc(file=f)
        
        print(f"Error details saved to: {error_file}")
        
        return None, None, None, None, None

if __name__ == "__main__":
    print("\n" + "="*80)
    print(f"RealGARCH Model Implementation")
    print(f"Started at: {datetime.datetime.now()}")
    print("="*80 + "\n")
    
    model, h_forecast, x_forecast, daily_returns, daily_rv = run_realgarch_analysis()
    
    print("\n" + "="*80)
    print(f"RealGARCH Analysis Complete")
    print(f"Finished at: {datetime.datetime.now()}")
    print("="*80)