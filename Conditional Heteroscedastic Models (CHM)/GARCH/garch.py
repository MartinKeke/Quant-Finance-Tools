import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import datetime
import os

class GARCH11:
    def __init__(self, returns=None):
        """Initialize the GARCH(1,1) model"""
        self.returns = returns if returns is not None else np.array([])
        self.params = None  # Will store [alpha0, alpha1, beta1]
        
    def log_likelihood(self, params):
        """Negative log-likelihood function for GARCH(1,1)"""
        alpha0, alpha1, beta1 = params
        
        # Check parameter constraints
        if alpha0 <= 0 or alpha1 < 0 or beta1 < 0 or alpha1 + beta1 >= 1:
            return np.inf
        
        T = len(self.returns)
        if T == 0:
            return np.inf
            
        sigma2 = np.zeros(T)
        
        # Initialize variance with sample variance
        sigma2[0] = np.var(self.returns) if T > 1 else self.returns[0]**2
        
        # Compute variance series
        for t in range(1, T):
            sigma2[t] = alpha0 + alpha1 * self.returns[t-1]**2 + beta1 * sigma2[t-1]
        
        # Log-likelihood
        llh = -0.5 * np.sum(np.log(sigma2) + self.returns**2 / sigma2)
        
        return -llh  # Return negative since we minimize
    
    def fit(self, initial_params=None):
        """Estimate GARCH(1,1) parameters via MLE"""
        if len(self.returns) <= 1:
            raise ValueError("Need at least 2 return observations to fit GARCH model")
            
        if initial_params is None:
            # Default initial parameters
            initial_params = [0.01, 0.1, 0.8]  # [alpha0, alpha1, beta1]
        
        # Optimize negative log-likelihood
        result = minimize(self.log_likelihood, 
                         initial_params, 
                         method='L-BFGS-B',
                         bounds=((1e-6, None), (0, 1), (0, 1)))
        
        if result.success:
            self.params = result.x
            return self.params
        else:
            raise Exception(f"Parameter estimation failed! {result.message}")
    
    def compute_volatility(self, returns=None):
        """Compute historical volatility using estimated parameters"""
        if returns is None:
            returns = self.returns
            
        if self.params is None:
            raise ValueError("Model parameters not yet estimated. Call fit() first.")
            
        alpha0, alpha1, beta1 = self.params
        T = len(returns)
        
        if T <= 0:
            return np.array([])
            
        sigma2 = np.zeros(T)
        sigma2[0] = np.var(returns) if T > 1 else returns[0]**2
        
        for t in range(1, T):
            sigma2[t] = alpha0 + alpha1 * returns[t-1]**2 + beta1 * sigma2[t-1]
            
        return np.sqrt(sigma2)  # Return volatility (not variance)
    
    def forecast_volatility(self, steps, returns=None):
        """Forecast future volatility for 'steps' periods"""
        if returns is None:
            returns = self.returns
            
        if len(returns) <= 0:
            return np.array([])
            
        if self.params is None:
            raise ValueError("Model parameters not yet estimated. Call fit() first.")
            
        alpha0, alpha1, beta1 = self.params
        
        # Compute equilibrium variance h² = α₀/(1-β₁) as in equation (2.8)
        h_squared_eq = alpha0 / (1 - beta1)
        
        # Start with last observed variance
        sigma2 = np.zeros(steps + 1)
        last_return_squared = returns[-1]**2
        last_variance = np.var(returns) if len(returns) > 1 else last_return_squared
        
        sigma2[0] = alpha0 + alpha1 * last_return_squared + beta1 * last_variance
        
        # Generate forecasts
        for t in range(1, steps + 1):
            # For longer horizons, variance approaches the equilibrium level
            sigma2[t] = alpha0 + (alpha1 + beta1) * sigma2[t-1]
            
        return np.sqrt(sigma2[1:])  # Return volatility forecasts
    
    def simulate_path(self, steps, seed=None):
        """Simulate future returns and volatility paths"""
        if len(self.returns) <= 0:
            return np.array([]), np.array([])
            
        if self.params is None:
            raise ValueError("Model parameters not yet estimated. Call fit() first.")
            
        if seed is not None:
            np.random.seed(seed)
            
        alpha0, alpha1, beta1 = self.params
        
        # Initialize arrays
        simulated_returns = np.zeros(steps)
        simulated_h2 = np.zeros(steps)
        
        # Start with last observed variance or equilibrium variance
        h_squared_eq = alpha0 / (1 - beta1)
        last_return_squared = self.returns[-1]**2
        last_variance = np.var(self.returns) if len(self.returns) > 1 else last_return_squared
        
        simulated_h2[0] = alpha0 + alpha1 * last_return_squared + beta1 * last_variance
        
        # Generate random standard normal innovations
        epsilon = np.random.standard_normal(steps)
        
        # Simulate path according to equations (2.5)-(2.7)
        for t in range(steps):
            h_t = np.sqrt(simulated_h2[t])
            simulated_returns[t] = h_t * epsilon[t]  # Equation (2.5)
            
            if t < steps - 1:
                # Update volatility using equation (2.6)
                simulated_h2[t+1] = alpha0 + alpha1 * simulated_returns[t]**2 + beta1 * simulated_h2[t]
                
        return simulated_returns, np.sqrt(simulated_h2)


def load_bitcoin_data(file_path='Bitcoin Historical Data.csv'):
    """Load Bitcoin data from CSV file"""
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Convert the 'Price' column to numeric
        # Remove any commas and convert to float
        df['Price'] = df['Price'].str.replace(',', '').astype(float)
        
        # Convert the 'Date' column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by date (oldest to newest)
        df = df.sort_values('Date')
        
        # Calculate daily returns
        df['Return'] = df['Price'].pct_change()
        
        # Drop rows with NaN returns
        df = df.dropna(subset=['Return'])
        
        print(f"Successfully loaded Bitcoin data with {len(df)} observations")
        return df
        
    except Exception as e:
        print(f"Error loading Bitcoin data: {str(e)}")
        raise


# Example usage
def main():
    try:
        print("Loading Bitcoin historical data...")
        
        # Create directory for saving figures if it doesn't exist
        output_dir = "garch_results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        
        # Load Bitcoin data from CSV
        bitcoin_data = load_bitcoin_data()
        
        # Get returns
        returns = bitcoin_data['Return'].values
        
        print(f"Successfully processed {len(returns)} return observations")
        
        # Initialize and fit GARCH model
        garch_model = GARCH11(returns)
        params = garch_model.fit()
        alpha0, alpha1, beta1 = params
        
        print(f"\nEstimated GARCH(1,1) parameters for Bitcoin:")
        print(f"alpha0 = {alpha0:.6f}")
        print(f"alpha1 = {alpha1:.6f}")
        print(f"beta1 = {beta1:.6f}")
        print(f"Estimated equilibrium volatility = {np.sqrt(alpha0/(1-alpha1-beta1)):.6f}")
        
        # Compute historical volatility
        volatility = garch_model.compute_volatility()
        
        # Forecast future volatility (30 days)
        forecast_days = 30
        forecast_vol = garch_model.forecast_volatility(forecast_days)
        
        # Simulate future paths
        n_simulations = 5
        plt.figure(figsize=(14, 10))
        
        # Plot price
        plt.subplot(3, 1, 1)
        plt.plot(bitcoin_data['Date'].values[-252:], bitcoin_data['Price'].values[-252:], 'b-', label='Bitcoin Price')
        plt.title('Bitcoin Price (Last 252 Days)')
        plt.legend()
        
        # Plot historical volatility
        plt.subplot(3, 1, 2)
        plt.plot(bitcoin_data['Date'].values[-252:], volatility[-252:]*100, 'r-', label='Estimated Volatility (%)')
        
        # Create dates for forecast
        last_date = bitcoin_data['Date'].iloc[-1]
        forecast_dates = [last_date + datetime.timedelta(days=i) for i in range(1, forecast_days+1)]
        
        plt.plot(forecast_dates, forecast_vol*100, 'r--', label='Forecasted Volatility (%)')
        plt.legend()
        plt.title('GARCH(1,1) Volatility')
        
        # Plot simulated paths
        plt.subplot(3, 1, 3)
        for i in range(n_simulations):
            sim_returns, sim_vol = garch_model.simulate_path(forecast_days, seed=42+i)
            plt.plot(forecast_dates, sim_vol*100, label=f'Simulation {i+1}')
        
        plt.plot(forecast_dates, forecast_vol*100, 'k--', linewidth=2, label='Expected Volatility')
        plt.legend()
        plt.title('Simulated Volatility Paths (%)')
        
        plt.tight_layout()
        
        # Save figure with current date in filename
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        figure_path = os.path.join(output_dir, f"bitcoin_garch_analysis_{current_date}.png")
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {figure_path}")
        
        # Also save parameters and forecasts to a text file
        results_path = os.path.join(output_dir, f"bitcoin_garch_results_{current_date}.txt")
        with open(results_path, 'w') as f:
            f.write(f"GARCH(1,1) Analysis for Bitcoin - {datetime.datetime.now().strftime('%Y-%m-%d')}\n")
            f.write("=" * 50 + "\n\n")
            f.write("Model Parameters:\n")
            f.write(f"alpha0 = {alpha0:.6f}\n")
            f.write(f"alpha1 = {alpha1:.6f}\n")
            f.write(f"beta1 = {beta1:.6f}\n")
            f.write(f"Equilibrium volatility = {np.sqrt(alpha0/(1-alpha1-beta1)):.6f}\n\n")
            
            f.write("Volatility Forecast (30 days):\n")
            for i, vol in enumerate(forecast_vol):
                f.write(f"Day {i+1}: {vol*100:.2f}%\n")
        
        print(f"Results saved to: {results_path}")
        
        # Show the plot
        plt.show()
        
        # Print forecast values to console
        print("\nForecasted Bitcoin Volatility (next 30 days):")
        for i, vol in enumerate(forecast_vol):
            print(f"Day {i+1}: {vol*100:.2f}%")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()