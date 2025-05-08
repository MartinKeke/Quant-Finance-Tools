import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Use a raw string for the database path to avoid escape character issues
DB_NAME = r"C:\Users\Martin\Quant Finance Tools\Conditional Heteroscedastic Models (CHM)\Realized Volatility\bitcoin_prices_test.db"

def load_data(days=30):
    """Load Bitcoin price data from SQLite database"""
    conn = sqlite3.connect(DB_NAME)
    
    # Calculate the timestamp for X days ago
    days_ago = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    query = f"""
    SELECT timestamp, open, high, low, close, volume, date_time 
    FROM bitcoin_1m 
    WHERE timestamp >= {days_ago}
    ORDER BY timestamp ASC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print(f"No data found for the last {days} days")
        return None
    
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    print(f"Loaded {len(df)} data points spanning {(df['datetime'].max() - df['datetime'].min()).total_seconds() / 3600:.1f} hours")
    return df

def calculate_returns(df):
    """Calculate log returns from price data"""
    if df is None or df.empty:
        return None
    
    # Calculate log returns using close price
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df = df.dropna()  # Drop rows with NaN values
    
    return df

def calculate_realized_volatility(returns_df, window_minutes=1440):
    """
    Calculate realized volatility from log returns
    window_minutes: size of the rolling window in minutes (1440 = 1 day)
    """
    if returns_df is None or returns_df.empty:
        return None
    
    # Calculate squared returns
    returns_df['squared_return'] = returns_df['log_return'] ** 2
    
    # Calculate rolling sum of squared returns
    returns_df['rolling_variance'] = returns_df['squared_return'].rolling(window=window_minutes).sum()
    
    # Calculate realized volatility (annualized)
    # For 1-minute data, multiply by sqrt(365*24*60) for annual volatility
    minutes_per_year = 365 * 24 * 60
    returns_df['realized_vol'] = np.sqrt(returns_df['rolling_variance'] * (minutes_per_year / window_minutes))
    
    return returns_df

def analyze_volatility():
    """Analyze Bitcoin volatility"""
    # Load data
    df = load_data(days=30)  # Get up to 30 days of data
    
    if df is None or len(df) < 1440:  # Need at least 1 day of data
        print(f"Not enough data for volatility analysis. Need at least 1440 minutes, have {len(df) if df is not None else 0}.")
        return
    
    # Calculate returns
    returns_df = calculate_returns(df)
    
    # Calculate volatilities
    # Daily volatility (1-day window)
    daily_vol_df = calculate_realized_volatility(returns_df, window_minutes=1440)
    daily_vol = daily_vol_df['realized_vol'].iloc[-1]
    
    # If we have enough data, calculate weekly volatility (7-day window)
    weekly_vol = None
    if len(returns_df) >= 10080:  # 7 days * 24 hours * 60 minutes
        weekly_vol_df = calculate_realized_volatility(returns_df, window_minutes=10080)
        weekly_vol = weekly_vol_df['realized_vol'].iloc[-1]
    
    # If we have enough data, calculate monthly volatility (30-day window)
    monthly_vol = None
    if len(returns_df) >= 43200:  # 30 days * 24 hours * 60 minutes
        monthly_vol_df = calculate_realized_volatility(returns_df, window_minutes=43200)
        monthly_vol = monthly_vol_df['realized_vol'].iloc[-1]
    
    # Print results
    print("\nBitcoin Realized Volatility Analysis:")
    print(f"Data range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Total data points: {len(df)}")
    print(f"Daily realized volatility: {daily_vol:.6f} ({daily_vol*100:.2f}%)")
    if weekly_vol:
        print(f"Weekly realized volatility: {weekly_vol:.6f} ({weekly_vol*100:.2f}%)")
    else:
        print("Weekly realized volatility: Not enough data (need 7 days)")
    if monthly_vol:
        print(f"Monthly realized volatility: {monthly_vol:.6f} ({monthly_vol*100:.2f}%)")
    else:
        print("Monthly realized volatility: Not enough data (need 30 days)")
    
    # Plot volatility over time
    plt.figure(figsize=(12, 6))
    plt.plot(daily_vol_df['datetime'], daily_vol_df['realized_vol'], label='Daily Realized Volatility')
    plt.title('Bitcoin Realized Volatility')
    plt.xlabel('Date')
    plt.ylabel('Realized Volatility (Annualized)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot to file
    plot_filename = f'bitcoin_volatility_{datetime.now().strftime("%Y%m%d_%H%M")}.png'
    plt.savefig(plot_filename)
    print(f"Volatility plot saved as {plot_filename}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    analyze_volatility()