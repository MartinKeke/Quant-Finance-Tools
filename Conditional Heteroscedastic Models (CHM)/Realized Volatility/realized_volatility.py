import requests
import sqlite3
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import math
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# MEXC API endpoints
BASE_URL = "https://api.mexc.com"
KLINE_ENDPOINT = "/api/v3/klines"

# Database setup
DB_NAME = "bitcoin_prices.db"

def create_database():
    """Create SQLite database with table for Bitcoin price data"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Create table for 1-minute Bitcoin price data
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS bitcoin_1m (
        timestamp INTEGER PRIMARY KEY,
        open REAL,
        high REAL, 
        low REAL,
        close REAL,
        volume REAL,
        date_time TEXT
    )
    ''')
    
    # Create table for calculated realized volatility
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS realized_volatility (
        date TEXT PRIMARY KEY,
        daily_rv REAL,
        weekly_rv REAL,
        monthly_rv REAL
    )
    ''')
    
    conn.commit()
    conn.close()
    print("Database created successfully")

def fetch_kline_data(symbol="BTCUSDT", interval="1m", limit=1000, start_time=None, end_time=None):
    """Fetch kline (candlestick) data from MEXC API"""
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time
        
    try:
        response = requests.get(f"{BASE_URL}{KLINE_ENDPOINT}", params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching data: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return None

def save_to_database(kline_data):
    """Save kline data to SQLite database"""
    if not kline_data:
        print("No data to save")
        return
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    for candle in kline_data:
        timestamp = candle[0]  # Open time
        open_price = float(candle[1])
        high_price = float(candle[2])
        low_price = float(candle[3])
        close_price = float(candle[4])
        volume = float(candle[5])
        date_time = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')
        
        # Insert into database, ignore if duplicate timestamp
        cursor.execute('''
        INSERT OR IGNORE INTO bitcoin_1m 
        (timestamp, open, high, low, close, volume, date_time) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, open_price, high_price, low_price, close_price, volume, date_time))
    
    conn.commit()
    conn.close()
    print(f"Saved {len(kline_data)} candles to database")

def load_data_from_db(days=30):
    """Load data from SQLite database for the specified number of days"""
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

def calculate_daily_weekly_monthly_rv(df):
    """Calculate realized volatility for different time windows"""
    if df is None or df.empty:
        return None
    
    # Calculate log returns
    returns_df = calculate_returns(df)
    
    # Calculate realized volatility for different periods
    daily_rv = calculate_realized_volatility(returns_df, window_minutes=1440)  # 1 day = 1440 minutes
    weekly_rv = calculate_realized_volatility(returns_df, window_minutes=10080)  # 1 week = 10080 minutes
    monthly_rv = calculate_realized_volatility(returns_df, window_minutes=43200)  # 1 month (30 days) = 43200 minutes
    
    # Combine the results
    result_df = pd.DataFrame({
        'datetime': daily_rv['datetime'],
        'daily_rv': daily_rv['realized_vol'],
        'weekly_rv': weekly_rv['realized_vol'],
        'monthly_rv': monthly_rv['realized_vol']
    })
    
    return result_df

def save_volatility_to_db(vol_df):
    """Save calculated volatility to database"""
    if vol_df is None or vol_df.empty:
        print("No volatility data to save")
        return
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Get only the data for each day (one entry per day)
    vol_df['date'] = vol_df['datetime'].dt.date.astype(str)
    daily_summary = vol_df.groupby('date').last().reset_index()
    
    for _, row in daily_summary.iterrows():
        date = row['date']
        daily_rv = row['daily_rv']
        weekly_rv = row['weekly_rv']
        monthly_rv = row['monthly_rv']
        
        # Insert or update volatility in database
        cursor.execute('''
        INSERT OR REPLACE INTO realized_volatility 
        (date, daily_rv, weekly_rv, monthly_rv) 
        VALUES (?, ?, ?, ?)
        ''', (date, daily_rv, weekly_rv, monthly_rv))
    
    conn.commit()
    conn.close()
    print(f"Saved volatility data for {len(daily_summary)} days")

def plot_volatility(vol_df, window='daily'):
    """Plot the realized volatility over time"""
    if vol_df is None or vol_df.empty:
        print("No data to plot")
        return
    
    plt.figure(figsize=(12, 6))
    
    if window == 'daily':
        plt.plot(vol_df['datetime'], vol_df['daily_rv'], label='Daily Realized Volatility')
    elif window == 'weekly':
        plt.plot(vol_df['datetime'], vol_df['weekly_rv'], label='Weekly Realized Volatility')
    elif window == 'monthly':
        plt.plot(vol_df['datetime'], vol_df['monthly_rv'], label='Monthly Realized Volatility')
    else:
        plt.plot(vol_df['datetime'], vol_df['daily_rv'], label='Daily Realized Volatility')
        plt.plot(vol_df['datetime'], vol_df['weekly_rv'], label='Weekly Realized Volatility')
        plt.plot(vol_df['datetime'], vol_df['monthly_rv'], label='Monthly Realized Volatility')
    
    plt.title(f'Bitcoin Realized Volatility ({window.capitalize()})')
    plt.xlabel('Date')
    plt.ylabel('Realized Volatility (Annualized)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot to file
    plt.savefig(f'bitcoin_rv_{window}.png')
    plt.close()

def main():
    # Create database if it doesn't exist
    create_database()
    
    # Data collection loop - runs every hour to collect the latest data
    while True:
        try:
            print(f"Fetching Bitcoin price data at {datetime.now()}")
            
            # Fetch the latest 1000 1-minute candles
            kline_data = fetch_kline_data(symbol="BTCUSDT", interval="1m", limit=1000)
            
            if kline_data:
                # Save to database
                save_to_database(kline_data)
                
                # Load all data from the last 30 days
                df = load_data_from_db(days=30)
                
                if df is not None and not df.empty:
                    # Calculate and save volatility
                    vol_df = calculate_daily_weekly_monthly_rv(df)
                    save_volatility_to_db(vol_df)
                    
                    # Plot the volatility (comment out to avoid excessive plotting)
                    plot_volatility(vol_df, 'all')
            
            # For historical data collection, wait 1 minute between calls
            print(f"Waiting for 60 seconds until next data collection...")
            time.sleep(60)
        except Exception as e:
            print(f"Error in main loop: {str(e)}")
            # Wait a bit before retrying
            time.sleep(300)

if __name__ == "__main__":
    main()