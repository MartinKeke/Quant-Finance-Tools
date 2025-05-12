import requests
import sqlite3
import time
import os
import subprocess
from datetime import datetime

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
    
    conn.commit()
    conn.close()
    print("Database checked/created successfully")

def fetch_kline_data(symbol="BTCUSDT", interval="1m", limit=1000):
    """Fetch kline (candlestick) data from MEXC API"""
    base_url = "https://api.mexc.com"
    endpoint = "/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    
    try:
        response = requests.get(f"{base_url}{endpoint}", params=params)
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
    
    records_added = 0
    for candle in kline_data:
        timestamp = candle[0]  # Open time
        open_price = float(candle[1])
        high_price = float(candle[2])
        low_price = float(candle[3])
        close_price = float(candle[4])
        volume = float(candle[5])
        date_time = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')
        
        # Insert into database, ignore if duplicate timestamp
        try:
            cursor.execute('''
            INSERT OR IGNORE INTO bitcoin_1m 
            (timestamp, open, high, low, close, volume, date_time) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, open_price, high_price, low_price, close_price, volume, date_time))
            if cursor.rowcount > 0:
                records_added += 1
        except sqlite3.Error as e:
            print(f"Database error: {e}")
    
    conn.commit()
    conn.close()
    print(f"Saved {records_added} new candles to database")

def update_github():
    """Create simple daily BTC summary and push to GitHub"""
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Make sure we have a directory for our summaries
    os.makedirs("daily_summaries", exist_ok=True)
    summary_file = f"daily_summaries/{today}_btc_summary.txt"
    
    # Connect to database and get simple stats
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Simple query for basic daily stats
    cursor.execute('''
    SELECT 
        MIN(low) as daily_low,
        MAX(high) as daily_high,
        (SELECT close FROM bitcoin_1m ORDER BY timestamp DESC LIMIT 1) as latest_price
    FROM bitcoin_1m 
    WHERE date_time >= date('now', '-1 day')
    ''')
    
    stats = cursor.fetchone()
    conn.close()
    
    # Write simple stats to file
    with open(summary_file, 'w') as f:
        f.write(f"Bitcoin Summary for {today}\n")
        f.write(f"Daily Low: ${stats[0]}\n")
        f.write(f"Daily High: ${stats[1]}\n")
        f.write(f"Latest Price: ${stats[2]}\n")
    
    # Commit and push to GitHub
    try:
        subprocess.run(['git', 'add', summary_file])
        subprocess.run(['git', 'commit', '-m', f'Add BTC summary for {today}'])
        subprocess.run(['git', 'push', 'origin', 'main'])
        print(f"Successfully updated GitHub with daily summary")
    except Exception as e:
        print(f"GitHub update failed: {str(e)}")

def main():
    # Create database if it doesn't exist
    create_database()
    
    # Data collection loop - runs every minute
    while True:
        try:
            current_time = datetime.now()
            print(f"Fetching Bitcoin price data at {current_time}")
            
            # Fetch the latest 1000 1-minute candles
            kline_data = fetch_kline_data(symbol="BTCUSDT", interval="1m", limit=1000)
            
            if kline_data:
                # Save to database
                save_to_database(kline_data)
            
            # Check if it's midnight to update GitHub
            if current_time.hour == 0 and current_time.minute == 0:
                update_github()
            
            # Wait for 60 seconds before the next fetch
            print(f"Waiting for 60 seconds until next data collection...")
            time.sleep(60)
        except Exception as e:
            print(f"Error in main loop: {str(e)}")
            # Wait a bit before retrying
            time.sleep(60)

if __name__ == "__main__":
    main()