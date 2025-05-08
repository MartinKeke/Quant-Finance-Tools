# Bitcoin Price Tracker

A simple data collection system that tracks Bitcoin price movements and updates daily summaries.

## Overview

This project continuously collects Bitcoin price data (BTCUSDT) at 1-minute intervals from the MEXC API and stores it in a SQLite database. Every day at midnight, it creates a summary with key statistics and pushes it to this repository, maintaining a historical record of Bitcoin's price action.

## Components

- **Data Collector**: Python script running on an EC2 instance that fetches and stores price data
- **Database**: SQLite database containing detailed 1-minute candlestick data
- **Daily Summaries**: Text files with daily high, low, and closing prices

## Daily Summary Format

Each day, a new summary file is created with the following information:
- Daily low price
- Daily high price
- Latest closing price

## Technologies Used

- Python
- SQLite
- AWS EC2
- MEXC API
- Git

## Future Enhancements

Potential future additions to this project:
- Price visualization dashboards
- Technical indicator calculations
- Correlation analysis with other market factors
- Machine learning model development for pattern recognition

---

*This repository is automatically updated daily via a Python script running on AWS EC2.*