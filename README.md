# Bitcoin Price Prediction and Trading Signal System

This project provides real-time Bitcoin price predictions, technical analysis indicators, and trading signals. It combines several machine learning and deep learning models to predict Bitcoin's next price movement, generate trading signals, and forecast trends. The project uses Yahoo Finance for live Bitcoin price data and includes both price and hybrid prediction models as well as technical indicators like RSI, MACD, SMA, and Bollinger Bands.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Model Details](#model-details)
5. [Technical Indicators](#technical-indicators)
6. [Real-Time Prediction](#real-time-prediction)
7. [Notes](#notes)

## Project Structure
- `bitcoin_price_prediction_model.h5`: Trained model for price prediction.
- `trading_signal_model_gbm.pkl`: GBM model for generating trading signals.
- `hybrid_trend_prediction.h5`: Hybrid model for predicting trend direction.
- `script.py`: Main script that performs real-time prediction and trading signal generation.

## Installation

To run this project, you will need the following libraries:
```bash
pip install yfinance numpy pandas scikit-learn tensorflow keras joblib



markdown
Copy code
# Bitcoin Price Prediction and Trading Signal System

This project provides real-time Bitcoin price predictions, technical analysis indicators, and trading signals. It combines several machine learning and deep learning models to predict Bitcoin's next price movement, generate trading signals, and forecast trends. The project uses Yahoo Finance for live Bitcoin price data and includes both price and hybrid prediction models as well as technical indicators like RSI, MACD, SMA, and Bollinger Bands.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Model Details](#model-details)
5. [Technical Indicators](#technical-indicators)
6. [Real-Time Prediction](#real-time-prediction)
7. [Notes](#notes)

## Project Structure
- `bitcoin_price_prediction_model.h5`: Trained model for price prediction.
- `trading_signal_model_gbm.pkl`: GBM model for generating trading signals.
- `hybrid_trend_prediction.h5`: Hybrid model for predicting trend direction.
- `script.py`: Main script that performs real-time prediction and trading signal generation.

## Installation

To run this project, you will need the following libraries:
```bash
pip install yfinance numpy pandas scikit-learn tensorflow keras joblib


## Usage
Download the Required Models
Make sure you have the .h5 and .pkl model files in the same directory as the script.


## Model Details
- `Models Used:

- `Bitcoin Price Prediction Model (bitcoin_price_prediction_model.h5): Uses a time-series model to predict the Bitcoin price for the next minute.
- `Trading Signal Model (trading_signal_model_gbm.pkl): A GBM (Gradient Boosting Machine) that generates trading signals (Buy or Sell) based on technical indicators.
- `Hybrid Trend Prediction Model (hybrid_trend_prediction.h5): A model that uses price predictions and additional features to forecast the overall trend (Increase or Decrease).

- `Data Source:

- `Yahoo Finance: Real-time Bitcoin prices are fetched using the yfinance library, updating every minute.

## Technical Indicators
- `The project calculates various technical indicators to assess Bitcoin’s price action, including:

- `Relative Strength Index (RSI): Measures price momentum.
- `Simple Moving Average (SMA): Calculates average prices over 50 and 200 periods.
- `Exponential Moving Average (EMA): Calculates weighted average prices with a span of 20 periods.
- `Bollinger Bands: Calculates upper and lower bounds based on a 20-period moving average.
- `Moving Average Convergence Divergence (MACD): Measures the difference between short-term and long-term EMAs.
- `Stochastic Oscillator: Shows where the last closing price was relative to the recent high/low range.
- `Average True Range (ATR): Measures market volatility.

## Real-Time Prediction
- `The predict_real_time() function performs continuous Bitcoin price predictions every minute and calculates the following outputs:

- `Current and Next Minute Price Prediction: Predicts Bitcoin price for the current and next minute using price_model.
- `Trading Signal: Generates a trading signal (Buy or Sell) based on signal_model predictions.
- `Trend Prediction: Predicts overall trend direction (Increase or Decrease) using hybrid_model.



![image](https://github.com/user-attachments/assets/ff63754d-a69d-489a-9105-7349bec5722f)


![image](https://github.com/user-attachments/assets/05cd884e-1b4a-432a-8b0a-a82ab6ac96ad)


![image](https://github.com/user-attachments/assets/318035c3-8725-49be-a391-b07e5c966b54)


