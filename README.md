# multi-stock-lstm-predictor
A Python project that trains a separate LSTM (Long Short-Term Memory) neural network for each stock in a list of tickers, then predicts the next closing price for each one.

What it does
Downloads historical daily closing price data for multiple stocks (MSFT, AAPL, GOOG, AMZN) using the yfinance API
Trains an independent LSTM model per ticker, since each stock has its own price scale and behaviour, so a single shared model would blend them incorrectly
Evaluates each model using RMSE (Root Mean Squared Error) against a held-out test set
Plots training data, actual prices, and predicted prices for all stocks in a single grid of subplots
Prints a next-day price prediction for every ticker
How it works
Data pipeline: Closing prices are pulled per ticker, then normalised to a 0–1 range with MinMaxScaler so the neural network trains efficiently.
Windowing: Each training example uses the previous 60 days of prices to predict the next day's price (a sliding window approach standard for time-series forecasting).
Model: A two-layer LSTM network (50 units each) followed by two dense layers, built with Keras.
Training/testing split: 80% of the data trains the model; the remaining 20% is used to test prediction accuracy.
Evaluation: Predictions are compared against actual prices using RMSE, and visualised alongside the training data.
Tech stack
Python
yfinance — historical market data
pandas / numpy — data handling
scikit-learn — feature scaling
keras — LSTM model
matplotlib — visualisation
Results
Ticker	RMSE
MSFT	4.60
AAPL	1.32
GOOG	1.35
AMZN	2.80

Lower RMSE indicates predictions closer to actual closing prices.

A debugging note

The initial version of the test-set construction logic raised an IndexError: tuple index out of range. The cause was an off-by-one slicing issue when building the test window: the slice used to select the test portion of the scaled dataset didn't correctly account for the WINDOW size offset, which meant the array being indexed didn't have the shape the later code expected. Fixing the slice boundary (scaled_data[training_data_len - window:, :]) resolved it. This turned out to be a useful lesson in being precise about array shapes and index boundaries when working with time-series windowing.

Running it
bash
pip install numpy pandas matplotlib yfinance scikit-learn keras
python ml_multi_stock_predictor.py

Edit the TICKERS list at the top of the script to run it against different stocks.

Limitations & possible extensions
EPOCHS = 1 is set for fast iteration/testing; a real model would need many more epochs and likely hyperparameter tuning to be genuinely predictive.
Uses only closing price as a feature; volume, technical indicators, or other stocks' correlated movements could improve accuracy.
Next-day price prediction from historical price patterns alone is a simplified approach; real markets are influenced by many factors this model doesn't capture.
