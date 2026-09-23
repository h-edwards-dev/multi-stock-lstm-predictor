

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

plt.style.use('fivethirtyeight')

# Config

TICKERS = ['MSFT', 'AAPL', 'GOOG', 'AMZN']   # <-- add/remove tickers here
START_DATE = '2012-01-01'
END_DATE = '2019-12-31'
WINDOW = 60          # days of history used per prediction
TRAIN_FRACTION = 0.8
EPOCHS = 1           # bump this up for real use (1 is just for speed/testing)
BATCH_SIZE = 1


def fetch_data(ticker, start, end):
    """Download and flatten Close price data for one ticker."""
    df = yf.download(ticker, start, end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.filter(['Close'])


def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def make_training_set(scaled_data, training_data_len, window):
    train_data = scaled_data[0:training_data_len]
    x_train, y_train = [], []
    for i in range(window, len(train_data)):
        x_train.append(train_data[i - window:i, 0])
        y_train.append(train_data[i, 0])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train


def make_test_set(scaled_data, dataset, training_data_len, window):
    test_data = scaled_data[training_data_len - window:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(window, len(test_data)):
        x_test.append(test_data[i - window:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return x_test, y_test


def run_for_ticker(ticker):
    print(f'\n=== {ticker} ===')

    data = fetch_data(ticker, START_DATE, END_DATE)
    dataset = data.values
    training_data_len = math.ceil(len(dataset) * TRAIN_FRACTION)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    x_train, y_train = make_training_set(scaled_data, training_data_len, WINDOW)

    model = build_model((x_train.shape[1], 1))
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    x_test, y_test = make_test_set(scaled_data, dataset, training_data_len, WINDOW)
    predictions = model.predict(x_test, verbose=0)
    predictions = scaler.inverse_transform(predictions)

    rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
    print(f'{ticker} RMSE: {rmse:.2f}')

    valid = data[training_data_len:].copy()
    valid['Predictions'] = predictions

    # next-day prediction using the most recent WINDOW days
    last_window = data[-WINDOW:].values
    last_window_scaled = scaler.transform(last_window)
    X_next = np.array([last_window_scaled])
    X_next = np.reshape(X_next, (X_next.shape[0], X_next.shape[1], 1))
    next_price = scaler.inverse_transform(model.predict(X_next, verbose=0))[0][0]
    print(f'{ticker} predicted next close: {next_price:.2f}')

    return {
        'ticker': ticker,
        'train': data[0:training_data_len],
        'valid': valid,
        'rmse': rmse,
        'next_price': next_price,
    }


def plot_all(results):
    n = len(results)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows))
    axes = np.array(axes).reshape(-1)  # flatten in case of 1 row

    for ax, res in zip(axes, results):
        ax.set_title(f"{res['ticker']} (RMSE {res['rmse']:.2f})")
        ax.plot(res['train']['Close'], label='Train')
        ax.plot(res['valid']['Close'], label='Valid')
        ax.plot(res['valid']['Predictions'], label='Predictions')
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price USD ($)')
        ax.legend(loc='lower right')

    # hide any unused subplot axes
    for ax in axes[n:]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('all_stocks_predictions.png', dpi=150)
    plt.show()


def main():
    results = [run_for_ticker(t) for t in TICKERS]

    plot_all(results)

    print('\n=== Summary ===')
    summary = pd.DataFrame([
        {'Ticker': r['ticker'], 'RMSE': r['rmse'], 'Predicted Next Close': r['next_price']}
        for r in results
    ])
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()