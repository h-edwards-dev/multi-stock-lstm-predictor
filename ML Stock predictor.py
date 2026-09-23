#import libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import yfinance as yf

plt.style.use('fivethirtyeight')
#get data
df = yf.download('MSFT', '2012-01-01', '2019-12-31')
# Flatten MultiIndex columns if present
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
#show data
print(df)

#get number of rows and columns in data set
df.shape

#visualise the closing price history
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.show()

#create new data frame with only close column
data = df.filter(['Close'])
#convert dataframe to a numpy array
dataset = data.values
#get number of rows to train the model on
training_data_len = math.ceil(len(dataset) * 0.8)
#print(training_data_len)
#1609

#scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
#print(scaled_data)

#create training dataset
#create scaled training dataset
train_data = scaled_data[0:training_data_len]
#split the data into x_train and y_train datasets
x_train = []
y_train = []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    #if i<=60:
        #print(x_train)
        #print(y_train)
        #print()

#convert x_train and y_train to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)

#reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#build LSTM model
model = Sequential()
model.add(LSTM(50,return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#train the model
model.fit(x_train, y_train, epochs=1, batch_size=1)

#create the testing data set
#create a new array containing values from index 1543 to 2003
test_data = scaled_data[training_data_len - 60:, :]
#create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

#convert data to a numpy array
x_test = np.array(x_test)

#reshape data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#get the root mean squares error (RMSE)
rmse = np.sqrt(np.mean((predictions - y_test)**2))
print(rmse)

#plot the data
train = data[0:training_data_len]
valid = data[training_data_len:].copy()
valid['Predictions'] = predictions
#visualise data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Valid', 'Predictions'], loc='lower right')
plt.show()

#show the valid and predicted prices
print(valid)

#get the quote
microsoft_quote = yf.download('MSFT', start='2012-01-01', end='2019-12-17')
#create new dataframe
new_df = microsoft_quote.filter(['Close'])
#get the last 60 days closing price values and convert the dataframe to an array
last_60_days = new_df[-60:].values
#scale the values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
#create an empty list
X_test = []
#append the past 60 days to test list
X_test.append(last_60_days_scaled)
#convert X_test dataset to numpy array
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#get the predicted scale price
pred_price = model.predict(X_test)
#undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)

#get the quote
microsoft_quote2 = yf.download('MSFT', start='2019-12-18', end='2019-12-18')
print(microsoft_quote2['Close'])
