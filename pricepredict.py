# pricepredict.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Step 1: Download stock data
# Fix index frequency to avoid warnings
stock = yf.download('AAPL', start='2015-01-01', end='2024-12-31', auto_adjust=False)
stock = stock[['Close']]
stock.index = pd.date_range(start=stock.index[0], periods=len(stock), freq='B')
print(stock)

# Step 2: Plot close price
plt.figure(figsize=(12,6))
plt.plot(stock['Close'], label='AAPL Close')
plt.title('AAPL Stock Price (2015–2024)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(True)
plt.legend()
plt.show()

# Step 3: Train/Test Split
train = stock[:'2022']
test = stock['2023':]

# Step 4: Fit ARIMA model
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()
print(model_fit)
# Step 5: Forecast
forecast = model_fit.predict(start=len(train), end=len(train)+len(test)-1)
forecast.index = test.index  # align forecast index to test

# Step 6: Plot actual vs forecast
plt.figure(figsize=(12,6))
plt.plot(train['Close'], label='Train')
plt.plot(test['Close'], label='Test', color='green')
plt.plot(forecast, label='Forecast', color='red')
plt.title('Stock Price Forecast using ARIMA')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.show()

# Step 7: Evaluate the model (RMSE)
test_clean = test['Close'].dropna()
forecast_clean = forecast.dropna()
test_clean, forecast_clean = test_clean.align(forecast_clean, join='inner', axis=0)

rmse = np.sqrt(mean_squared_error(test_clean, forecast_clean))
print("RMSE:", rmse)

# Step 8: Predict future 30 days
future_model = ARIMA(stock, order=(5, 1, 0))
future_fit = future_model.fit()
future_forecast = future_fit.forecast(steps=30)

future_dates = pd.date_range(start=stock.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')

# Step 9: Plot future forecast
plt.figure(figsize=(12,6))
plt.plot(stock['Close'], label='Historical')
plt.plot(future_dates, future_forecast, label='Next 30 Days Forecast', color='purple')
plt.title('Future 30-Day Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.show()
