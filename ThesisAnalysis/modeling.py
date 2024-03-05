import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima.arima import auto_arima
from scipy.stats import t

btc = yf.download('BTC-USD', start='2014-10-01', end='2024-02-10')
btc['Log Returns'] = np.log(btc['Adj Close']) - np.log(btc['Adj Close']).shift(1)
btc.drop(btc.index[0], axis=0, inplace=True)
btc['Log Returns'] = btc['Log Returns'].rolling(window=500).mean()


def plot_ticker_price(ticker):
    plt.plot(ticker.index, ticker['Adj Close'])
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('BTC/USD Price Chart')
    plt.show()

def plot_ticker_log(ticker):
    plt.plot(ticker.index, ticker['Log Returns'])
    plt.xlabel('Date')
    plt.ylabel('Log Returns')
    plt.title('BTC/USD Log Returns')
    plt.show()

def decompose_data(ticker):
    decompose_data = seasonal_decompose(ticker['Log Returns'].dropna(), model = 'additive', period=365, extrapolate_trend='freq')
    daily_frequency = ticker.asfreq(freq='D')
    decompose_data.plot()
    plt.show()

def stationarity_test(ticker):
    ticker['Log Returns'] = ticker['Log Returns'].diff()
    dftest = adfuller(ticker['Log Returns'].dropna(), autolag = 'AIC')
    print("1. ADF : ", dftest[0])
    print("2. P-Value : ", dftest[1])
    print("3. Num Of Lags : ", dftest[2])
    print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
    print("5. Critical Values :")
    for key, val in dftest[4].items():
        print("\t",key, ": ", val)

def acf_plot(ticker):
    fig, ax = plt.subplots(figsize=(12,8))
    plot_acf(ticker['Log Returns'].diff().dropna(), lags=58, alpha=0.05, ax=ax)
    ax.set_title('Autocorrelation Function with 95% Confidence Interval')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    plt.show()

def pacf_plot(ticker):
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_pacf(ticker['Log Returns'].diff().dropna(), lags=58, alpha=0.05, ax=ax)
    ax.set_title('Partial Autocorrelation Function with 95% Confidence Interval')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Partial Autocorrelation')
    plt.show()

def arima_model(ticker, train, test, params):
    model = ARIMA(train, order=params)
    model_fit = model.fit()
    ticker['forecast log returns'] = model_fit.predict(start=3389, end=3419)
    forecast = model_fit.get_forecast(steps=30)
    ci = forecast.conf_int(alpha=0.05)
    plt.figure(figsize=(12, 8))
    plt.plot(ticker.index, ticker['Log Returns'], label='Log Returns')
    plt.plot(test.index, test, label='Actual Log Returns')
    plt.plot(ci.iloc[:, 0], label='Lower CI', color='gray')
    plt.plot(ci.iloc[:, 1], label='Upper CI', color='gray')
    plt.plot(ticker['forecast log returns'], label='Forecasted Log Returns')
    plt.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color='gray', alpha=0.25)
    plt.title('ARIMA Forecasted Log Returns vs Actual Log Returns with 95% Confidence Interval')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Log Returns')
    plt.show()

def sarima_model(ticker, train, test, params, seasonal_params):
    model = SARIMAX(train, order=params, seasonal_order=seasonal_params)
    model_fit = model.fit()
    ticker['forecasting sarima'] = model_fit.predict(start=3389, end=3419)
    forecast = model_fit.get_forecast(steps=30)
    ci = forecast.conf_int(alpha=0.05)
    plt.figure(figsize=(12, 8))
    plt.plot(ticker.index, ticker['Log Returns'], label='Log Returns')
    plt.plot(test.index, test, label='Actual Log Returns')
    plt.plot(ci.iloc[:, 0], label='Lower CI', color='gray')
    plt.plot(ci.iloc[:, 1], label='Upper CI', color='gray')
    plt.plot(ticker['forecasting sarima'], label='Forecasted Log Returns')
    plt.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color='gray', alpha=0.25)
    plt.title('SARIMA Forecasted Log Returns vs Actual Log Returns with 95% Confidence Interval')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Log Returns')
    plt.show()

def arima_diagnostics(ticker, train, test, params):
    arima_model = ARIMA(train, order=params)
    arima_model_fit = arima_model.fit()
    arima_model_fit.plot_diagnostics(figsize=(12, 8))
    plt.show()

def sarima_diagnostics(ticker, train, test, params, seasonal_params):
    sarima_model = SARIMAX(train, order=params, seasonal_order=seasonal_params)
    sarima_model_fit = sarima_model.fit()
    sarima_model_fit.plot_diagnostics(figsize=(12, 8))
    plt.show()

# ERRORS CHANGE WHEN COMPUTING
def mae_rmse(model, ticker):
    forecast = model.get_forecast(steps=3419-3349)
    rmse = mean_squared_error(ticker['Log Returns'].iloc[3348:3419], forecast.predicted_mean, squared=False)
    mae = mean_absolute_error(ticker['Log Returns'].iloc[3348:3419], forecast.predicted_mean)
    print(f'MAE: {mae}\nRMSE: {rmse}')
    None

def check_parameters(train, test):
    model = auto_arima(train,
                       start_P=1,
                       start_q=1,
                       test='adf',
                       tr=13, max_q=13,
                       d=2,
                       seasonal=True,
                       m=12,
                       suppress_warnings=True)
    print(model.summary())
    print(model.order)
    

train_data = btc['Log Returns'][:3389]
test_data = btc['Log Returns'][3389:]
arima_model(btc, train_data, test_data, (1,1,1))
arima_diagnostics(btc, train_data, test_data, (1,1,1))
sarima_model(btc, train_data, test_data, (1,1,1), (3,1,3,12))
sarima_diagnostics(btc, train_data, test_data, (1,1,1), (3,1,3,12))
