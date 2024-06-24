import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm
from scipy.stats import t

btc = yf.download('BTC-USD', start='2014-10-01', end='2024-02-10')
btc['Log Returns'] = np.log(btc['Adj Close']) - np.log(btc['Adj Close']).shift(1)
btc.drop(btc.index[0], axis=0, inplace=True)
# btc['Log Returns'].diff()
# btc['Log Returns'] = btc['Log Returns'].rolling(window=500).mean()


def plot_ticker_price(ticker):
    plt.plot(ticker.index, ticker['Adj Close'], color='blue')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('BTC/USD Price Chart')
    plt.show()

def plot_ticker_log(ticker):
    plt.plot(ticker.index, ticker['Log Returns'], color='blue')
    plt.xlabel('Date')
    plt.ylabel('Log Returns')
    plt.title('BTC/USD Log Returns')
    plt.show()
# plot_ticker_log(btc)
def decompose_data(ticker):
    decompose_data = seasonal_decompose(ticker['Adj Close'].dropna(), model = 'additive', period=365, extrapolate_trend='freq')
    daily_frequency = ticker.asfreq(freq='D')
    fig, axes = plt.subplots(4, 1, sharex=True)

    decompose_data.observed.plot(ax=axes[0], legend=False, color="blue")
    axes[0].set_ylabel('Observed')
    decompose_data.trend.plot(ax=axes[1], legend=False, color="blue")
    axes[1].set_ylabel('Trend')
    decompose_data.seasonal.plot(ax=axes[2], legend=False, color="blue")
    axes[2].set_ylabel('Seasonal')
    decompose_data.resid.plot(ax=axes[3], legend=False, color="blue")
    axes[3].set_ylabel('Residual')
    plt.show()

# decompose_data(btc)

def stationarity_test(ticker):
    ticker['Log Returns'] = ticker['Log Returns']
    dftest = adfuller(ticker['Log Returns'].dropna(), autolag = 'AIC')
    print("1. ADF : ", dftest[0])
    print("2. P-Value : ", dftest[1])
    print("3. Num Of Lags : ", dftest[2])
    print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
    print("5. Critical Values :")
    for key, val in dftest[4].items():
        print("\t",key, ": ", val)

# stationarity_test(btc)

def acf_plot(ticker):
    fig, ax = plt.subplots(figsize=(12,8))
    plot_acf(ticker['Log Returns'].diff().dropna(), lags=58, alpha=0.05, ax=ax, color='blue')
    ax.set_title('Autocorrelation Function with 95% Confidence Interval')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    plt.show()

# acf_plot(btc)

def pacf_plot(ticker):
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_pacf(ticker['Log Returns'].diff().dropna(), lags=58, alpha=0.05, ax=ax, color='blue')
    ax.set_title('Partial Autocorrelation Function with 95% Confidence Interval')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Partial Autocorrelation')
    plt.show()
#pacf_plot(btc)

def model_arima(ticker, train, test, params):
    model = ARIMA(train, order=params)
    model_fit = model.fit()
    forecast_series = model_fit.forecast(30, alpha=0.05)
    forecast = model_fit.get_forecast(30)
    ci = forecast.conf_int(alpha=0.05)

    plt.figure(figsize=(12,6))
    train.plot(color='blue', label='Training Data (Log Returns)')
    test.plot(color='green', label='Actual Log Returns')
    plt.plot(test.index, forecast_series, label='forecast', color='red')
    plt.plot(test.index, ci.iloc[:, 0], label='Lower Confidence Interval', color='gray')
    plt.plot(test.index, ci.iloc[:, 1], label='Upper Confidence Interval', color='gray')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Log Returns')
    plt.title('BTC/USD ARIMA Forecasted vs Actual Log Returns with 95% Confidence Interval')
    plt.show()

def model_close_arima(ticker, train, test, params):
    model = ARIMA(train, order=params)
    model_fit = model.fit()
    forecast = model_fit.get_forecast(30)
    ci = forecast.conf_int(alpha=0.05)
    forecast_series = model_fit.forecast(30, alpha=0.05)
    plt.figure(figsize=(12,6))
    test.plot(color='green', label='Actual Log Returns')
    plt.plot(test.index, forecast_series, label='forecast', color='red')
    plt.plot(test.index, ci.iloc[:, 0], label='Lower Confidence Interval', color='gray')
    plt.plot(test.index, ci.iloc[:, 1], label='Upper Confidence Interval', color='gray')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Log Returns')
    plt.title('BTC/USD ARIMA Close-up Forecasted vs Actual Log Returns')
    plt.show()

def sarima_model(ticker, train, test, params, seasonal_params):
    model = SARIMAX(train, order=params, seasonal_order=seasonal_params)
    model_fit = model.fit()
    ticker['forecasting sarima'] = model_fit.predict(start=3389, end=3419)
    
    forecast = model_fit.get_forecast(steps=30)
    ci = forecast.conf_int(alpha=0.05)
    plt.figure(figsize=(12, 8))
    plt.plot(ticker.index, ticker['Log Returns'], label='Log Returns', color = "blue" )
    plt.plot(test.index, test, label='Actual Log Returns', color = "green")
    plt.plot(ci.iloc[:, 0], label='Lower Confidence Interval', color='gray')
    plt.plot(ci.iloc[:, 1], label='Upper Confidence Interval', color='gray')
    plt.plot(ticker['forecasting sarima'], label='Forecasted Log Returns', color = "red")
    plt.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color='gray', alpha=0.25)
    plt.title('BTC/USD SARIMA Forecasted vs Actual Log Returns with 95% Confidence Interval')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Log Returns')
    plt.show()

def model_close_sarima(ticker, train, test, params, seasonal_params):
    model = SARIMAX(train, order=params, seasonal_order=seasonal_params)
    model_fit = model.fit()
    forecast = model_fit.get_forecast(30)
    ci = forecast.conf_int(alpha=0.05)
    forecast_series = model_fit.forecast(30, alpha=0.05)
    plt.figure(figsize=(12,6))
    test.plot(color='green', label='Actual Log Returns')
    plt.plot(test.index, forecast_series, label='forecast', color='red')
    plt.plot(test.index, ci.iloc[:, 0], label='Lower Confidence Interval', color='gray')
    plt.plot(test.index, ci.iloc[:, 1], label='Upper Confidence Interval', color='gray')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Log Returns')
    plt.title('SARIMA Close-up Forecasted vs Actual Log Returns')
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
test_data = btc['Log Returns'][3388:]
train_data.dropna(inplace=True)
# arima_model = pm.auto_arima(train_data, seasonal=False, stepwise=False)
# sarima_model = pm.auto_arima(train_data, start_p=1,start_q=1,test='adf',m=6,seasonal=True,trace=True)
# # print(arima_model.summary())
# print(sarima_model.summary())
# order = sarima_model.order
# seasonal_order = sarima_model.seasonal_order

# print(f"Order (p, d, q): {order}")
# print(f"Seasonal Order (P, D, Q, m): {seasonal_order}")
# model_arima(btc, train_data, test_data, (1,0,2))
# model_close_arima(btc, train_data, test_data, (1,0,2))
arima_diagnostics(btc, train_data, test_data, (1,0,2))

# sarima_model(btc, train_data, test_data, (0,0,0), (0,0,1,6))
sarima_diagnostics(btc, train_data, test_data, (0,0,0), (0,0,1,6))
# model_close_sarima(btc, train_data, test_data, (0,0,0), (0,0,1,6))

# model_arima(btc, train_data, test_data, (0,1,0))
