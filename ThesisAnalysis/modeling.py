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
    ticker['forecast log returns'] = model_fit.predict(start=3348, end=3419)
    forecast = model_fit.get_forecast(steps=70)
    ci = forecast.conf_int(alpha=0.05)
    plt.figure(figsize=(12, 8))
    plt.plot(ticker.index, ticker['Log Returns'], label='Log Returns')
    plt.plot(test, label='Actual Log Returns')
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
    ticker['forecasting sarima'] = model_fit.predict(start=3348, end=3419)
    forecast = model_fit.get_forecast(steps=70)
    ci = forecast.conf_int(alpha=0.05)
    plt.figure(figsize=(12, 8))
    plt.plot(ticker.index, ticker['Log Returns'], label='Log Returns')
    plt.plot(test, label='Actual Log Returns')
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
    

train_data = btc['Log Returns'][:3348]
test_data = btc['Log Returns'][3348:]
#arima_model(btc, train_data, test_data, (1,1,1))
sarima_model(btc, train_data, test_data, (1,1,1), (2,1,3,12))
sarima_diagnostics(btc, train_data, test_data, (1,1,1), (2,1,3,12))
# model_arima = ARIMA(train_data, order=(1,1,1))
# model_fit_arima = model_arima.fit()
# mae_rmse(model_fit_arima, btc)
# model_sarima = SARIMAX(train_data, order=(1,1,1), seasonal_order=(1,1,1,12))
# model_fit_sarima = model_sarima.fit()
# mae_rmse(model_fit_sarima, btc)



# model = ARIMA(train_data, order=(3,3,3))
# model_fit = model.fit()
# btc['forecasting return'] = model_fit.predict(start=3348, end=3419)
# plt.figure(figsize=(12, 8))
# plt.plot(btc.index, btc['Log Returns'], label='Log Returns')
# plt.plot(test_data, label='Actual Log Returns')
# plt.plot(btc['forecasting return'], label='Forecasted Returns')
# plt.legend()
# plt.grid()
# plt.show()


# # model_sarima = SARIMAX(btc['Log Returns'], order=(3,3,3), seasonal_order=(2,1,2,12) )
# # model_fit_sarima = model.fit()
# # print(f'AIC: {model_fit_sarima.aic}')
# # print(f'BIC: {model_fit_sarima.bic}')
# # btc['forecast returns sarima'] = model_fit_sarima.predict(start=2850, end=3419, dynamic=True)
# # btc['forecast returns sarima'] += btc['Log Returns'].mean()
# # plt.figure(figsize=(12, 8))
# # plt.plot(btc.index, btc['Log Returns'], label='Log Returns')
# # plt.plot(btc['forecast returns sarima'], label='SARIMA Forecast')
# # plt.xlabel('Date')
# # plt.ylabel('Log Returns')
# # plt.title('Bitcoin Log Returns and ARIMA Forecast with 95% Confidence Interval')
# # plt.legend()
# # plt.grid()
# # plt.show()
# # btc[['Log Returns', 'forecast returns sarima']].plot(figsize=(12, 8))
# # plt.show()

# model = SARIMAX(train_data, order=(3,3,3), seasonal_order=(2,1,2,12))
# model_fit_test_sarimax = model.fit()
# btc['forecasting sarima'] = model_fit_test_sarimax.predict(start=3350, end=3419, dynamic=True)
# btc['forecasting sarima'] += btc['Log Returns'].mean()
# plt.figure(figsize=(12, 8))
# plt.plot(btc.index, btc['Log Returns'], label='Log Returns')
# plt.plot(test_data, label='Actual Log Returns')
# plt.plot(btc['forecasting'], label=('Forecast'))
# plt.title('SARIMA Forecast vs Actual Log Returns with 95% Confidence Interval')
# plt.legend()
# plt.show()

# print('MEAN ABSOLUTE ERROR')
# forecast = model_fit.get_forecast(steps=3419-2849)
# mae = mean_absolute_error(btc['Log Returns'].iloc[2848:3419], forecast.predicted_mean)
# forecast_sarima = model_fit_sarima.get_forecast(steps=3419-2849)
# mae_sarima = mean_absolute_error(btc['Log Returns'].iloc[2848:3419], forecast_sarima.predicted_mean)
# print(f'ARIMA MAE: {mae}')
# print(f'SARIMA MAE: {mae_sarima}')
# # param_grid =  [(1,1,1), (1,1,2), (1,2,1), (2,1,1), (2,2,1), (1,2,2), (2,1,2), (2,2,2), (1,1,3), (1,3,1), (3,1,1), (1,2,3), (2,1,3), (2,2,3), (1,3,1), (1,3,2), (2,3,1), (3,1,1), (3,1,2), (3,2,1), (3,2,2), (1,3,3),(3,3,1), (3,1,3), (2,3,3), (3,2,3), (3,3,2), (3,3,3)]


# def get_rmse(model):
#     forecast = model.get_forecast(steps=3419-2849)
#     rmse = mean_squared_error(btc['Log Returns'].iloc[2848:3419], forecast.predicted_mean, squared=False)
#     print(f'RMSE: {rmse}')


