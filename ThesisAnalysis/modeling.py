import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from scipy.stats import norm
import seaborn as sns


btc = yf.download('BTC-USD', start='2014-10-01', end='2024-02-10')
print(btc)
btc['Returns'] = btc['Adj Close'] - btc['Adj Close'].shift(1)
btc['Log Returns'] = np.log(btc['Adj Close']) - np.log(btc['Adj Close']).shift(1)
btc.drop(btc.index[0], axis=0, inplace=True)

train_data = btc['Log Returns'][:3389]
test_data = btc['Log Returns'][3388:]
train_data.dropna(inplace=True)

# Outlier Detection and Trimming
print(len(train_data))
upper_limit = train_data.mean() + 3 * train_data.std()
lower_limit = train_data.mean() - 3 * train_data.std()

# Applying the condition correctly within a single .loc method
train_data = train_data[(train_data < upper_limit) & (train_data > lower_limit)]

def plot_ticker_price(ticker):
    plt.plot(ticker.index[:3389], ticker['Returns'][:3389], color='darkblue')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('BTC/USD Returns Chart')
    plt.show()

def plot_ticker_log(ticker):
    plt.plot(ticker.index[:3389], ticker['Log Returns'][:3389], color='darkblue')
    plt.xlabel('Date')
    plt.ylabel('Log Returns')
    plt.title('BTC/USD Log Returns')
    plt.show()

def decompose_data(ticker):
    decompose_data = seasonal_decompose(ticker['Returns'][:3389].dropna(), model = 'additive', period=365, extrapolate_trend='freq')
    daily_frequency = ticker.asfreq(freq='D')
    fig, axes = plt.subplots(4, 1, sharex=True)

    decompose_data.observed.plot(ax=axes[0], legend=False, color="darkblue")
    axes[0].set_ylabel('Observed')
    decompose_data.trend.plot(ax=axes[1], legend=False, color="darkblue")
    axes[1].set_ylabel('Trend')
    decompose_data.seasonal.plot(ax=axes[2], legend=False, color="darkblue")
    axes[2].set_ylabel('Seasonal')
    decompose_data.resid.plot(ax=axes[3], legend=False, color="darkblue")
    axes[3].set_ylabel('Residual')
    plt.show()

def stationarity_test(train):
    dftest = adfuller(train.dropna(), autolag = 'AIC')
    print("1. ADF : ", dftest[0])
    print("2. P-Value : ", dftest[1])
    print("3. Num Of Lags : ", dftest[2])
    print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
    print("5. Critical Values :")
    for key, val in dftest[4].items():
        print("\t",key, ": ", val)

def acf_plot(train):
    fig, ax = plt.subplots(figsize=(12,8))
    plot_acf(train_data, lags=20, alpha=0.05, ax=ax, color='darkblue')
    ax.set_title('Autocorrelation Function with 95% Confidence Interval')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    plt.show()

def pacf_plot(train):
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_pacf(train_data, lags=20, alpha=0.05, ax=ax, color='darkblue')
    ax.set_title('Partial Autocorrelation Function with 95% Confidence Interval')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Partial Autocorrelation')
    plt.show()
pacf_plot(train_data)

def model_arima(ticker, train, test, params):
    model = ARIMA(train, order=params)
    model_fit = model.fit()
    forecast_series = model_fit.forecast(30, alpha=0.05)
    forecast = model_fit.get_forecast(30)
    ci = forecast.conf_int(alpha=0.05)

    plt.figure(figsize=(12,6))
    train.plot(color='darkblue', label='Training Data (Log Returns)')
    test.plot(color='green', label='Actual Log Returns')
    plt.plot(test.index, forecast_series, label='forecast', color='red')
    plt.plot(test.index, ci.iloc[:, 0], label='Lower Confidence Interval', color='gray')
    plt.plot(test.index, ci.iloc[:, 1], label='Upper Confidence Interval', color='gray')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Log Returns')
    plt.title('BTC Log Return ARIMA Forecasted vs Actual Log Returns with 95% Confidence Interval')
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
    plt.title('BTC Log Return ARIMA Close-up Forecasted vs Actual Log Returns')
    plt.show()

def sarima_model(ticker, train, test, params, seasonal_params):
    model = SARIMAX(train, order=params, seasonal_order=seasonal_params)
    model_fit = model.fit(disp=False)
    
    # Forecasting the next 30 steps
    forecast = model_fit.get_forecast(steps=30)
    forecast_series = forecast.predicted_mean
    ci = forecast.conf_int(alpha=0.05)
    
    # Ensuring the forecast index matches the test data index
    forecast_index = test.index[:30]

    plt.figure(figsize=(12, 8))
    plt.plot(train.index, train, label='Training Data (Log Returns)', color="darkblue")
    plt.plot(test.index, test, label='Actual Log Returns', color="green")
    plt.plot(forecast_index, forecast_series, label='Forecasted Log Returns', color="red")
    plt.fill_between(forecast_index, ci.iloc[:, 0], ci.iloc[:, 1], color='gray', alpha=0.25)
    
    plt.title('BTC Log Return SARIMA Forecasted vs Actual Log Returns with 95% Confidence Interval')
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
        # Fit the ARIMA model
    arima_model = ARIMA(train, order=params)
    arima_model_fit = arima_model.fit()
    
    # Extract the residuals
    residuals = arima_model_fit.resid
    
    # Create custom diagnostic plots with dark blue color
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    # Residuals plot
    ax[0, 0].plot(residuals, color='darkblue')
    ax[0, 0].set_title('Residuals')
    
    # Q-Q plot
    qqplot(residuals, line='s', ax=ax[0, 1], color='darkblue')
    ax[0, 1].set_title('Q-Q Plot')

    # ACF plot
    plot_acf(residuals, ax=ax[1, 0], color='darkblue')
    ax[1, 0].set_title('Autocorrelation')
    
    # Residuals histogram with density line and normal distribution reference line
    sns.histplot(residuals, kde=True, ax=ax[1, 1], color='darkblue', edgecolor='black', stat='density', bins=30)

    # Calculate mean and standard deviation of residuals
    mean_resid = np.mean(residuals)
    std_resid = np.std(residuals)
    
    # Generate values for the normal distribution reference line
    x = np.linspace(min(residuals), max(residuals), 100)
    y = norm.pdf(x, mean_resid, std_resid)
    
    # Plot the normal distribution reference line
    ax[1, 1].plot(x, y, 'r', linestyle='--', label='Normal Distribution')
    ax[1, 1].legend()
    ax[1, 1].set_title('Histogram with Density and Normal Distribution')

    plt.tight_layout()
    plt.show()

    residuals = arima_model_fit.resid
    Btest = acorr_ljungbox(residuals, lags=10, return_df=True)

    print(Btest)

def sarima_diagnostics(ticker, train, test, params, seasonal_params):
    sarima_model = SARIMAX(train, order=params, seasonal_order=seasonal_params)
    sarima_model_fit = sarima_model.fit()
    residuals = sarima_model_fit.resid
    
    # Creat Custom diangnostics with dark blue color
    
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    # Residuals plot
    ax[0, 0].plot(residuals, color='darkblue')
    ax[0, 0].set_title('Residuals')
    
    # Q-Q plot
    qqplot(residuals, line='s', ax=ax[0, 1], color='darkblue')
    ax[0, 1].set_title('Q-Q Plot')

    # ACF plot
    plot_acf(residuals, ax=ax[1, 0], color='darkblue')
    ax[1, 0].set_title('Autocorrelation')
    
    # Residuals histogram with density line and normal distribution reference line
    sns.histplot(residuals, kde=True, ax=ax[1, 1], color='darkblue', edgecolor='black', stat='density', bins=30)

    # Calculate mean and standard deviation of residuals
    mean_resid = np.mean(residuals)
    std_resid = np.std(residuals)
    
    # Generate values for the normal distribution reference line
    x = np.linspace(min(residuals), max(residuals), 100)
    y = norm.pdf(x, mean_resid, std_resid)
    
    # Plot the normal distribution reference line
    ax[1, 1].plot(x, y, 'r', linestyle='--', label='Normal Distribution')
    ax[1, 1].legend()
    ax[1, 1].set_title('Histogram with Density and Normal Distribution')

    plt.tight_layout()
    plt.show()

    residuals = sarima_model_fit.resid
    Btest = acorr_ljungbox(residuals, lags=10, return_df=True)




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

def arima_eval(train, test, params):
    model = ARIMA(train, order=params)
    model_fit = model.fit()
    forecast = model_fit.get_forecast(30)
    rmse = mean_squared_error(test, forecast.predicted_mean, squared=False)
    mae = mean_absolute_error(test, forecast.predicted_mean)
    mape = mean_absolute_percentage_error(test, forecast.predicted_mean) 
    r2 = r2_score(test, forecast.predicted_mean)
    print(f'MAE: {mae}\nRMSE: {rmse}\nMAPE: {mape}\nR-Squared = {r2}')
   

def sarima_eval(train, test, params, seasonalparams):
    model = SARIMAX(train, order = params, seasonal_order = seasonalparams)
    model_fit = model.fit()
    forecast = model_fit.get_forecast(30)
    rmse = mean_squared_error(test, forecast.predicted_mean, squared=False)
    mae = mean_absolute_error(test, forecast.predicted_mean)
    mape = mean_absolute_percentage_error(test, forecast.predicted_mean) * 100
    r2 = r2_score(test, forecast.predicted_mean)
    print(f'MAE: {mae}\nRMSE: {rmse}\nMAPE: {mape}\nR-Squared = {r2}')

def ar_model(train, test, lags):
    model = AutoReg(train, lags = lags)
    model_fit = model.fit()
    forecast = model_fit.predict(start=len(train), end=len(train)+len(test)-1)
    rmse = mean_squared_error(test, forecast, squared=False)
    mae = mean_absolute_error(test, forecast)
    mape = mean_absolute_percentage_error(test, forecast) * 100
    r2 = r2_score(test, forecast)
    print(f'AR MODEL:\nMAE: {mae}\nRMSE: {rmse}\nMAPE: {mape}\nR-Squared = {r2}')

def sma_model(train, test, window):
    train_series = pd.Series(train)
    forecast = train_series.rolling(window=window).mean().iloc[-len(test):]
    rmse = mean_squared_error(test, forecast, squared=False)
    mae = mean_absolute_error(test, forecast)
    mape = mean_absolute_percentage_error(test, forecast) * 100
    r2 = r2_score(test, forecast)
    print(f'SMA:\nMAE: {mae}\nRMSE: {rmse}\nMAPE: {mape}\nR-Squared = {r2}')

def ema_model(train, test, span):
    train_series = pd.Series(train)
    forecast = train_series.ewm(span=span, adjust=False).mean().iloc[-len(test):]
    rmse = mean_squared_error(test, forecast, squared=False)
    mae = mean_absolute_error(test, forecast)
    mape = mean_absolute_percentage_error(test, forecast) * 100
    r2 = r2_score(test, forecast)
    print(f'EMA:\nMAE: {mae}\nRMSE: {rmse}\nMAPE: {mape}\nR-Squared = {r2}')


