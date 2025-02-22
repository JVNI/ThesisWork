import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

btc = yf.download('BTC-USD', start='2014-10-01', end='2024-02-10')
btc['Log Returns'] = np.log(btc['Adj Close']) - np.log(btc['Adj Close']).shift(1)
btc.drop(btc.index[0], axis=0, inplace=True)


train_data = btc['Log Returns'][:3389]
test_data = btc['Log Returns'][3389:]

param_grid =  [(0,0,0), (1,0,0), (1,0,1), (0,0,1), (0,0,1),(1,1,0), (0,1,1), (1,1,1), (1,1,2), (1,2,1), (2,1,1), (2,2,1), (1,2,2), (2,1,2), (2,2,2), (1,1,3), (1,3,1), (3,1,1), (1,2,3), (2,1,3), (2,2,3), (1,3,1), (1,3,2), (2,3,1), (3,1,2), (3,2,1), (3,2,2), (1,3,3),(3,3,1), (3,1,3), (2,3,3), (3,2,3), (3,3,2), (3,3,3)]
seasonal_grid = [(1,1,1,12), (1,1,2,12), (1,2,1,12), (2,1,1,12), (2,2,1,12), (1,2,2,12), (2,1,2,12), (2,2,2,12), (1,1,3,12), (1,3,1,12), (3,1,1,12), (1,2,3,12), (2,1,3,12), (2,2,3,12), (1,3,1,12), (1,3,2,12), (2,3,1,12), (3,1,2,12), (3,2,1,12), (3,2,2,12), (1,3,3,12),(3,3,1,12), (3,1,3,12), (2,3,3,12), (3,2,3,12), (3,3,2,12), (3,3,3,12)]
for params in param_grid:
    warnings.filterwarnings('ignore')
    model = ARIMA(train_data, order=params)
    predictions = model.fit()
    print(f'For Params {params} AIC: {predictions.aic} BIC : {predictions.bic}')

for seasonal_params in seasonal_grid:
    warnings.filterwarnings('ignore')
    model = SARIMAX(train_data, order=(1,0,2), seasonal_order=seasonal_params)
    predictions = model.fit()
    print(f'For Params: {seasonal_params} AIC: {predictions.aic} BIC: {predictions.bic}')
