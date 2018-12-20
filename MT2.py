import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt


def moving_average(train, test, value, windowsize):
    # print("Moving Average")
    y_hat_avg = test.copy()
    y_hat_avg['Moving_Average'] = train[value].rolling(windowsize).mean().iloc[-1]
    rms = sqrt(mean_squared_error(test[value], y_hat_avg.Moving_Average))
    return rms

def simple_exponential_smoothing(train, test, value, alpha):
    # print("Simple Exponential Smoothing")
    y_hat_avg = test.copy()
    fit2 = SimpleExpSmoothing(np.asarray(train[value])).fit(smoothing_level=alpha,optimized=False)
    y_hat_avg['SES'] = fit2.forecast(len(test))
    rms =sqrt(mean_squared_error(test[value], y_hat_avg.SES))
    return rms

def holt_winters(train, test, value, seasons):
    # print("Holt_Winter")
    y_hat_avg = test.copy()
    array =np.asarray(train[value])
    fit = ExponentialSmoothing(array, seasonal_periods=seasons, trend='add', seasonal='add',).fit()
    y_hat_avg['Holt_Winter'] = fit.forecast(len(test))
    rms =sqrt(mean_squared_error(test[value], y_hat_avg.Holt_Winter))
    return rms


df = pd.read_csv("mt2data.txt", sep=';')
# print(df.axes)
size = len(df)
train = df[0:size-5]
test = df[size-6:]


error_ma = moving_average(train, test, value='WTI', windowsize=12)
error_ses = simple_exponential_smoothing(train, test, value='WTI', alpha=0.82)
error_hw = holt_winters(train, test, value='WTI', seasons=5)

# print("Moving Average RMSE:", error_ma)
print("Simple Exponential Smoothing", "\n", "RMSE:", error_ses)
# print("Holt Winters RMSE:", error_hw)

def forecastprice():
    oilprice = []
    y_hat_avg = test.copy()
    alpha = 0.82
    array = np.asarray(df['WTI'])
    fit2 = SimpleExpSmoothing(np.asarray(train[value])).fit(smoothing_level=alpha,optimized=False)
    futureprice = fit.forecast(len(test))
    oilprice = oilprice.append(futureprice)
    return oilprice

print("Expected price for 21.12.2018 :", oilprice.forecastprice[3])
    

