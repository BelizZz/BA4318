import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm

#Importing data
# dataframe
df = pd.read_csv("Danish Kron.txt", sep='\t')
print(df.axes)

# Homework

def replacezero(array, nv):
    array[array == 0] = nv

df_copy = df.copy()
df_asarray = np.asarray(df_copy.VALUE)
replacezero(df_asarray, np.nan)
df_copy['VALUE'] = df_asarray
df_copy['NV'] = df_copy.VALUE.interpolate()

print(df_copy)

# Create slices for training and test data
size = len(df_copy)
head = df_copy[0:5]
tail = df_copy[size-5:]
print("Head")
print(head)
print("Tail")
print(tail)

train = df_copy[0:size-201]
test = df_copy[size-200:]


df.DATE = pd.to_datetime(df.DATE,format="%Y-%m-%d")
df_copy.index = df_copy.DATE 
train.DATE = pd.to_datetime(train.DATE,format="%Y-%m-%d") 
train.index = train.DATE 
test.DATE = pd.to_datetime(train.DATE,format="%Y-%m-%d") 
test.index = test.DATE 

#Naive approach
print("Naive")
dd = np.asarray(train.NV)
y_hat = test.copy()
y_hat['naive'] = dd[len(dd)-1]
rms = sqrt(mean_squared_error(test.NV, y_hat.naive))
print("RMSE: ",rms)

#Simple NV approach
print("Simple NV")
y_hat_avg = test.copy()
y_hat_avg['avg_forecast'] = train['NV'].mean()
rms = sqrt(mean_squared_error(test.NV, y_hat_avg.avg_forecast))
print("RMSE: ",rms)

#Moving NV approach
print("Moving NV")
windowsize = 15
y_hat_avg = test.copy()
y_hat_avg['moving_avg_forecast'] = train['NV'].rolling(windowsize).mean().iloc[-1]
rms = sqrt(mean_squared_error(test.NV, y_hat_avg.moving_avg_forecast))
print("RMSE: ",rms)

# Simple Exponential Smoothing
print("Simple Exponential Smoothing")
y_hat_avg = test.copy()
alpha = 0.2
fit2 = SimpleExpSmoothing(np.asarray(train['NV'])).fit(smoothing_level=alpha,optimized=False)
y_hat_avg['SES'] = fit2.forecast(len(test))
rms = sqrt(mean_squared_error(test.NV, y_hat_avg.SES))
print("RMSE: ",rms)

# Holt
print("Holt")
sm.tsa.seasonal_decompose(train.NV).plot()
result = sm.tsa.stattools.adfuller(train.NV)
# plt.show()

y_hat_avg = test.copy()
alpha = 0.4
fit1 = Holt(np.asarray(train['NV'])).fit(smoothing_level = alpha,smoothing_slope = 0.1)
y_hat_avg['Holt_linear'] = fit1.forecast(len(test))
rms = sqrt(mean_squared_error(test.NV, y_hat_avg.Holt_linear))
print("RMSE: ",rms)

# Holt-Winters
print("Holt-Winters")
y_hat_avg = test.copy()
seasons = 10
fit1 = ExponentialSmoothing(np.asarray(train['NV']) ,seasonal_periods=seasons ,trend='add', seasonal='add',).fit()
y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
rms = sqrt(mean_squared_error(test.NV, y_hat_avg.Holt_Winter))
print("RMSE: ",rms)