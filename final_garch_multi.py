import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import warnings
from datetime import datetime
import pickle as pkl
from time import time

from test_module import save_results, Test

warnings.filterwarnings('ignore')

def read_dataset(name='data/factor_capacidad.csv',data='wind'):
    df = pd.read_csv(name,index_col=0,parse_dates=True)
    return df[[data]]

def create_fourier(df, freqs=[24,24*365], levels=None):
    if levels is not None:
        assert len(levels) == len(freqs)
    else:
        levels = [1 for _ in freqs]

    T = len(df)
    t = np.arange(0,T)
    result_df = df.copy()
    for i, f in enumerate(freqs):
        for l in range(1, levels[i]+1):
            sin = np.sin(2*np.pi*t*l/f)
            result_df[f'sen_f{f}_l{l}'] = sin
            cos = np.cos(2*np.pi*t*l/f)
            result_df[f'cos_f{f}_l{l}'] = cos
    return result_df

def create_sarimax_df(df, df_ext=None, lags=[24*365], lags_ext=[1,24,365*24], freqs=[24, 24*365], levels=None, months=False, hours=False):
    y_name = df.columns[0]
    if freqs is not None:
        df = create_fourier(df[[y_name]], freqs=freqs, levels=levels)
    if df_ext is not None:
        x_names = df_ext.columns
        df = df.merge(df_ext, how='left', left_index=True, right_index=True)

        if lags_ext is not None:
            for x in x_names:
                for l in lags_ext:
                    df[f"{x}_lag_{l}"] = df[x].shift(l)
                df = df.drop([x], axis=1)

    if lags is not None:
        for l in lags:
            df[f'lag_{l}'] = df[y_name].shift(l)
    if months:
        df['month'] = df.index.month
    if hours:
        df['hours'] = df.index.hour
    return df.dropna()

def extend_df(df, new_y, freq='H'):
    y_name = df.columns[0]
    last_timestamp = df.index[-1]
    new_timestamps = pd.date_range(start=last_timestamp, periods=len(new_y)+1, freq=freq)[1:]
    new_data = pd.DataFrame({y_name: new_y})
    new_data.index=new_timestamps
    new_df = pd.concat([df[[y_name]], new_data])
    new_df = new_df.asfreq('H')
    return new_df

def invert_box_cox(df,lbc,y_col):
    df[y_col] = inv_boxcox(df[y_col], lbc)
    return df


STEP_HORIZON = 1
FULL_HORIZON = 24*365

# LAGS=[24]
# LAGS_EXT=[1,2,3,4,24]
# FREQS=[24,24*365]
# LEVELS=[2,4]
# ORDER=(4,1,4)

LAGS=[24]
LAGS_EXT=[1,2,24]
FREQS=[24,24*365]
LEVELS=[2,2]
ORDER=(2,1,2)

start = time()
df_wind = read_dataset(data='wind')
df_solar_th = read_dataset(data='solar_th')
df_solar_pv = read_dataset(data='solar_pv')
# df_wind = df_wind.iloc[-24*365*2:]
# df_solar_th = df_solar_th.iloc[-24*365*2:]
# df_solar_pv = df_solar_pv.iloc[-24*365*2:]

df_wind['wind'] = np.clip(df_wind['wind'], 0.0001, np.inf)
df_wind['wind'], lbc_wind = boxcox(df_wind['wind'])
df_solar_th['solar_th'] = np.clip(df_solar_th['solar_th'], 0.0001, np.inf)
df_solar_th['solar_th'], lbc_solar_th = boxcox(df_solar_th['solar_th'])
df_solar_pv['solar_pv'] = np.clip(df_solar_pv['solar_pv'], 0.0001, np.inf)
df_solar_pv['solar_pv'], lbc_solar_pv = boxcox(df_solar_pv['solar_pv'])

Y_full_train_wind = df_wind.iloc[:-FULL_HORIZON][['wind']]
Y_full_test_wind = df_wind.iloc[-FULL_HORIZON:][['wind']]
Y_full_train_solar_th = df_solar_th.iloc[:-FULL_HORIZON][['solar_th']]
Y_full_test_solar_th = df_solar_th.iloc[-FULL_HORIZON:][['solar_th']]
Y_full_train_solar_pv = df_solar_pv.iloc[:-FULL_HORIZON][['solar_pv']]
Y_full_test_solar_pv = df_solar_pv.iloc[-FULL_HORIZON:][['solar_pv']]

print(f'Preparation: {time()-start:.1f} s')
start = time()

df_ext = pd.merge(Y_full_train_solar_th,Y_full_train_wind,how='outer',left_index=True,right_index=True)
X_train_solar_pv = create_sarimax_df(Y_full_train_solar_pv, df_ext, lags=LAGS, lags_ext=LAGS_EXT, freqs=FREQS, levels=LEVELS)
Y_train_solar_pv = X_train_solar_pv[['solar_pv']]
X_train_solar_pv = X_train_solar_pv.drop('solar_pv',axis=1)
model_solar_pv = ARIMA(Y_train_solar_pv['solar_pv'],exog=X_train_solar_pv,order=ORDER)
fitted_model_solar_pv = model_solar_pv.fit()

df_ext = pd.merge(Y_full_train_solar_pv,Y_full_train_wind,how='outer',left_index=True,right_index=True)
X_train_solar_th = create_sarimax_df(Y_full_train_solar_th, df_ext, lags=LAGS, lags_ext=LAGS_EXT, freqs=FREQS, levels=LEVELS)
Y_train_solar_th = X_train_solar_th[['solar_th']]
X_train_solar_th = X_train_solar_th.drop('solar_th',axis=1)
model_solar_th = ARIMA(Y_train_solar_th['solar_th'],exog=X_train_solar_th,order=ORDER)
fitted_model_solar_th = model_solar_th.fit()

df_ext = pd.merge(Y_full_train_solar_pv,Y_full_train_solar_th,how='outer',left_index=True,right_index=True)
X_train_wind = create_sarimax_df(Y_full_train_wind, df_ext, lags=LAGS, lags_ext=LAGS_EXT, freqs=FREQS, levels=LEVELS)
Y_train_wind = X_train_wind[['wind']]
X_train_wind = X_train_wind.drop('wind',axis=1)
model_wind = arch_model(
    y=Y_train_wind['wind'],
    x=X_train_wind,
    mean='ARX',
    lags=[1,2],
    vol='EGARCH',
    p=2,
    o=1,
    q=2,
    power=2,
    dist='ged',
    )
fitted_model_wind = model_wind.fit(options={'maxiter':500})

# am = ARX(Y_full_train_wind['wind'])
# am.volatility = GARCH(2,0,2)
# am.distribution = Normal()
# res = am.fit()
# am = arch_model(
#     y=Y_train_solar_pv['solar_pv'],
#     x=X_train_solar_pv,
#     mean='ARX',
#     lags=None,
#     vol='EGARCH',
#     p=2,
#     o=1,
#     q=2,
#     power=2,
#     dist='ged',
#     )
# res = am.fit()
# print(res.summary())
# horizon = 24*7
# x = X_train_solar_pv.iloc[horizon:,:].values.transpose()
# x = x[:,np.newaxis,:]
# forecasts = res.forecast(horizon=horizon, x=x,method='simulation',simulations=1)
# fig, ax = plt.subplots()
# print(forecasts.simulations.values.shape)
# ax.plot(forecasts.simulations.values[0,0,:])
# plt.savefig('aux.png')

print(f'Fit: {time()-start:.1f} s')
start = time()
print("#### Starting ARCH prediction")
n_laps = ceil(FULL_HORIZON/STEP_HORIZON)
repren=10
repr_laps = n_laps//repren
for i in range(n_laps):
    if (i+1)%repr_laps==0:
        print(f'Lap {i+1}/{n_laps}')
    if -FULL_HORIZON + (i+1)*STEP_HORIZON == 0:
        horizon = len(Y_full_test_solar_pv.iloc[-FULL_HORIZON + i*STEP_HORIZON:])
    else:
        horizon = len(Y_full_test_solar_pv.iloc[-FULL_HORIZON + i*STEP_HORIZON:-FULL_HORIZON + (i+1)*STEP_HORIZON])
    if LAGS is not None:
        assert horizon <= min(LAGS)


    df_ext = pd.merge(Y_full_train_solar_th,Y_full_train_wind,how='outer',left_index=True,right_index=True)
    future_Y_solar_pv = extend_df(Y_full_train_solar_pv, pd.Series(np.ones(horizon)))
    X_train_solar_pv = create_sarimax_df(future_Y_solar_pv, df_ext=df_ext, lags=LAGS, lags_ext=LAGS_EXT, freqs=FREQS, levels=LEVELS)
    X_train_solar_pv = X_train_solar_pv.drop('solar_pv',axis=1)
    forecast = fitted_model_solar_pv.get_forecast(steps=horizon,exog=X_train_solar_pv.iloc[-horizon:])
    forecast = forecast.predicted_mean
    Y_full_train_solar_pv = extend_df(Y_full_train_solar_pv, forecast)
    # df_ext = pd.merge(Y_full_train_solar_th,Y_full_train_wind,how='outer',left_index=True,right_index=True)
    X_train_solar_pv = create_sarimax_df(Y_full_train_solar_pv, df_ext=df_ext, lags=LAGS, lags_ext=LAGS_EXT, freqs=FREQS, levels=LEVELS)
    X_train_solar_pv = X_train_solar_pv.drop('solar_pv',axis=1)
    fitted_model_solar_pv = fitted_model_solar_pv.append(Y_full_train_solar_pv.iloc[-horizon:].values, exog=X_train_solar_pv.iloc[-horizon:].values, refit=False)
    
    
    df_ext = pd.merge(Y_full_train_solar_pv,Y_full_train_wind,how='outer',left_index=True,right_index=True)
    future_Y_solar_th = extend_df(Y_full_train_solar_th, pd.Series(np.ones(horizon)))
    X_train_solar_th = create_sarimax_df(future_Y_solar_th, df_ext=df_ext, lags=LAGS, lags_ext=LAGS_EXT, freqs=FREQS, levels=LEVELS)
    X_train_solar_th = X_train_solar_th.drop('solar_th',axis=1)
    forecast = fitted_model_solar_th.get_forecast(steps=horizon,exog=X_train_solar_th.iloc[-horizon:])
    forecast = forecast.predicted_mean
    Y_full_train_solar_th = extend_df(Y_full_train_solar_th, forecast)
    X_train_solar_th = create_sarimax_df(Y_full_train_solar_th, df_ext=df_ext, lags=LAGS, lags_ext=LAGS_EXT, freqs=FREQS, levels=LEVELS)
    X_train_solar_th = X_train_solar_th.drop('solar_th',axis=1)
    fitted_model_solar_th = fitted_model_solar_th.append(Y_full_train_solar_th.iloc[-horizon:].values, exog=X_train_solar_th.iloc[-horizon:].values, refit=False)
    

    df_ext = pd.merge(Y_full_train_solar_pv,Y_full_train_solar_th,how='outer',left_index=True,right_index=True)
    future_Y_wind = extend_df(Y_full_train_wind, pd.Series(np.ones(horizon)))
    X_train_wind = create_sarimax_df(future_Y_wind, df_ext=df_ext, lags=LAGS, lags_ext=LAGS_EXT, freqs=FREQS, levels=LEVELS)
    X_train_wind = X_train_wind.drop('wind',axis=1)
    x = X_train_wind.iloc[-horizon:,:].values.transpose()
    x = x[:,np.newaxis,:]
    forecast = fitted_model_wind.forecast(horizon=horizon, x=x,method='simulation',simulations=1)
    forecast = forecast.simulations.values[0,0,:]
    Y_full_train_wind = extend_df(Y_full_train_wind, forecast)
    

print(f'Prediction: {time()-start:.1f} s')

Y_full_train_wind = invert_box_cox(Y_full_train_wind,lbc_wind,'wind')
Y_full_test_wind = invert_box_cox(Y_full_test_wind,lbc_wind,'wind')
Y_full_train_solar_pv = invert_box_cox(Y_full_train_solar_pv,lbc_solar_pv,'solar_pv')
Y_full_test_solar_pv = invert_box_cox(Y_full_test_solar_pv,lbc_solar_pv,'solar_pv')
Y_full_train_solar_th = invert_box_cox(Y_full_train_solar_th,lbc_solar_th,'solar_th')
Y_full_test_solar_th = invert_box_cox(Y_full_test_solar_th,lbc_solar_th,'solar_th')

df_obs = Y_full_test_solar_pv[['solar_pv']].merge(Y_full_test_solar_th[['solar_th']],left_index=True,right_index=True).merge(Y_full_test_wind[['wind']],left_index=True,right_index=True)
df_pred = Y_full_train_solar_pv[['solar_pv']].merge(Y_full_train_solar_th[['solar_th']],left_index=True,right_index=True).merge(Y_full_train_wind[['wind']],left_index=True,right_index=True)
df_pred = df_pred.iloc[-FULL_HORIZON:,:]

model = 'arch'
params = {
    'lags': LAGS,
    'lags_ext': LAGS_EXT,
    'freqs': FREQS,
    'levels': LEVELS,
}
experiment = {
    'step_horizon': STEP_HORIZON,
    'full_horizon': FULL_HORIZON,
    'obs': df_obs
}
prediction = df_pred
model_objects = {
    'solar_pv': None,
    'solar_th': None,
    'wind': None,
}

save_results(model,params,experiment,prediction,model_objects)

test = Test(df_obs,df_pred)
test.save_test_results('arch')
df_test_results = test.get_results_df()
print(df_test_results)