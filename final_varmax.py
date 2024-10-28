import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from test_module import save_results, Test
from statsmodels.tsa.statespace.varmax import VARMAX
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import warnings
from time import time

warnings.filterwarnings('ignore')

def read_full_dataset(name='data/factor_capacidad.csv'):
    df = pd.read_csv(name,index_col=0,parse_dates=True)
    return df

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

def create_varmax_df(df, lags=[24*365], freqs=[24, 24*365]):
    y_ = df.columns
    df = create_fourier(df, freqs=freqs)
    for l in lags:
        for y in y_:
            df[f'lag_{y}_{l}'] = df[y].shift(l)
    return df.dropna()

def extend_df(df, new_y, freq='H'):
    y_names = df.columns
    last_timestamp = df.index[-1]
    new_timestamps = pd.date_range(start=last_timestamp, periods=len(new_y)+1, freq=freq)[1:]
    if type(new_y) == pd.Series:
        new_data = pd.DataFrame(data={y_name: new_y.values for y_name in y_names}, index=new_timestamps)
    elif type(new_y) == pd.DataFrame:
        new_data = pd.DataFrame(data={y_name: new_y[y_name].values for y_name in y_names},index=new_timestamps)
    else:
        raise ValueError('new_y must be of type pd.Series or pd.DataFrame')
    new_df = pd.concat([df[y_names], new_data])
    new_df = new_df.asfreq('H')
    return new_df

def invert_box_cox(df,lbc,y_col):
    df[y_col] = inv_boxcox(df[y_col], lbc)
    return df

STEP_HORIZON = 24
FULL_HORIZON = 24*365

LAGS=[24]
FREQS=[24,24*365]
ORDER=(2,2)

start = time()
df = read_full_dataset()
df = df.iloc[-24*365-24*100:]
y_cols = df.columns
df['wind'] = np.clip(df['wind'], 0.0001, np.inf)
df['wind'], lbc_wind = boxcox(df['wind'])
df['solar_th'] = np.clip(df['solar_th'], 0.0001, np.inf)
df['solar_th'], lbc_solar_th = boxcox(df['solar_th'])
df['solar_pv'] = np.clip(df['solar_pv'], 0.0001, np.inf)
df['solar_pv'], lbc_solar_pv = boxcox(df['solar_pv'])

Y_full_train = df.iloc[:-FULL_HORIZON]
Y_full_test = df.iloc[-FULL_HORIZON:]

X_train = create_varmax_df(Y_full_train, lags=LAGS, freqs=FREQS)
Y_train = X_train[y_cols]
X_train = X_train.drop(y_cols,axis=1)
model = VARMAX(Y_train, exog=X_train, order=ORDER)
results = model.fit()

print(f'Fit: {time()-start:.1f} s')
start = time()
print("#### Starting VARMAX prediction")
n_laps = ceil(FULL_HORIZON/STEP_HORIZON)
repren=10
repr_laps = n_laps//repren
for i in range(n_laps):
    if (i+1)%repr_laps==0:
        print(f'Lap {i+1}/{n_laps}')
    if -FULL_HORIZON + (i+1)*STEP_HORIZON == 0:
        horizon = len(Y_full_test.iloc[-FULL_HORIZON + i*STEP_HORIZON:])
    else:
        horizon = len(Y_full_test.iloc[-FULL_HORIZON + i*STEP_HORIZON:-FULL_HORIZON + (i+1)*STEP_HORIZON])
    if LAGS is not None:
        assert horizon <= min(LAGS)


    future_Y = extend_df(Y_full_train, pd.Series(np.ones(horizon)))
    X_train = create_varmax_df(future_Y, lags=LAGS, freqs=FREQS)
    X_train = X_train.drop(y_cols, axis=1)
    forecast = results.get_forecast(steps=horizon, exog=X_train.iloc[-horizon:])
    forecast = forecast.predicted_mean
    Y_full_train = extend_df(Y_full_train, forecast)
    # print(results.endog.index[-5:])  # Model’s last index
    # print(Y_train.iloc[-horizon:].index)
    # # results.extend(Y_full_train.iloc[-horizon:])
    results = results.append(Y_full_train.iloc[-horizon:].values, exog=X_train.iloc[-horizon:].values, refit=False)

print(f'Prediction: {time()-start:.1f} s')

Y_full_train = invert_box_cox(Y_full_train,lbc_wind,'wind')
Y_full_test = invert_box_cox(Y_full_test,lbc_wind,'wind')
Y_full_train = invert_box_cox(Y_full_train,lbc_solar_pv,'solar_pv')
Y_full_test = invert_box_cox(Y_full_test,lbc_solar_pv,'solar_pv')
Y_full_train = invert_box_cox(Y_full_train,lbc_solar_th,'solar_th')
Y_full_test = invert_box_cox(Y_full_test,lbc_solar_th,'solar_th')

df_obs = Y_full_test
df_pred = Y_full_train
df_pred = df_pred.iloc[-FULL_HORIZON:,:]

model = 'varmax'
params = {
    'lags': LAGS,
    'freqs': FREQS,
    'order': ORDER
}
experiment = {
    'step_horizon': STEP_HORIZON,
    'full_horizon': FULL_HORIZON,
    'obs': df_obs
}
prediction = df_pred
model_objects = {
    'joint': None,
}

save_results(model,params,experiment,prediction,model_objects)

test = Test(df_obs,df_pred)
test.save_test_results('varmax')
df_test_results = test.get_results_df()
print(df_test_results)