import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from test_module import test_results
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import warnings
import xgboost as xgb
from datetime import datetime
from time import time
from sera_loss import SERALossXGBoost


warnings.filterwarnings('ignore')

def read_dataset(name='data/factor_capacidad.csv',data='solar'):
    df = pd.read_csv(name,index_col=0,parse_dates=True)
    # df = df.rename(columns={
    #     data: 'y'
    # })
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

# def create_arima_df(df, lags=[24*365], freqs=[24, 24*365], levels=None):
#     df = create_fourier(df[['y']], freqs=freqs, levels=levels)
#     for l in lags:
#         df[f'lag_{l}'] = df['y'].shift(l)
#     return df.dropna()

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

STEP_HORIZON = 1
FULL_HORIZON = 24*365
LAGS=[1,2,3,4,12,24,24*2]
LAGS_EXT=[1,2,3,4,12,24,24*2]
FREQS=[24,24*365]
LEVELS=[2,3]

start = time()
df_wind = read_dataset(data='wind')
df_solar_th = read_dataset(data='solar_th')
df_solar_pv = read_dataset(data='solar_pv')

df_wind['wind'] = np.clip(df_wind['wind'], 0.0001, np.inf)
df_wind['wind'], lbc_eolica = boxcox(df_wind['wind'])
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

MIN_P=5
MAX_P=20
# loss_class_eolica = SERALossXGBoost(Y_full_train_eolica['y'],MIN_P,MAX_P)
LOSS_SOLAR_PV = 'reg:squarederror'
LOSS_SOLAR_TH = 'reg:squarederror'
LOSS_WIND = 'reg:squarederror' #loss_class_eolica.sera_loss

print(f'Preparation: {time()-start:.1f} s')
start = time()
df_ext = pd.merge(Y_full_train_solar_th,Y_full_train_wind,how='outer',left_index=True,right_index=True)
X_train_solar_pv = create_sarimax_df(Y_full_train_solar_pv, df_ext, lags=LAGS, lags_ext=LAGS_EXT, freqs=FREQS, levels=LEVELS)
Y_train_solar_pv = X_train_solar_pv[['solar_pv']]
X_train_solar_pv = X_train_solar_pv.drop('solar_pv',axis=1)
model_solar_pv = xgb.XGBRegressor(n_estimators=8, verbosity=2,subsample=.7,max_depth=4,objective=LOSS_SOLAR_PV)
model_solar_pv.fit(X_train_solar_pv, Y_train_solar_pv, verbose=True)

df_ext = pd.merge(Y_full_train_solar_pv,Y_full_train_wind,how='outer',left_index=True,right_index=True)
X_train_solar_th = create_sarimax_df(Y_full_train_solar_th, df_ext, lags=LAGS, lags_ext=LAGS_EXT, freqs=FREQS, levels=LEVELS)
Y_train_solar_th = X_train_solar_th[['solar_th']]
X_train_solar_th = X_train_solar_th.drop('solar_th',axis=1)
model_solar_th = xgb.XGBRegressor(n_estimators=8, verbosity=2,subsample=.7,max_depth=4,objective=LOSS_SOLAR_TH)
model_solar_th.fit(X_train_solar_th, Y_train_solar_th, verbose=True)

df_ext = pd.merge(Y_full_train_solar_pv,Y_full_train_solar_th,how='outer',left_index=True,right_index=True)
X_train_wind = create_sarimax_df(Y_full_train_wind, df_ext, lags=LAGS, lags_ext=LAGS_EXT, freqs=FREQS, levels=LEVELS)
Y_train_wind = X_train_wind[['wind']]
X_train_wind = X_train_wind.drop('wind',axis=1)
model_wind = xgb.XGBRegressor(n_estimators=8, verbosity=2,subsample=.7,max_depth=4,objective=LOSS_WIND)
model_wind.fit(X_train_wind, Y_train_wind, verbose=True)

print(f'Fit: {time()-start:.1f} s')
start = time()
print("#### Starting XGBoost prediction")
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
    forecast = model_solar_pv.predict(X_train_solar_pv.iloc[-horizon:])
    Y_full_train_solar_pv = extend_df(Y_full_train_solar_pv, forecast)
    
    df_ext = pd.merge(Y_full_train_solar_pv,Y_full_train_wind,how='outer',left_index=True,right_index=True)
    future_Y_solar_th = extend_df(Y_full_train_solar_th, pd.Series(np.ones(horizon)))
    X_train_solar_th = create_sarimax_df(future_Y_solar_th, df_ext=df_ext, lags=LAGS, lags_ext=LAGS_EXT, freqs=FREQS, levels=LEVELS)
    X_train_solar_th = X_train_solar_th.drop('solar_th',axis=1)
    forecast = model_solar_th.predict(X_train_solar_th.iloc[-horizon:])
    Y_full_train_solar_th = extend_df(Y_full_train_solar_th, forecast)

    df_ext = pd.merge(Y_full_train_solar_pv,Y_full_train_solar_th,how='outer',left_index=True,right_index=True)
    future_Y_wind = extend_df(Y_full_train_wind, pd.Series(np.ones(horizon)))
    X_train_wind = create_sarimax_df(future_Y_wind, df_ext=df_ext, lags=LAGS, lags_ext=LAGS_EXT, freqs=FREQS, levels=LEVELS)
    X_train_wind = X_train_wind.drop('wind',axis=1)
    forecast = model_wind.predict(X_train_wind.iloc[-horizon:])
    Y_full_train_wind = extend_df(Y_full_train_wind, forecast)

# Y_full_train_solar, Y_full_train_eolica = predict(Y_full_train_solar,Y_full_train_eolica, FULL_HORIZON, STEP_HORIZON)
print(f'Prediction: {time()-start:.1f} s')
# Test metrics
print('\n###### Solar Test:')
Y_full_train_solar['y'] = inv_boxcox(Y_full_train_solar['y'], lbc_solar)
Y_full_test_solar['y'] = inv_boxcox(Y_full_test_solar['y'], lbc_solar)
test_results(Y_full_test_solar['y'].reset_index(drop=True), Y_full_train_solar.iloc[-FULL_HORIZON:]['y'].reset_index(drop=True),min_p=MIN_P,max_p=MAX_P)

print('###### Eolica Test:')
Y_full_train_eolica['y'] = inv_boxcox(Y_full_train_eolica['y'], lbc_eolica)
Y_full_test_eolica['y'] = inv_boxcox(Y_full_test_eolica['y'], lbc_eolica)
test_results(Y_full_test_eolica['y'].reset_index(drop=True), Y_full_train_eolica.iloc[-FULL_HORIZON:]['y'].reset_index(drop=True),min_p=MIN_P,max_p=MAX_P)

# Plot predictions
fig, ax = plt.subplots(1, 1, figsize = (20, 7))
Y_hat_df = Y_full_test_solar.merge(Y_full_train_solar, how='left', left_index=True, right_index=True)
plot_df = Y_hat_df
plot_df[['y_x', 'y_y']].plot(ax=ax, linewidth=2)
ax.set_title('Solar', fontsize=22)
ax.set_ylabel('Factor capacidad', fontsize=20)
ax.set_xlabel('Timestamp [t]', fontsize=20)
ax.legend(['Real','Prediction'])
ax.grid()
plt.savefig('xgboost_solar.png', dpi=300, bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize = (20, 7))
Y_hat_df = Y_full_test_eolica.merge(Y_full_train_eolica, how='left', left_index=True, right_index=True)
plot_df = Y_hat_df
plot_df[['y_x', 'y_y']].plot(ax=ax, linewidth=2)
ax.set_title('Eolica', fontsize=22)
ax.set_ylabel('Factor capacidad', fontsize=20)
ax.set_xlabel('Timestamp [t]', fontsize=20)
ax.legend(['Real','Prediction'])
ax.grid()
plt.savefig('xgboost_eolica.png', dpi=300, bbox_inches='tight')

# Save results
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_solar = f'xgboost_solar_{timestamp}.csv'
save_eolica = f'xgboost_eolica_{timestamp}.csv'
Y_full_train_solar.to_csv(f'./results/{save_solar}')
Y_full_train_eolica.to_csv(f'./results/{save_eolica}')

# Train metrics
def get_train_metrics(y_solar, y_eolica, full_horizon, step_horizon, folds=3):
    assert len(y_solar)==len(y_eolica)
    assert len(y_solar) > folds*full_horizon

    fmean = lambda x: sum(x)/len(x)
    y_train_solar = y_solar.copy()
    y_train_eolica = y_eolica.copy()
    solar_tests = []
    eolica_tests = []
    for _ in range(1,folds+1):
        y_test_solar = y_train_solar.iloc[-full_horizon:]
        y_test_eolica = y_train_eolica.iloc[-full_horizon:]
        y_train_solar = y_train_solar.iloc[:-full_horizon]
        y_train_eolica = y_train_eolica.iloc[:-full_horizon]

        y_pred_solar,y_pred_eolica = predict(y_train_solar,y_train_eolica, full_horizon, step_horizon)
        y_pred_solar['y'] = inv_boxcox(y_pred_solar['y'], lbc_solar)
        y_pred_eolica['y'] = inv_boxcox(y_pred_eolica['y'], lbc_eolica)
        solar_tests.append(test_results(y_test_solar['y'].reset_index(drop=True), y_pred_solar.iloc[-full_horizon:]['y'].reset_index(drop=True), disp_error=False,min_p=MIN_P,max_p=MAX_P))
        eolica_tests.append(test_results(y_test_eolica['y'].reset_index(drop=True), y_pred_eolica.iloc[-full_horizon:]['y'].reset_index(drop=True), disp_error=False,min_p=MIN_P,max_p=MAX_P))
        tests = (solar_tests, eolica_tests)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_solar = f'xgboost_solar_train_{timestamp}.csv'
        save_eolica = f'xgboost_eolica_train_{timestamp}.csv'
        y_pred_solar.to_csv(f'./results/{save_solar}')
        y_pred_eolica.to_csv(f'./results/{save_eolica}')

    for v in range(2):
        if v==0:
            print('\n###### Solar Train:')
        elif v==1:
            print('###### Eolica Train:')
        rmse = fmean([tests[v][i][0] for i in range(folds)])
        mae = fmean([tests[v][i][1] for i in range(folds)])
        corr = fmean([tests[v][i][2] for i in range(folds)])
        sera = fmean([tests[v][i][3] for i in range(folds)])
        bias = fmean([tests[v][i][4] for i in range(folds)])
        rmse95 = fmean([tests[v][i][5] for i in range(folds)])
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"SERA [{MIN_P}%, {MAX_P}%]:  {sera:.4f}")
        print(f"Bias: {bias:.4f}")
        print(f"Correlation: {corr:.4f}")
        print(f"RMSE at lower 5%: {rmse95:.4f}")
get_train_metrics(Y_full_train_solar.iloc[:-FULL_HORIZON],Y_full_train_eolica.iloc[:-FULL_HORIZON],FULL_HORIZON,STEP_HORIZON)
pass