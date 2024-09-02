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

def read_dataset(name='data/factor_capacidad_conjunto.csv',data='solar'):
    df = pd.read_csv(name,index_col=0,parse_dates=True)
    df = df.rename(columns={
        data: 'y'
    })
    return df[['y']]

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

def create_sarimax_df(df, df_ext=None, lags=[24*365], lags_ext=[1,24,365*24], freqs=[24, 24*365], levels=None, months=True, hours=True):
    if freqs is not None:
        df = create_fourier(df[['y']], freqs=freqs, levels=levels)

    if df_ext is not None:
        df = df.merge(df_ext, how='left', left_index=True, right_index=True)
        df = df.rename(columns={
            'y_x': 'y',
            'y_y': 'x',
        })

        if lags_ext is not None:
            for l in lags_ext:
                df[f"xlag_{l}"] = df['x'].shift(l)
        
        df = df.drop(['x'], axis=1)

    if lags is not None:
        for l in lags:
            df[f'lag_{l}'] = df['y'].shift(l)
    if months:
        df['month'] = df.index.month
    if hours:
        df['hours'] = df.index.hour
    return df.dropna()

def extend_df(df, new_y, freq='H'):
    last_timestamp = df.index[-1]
    new_timestamps = pd.date_range(start=last_timestamp, periods=len(new_y)+1, freq=freq)[1:]
    new_data = pd.DataFrame({'y': new_y})
    new_data.index=new_timestamps
    new_df = pd.concat([df[['y']], new_data])
    new_df = new_df.asfreq('H')
    return new_df

STEP_HORIZON = 24*365
FULL_HORIZON = 24*365
LAGS=None#[1,2,3,4,12,24,24*2]
LAGS_EXT=None#[1,2,3,4,12,24,24*2]
FREQS=[24,24*365]
LEVELS=[4,6]

start = time()
df_eolica = read_dataset(data='eolica')
df_solar = read_dataset(data='solar')
# df = df.iloc[-24*365*3:]
# df_exog = df_exog.iloc[-24*365*3:]
df_eolica['y'] = np.clip(df_eolica['y'], 0.0001, np.inf)
df_eolica['y'], lbc_eolica = boxcox(df_eolica['y'])
df_solar['y'] = np.clip(df_solar['y'], 0.0001, np.inf)
df_solar['y'], lbc_solar = boxcox(df_solar['y'])

Y_full_train_eolica = df_eolica.iloc[:-FULL_HORIZON][['y']]
Y_full_test_eolica = df_eolica.iloc[-FULL_HORIZON:][['y']]
Y_full_train_solar = df_solar.iloc[:-FULL_HORIZON][['y']]
Y_full_test_solar = df_solar.iloc[-FULL_HORIZON:][['y']]

MIN_P=5
MAX_P=20
loss_class_eolica = SERALossXGBoost(Y_full_train_eolica['y'],MIN_P,MAX_P)
LOSS_SOLAR = 'reg:squarederror'
LOSS_EOLICA = loss_class_eolica.sera_loss #'reg:squarederror' #

print(f'Preparation: {time()-start:.1f} s')
start = time()
X_train_solar = create_sarimax_df(Y_full_train_solar, Y_full_train_eolica, lags=LAGS, lags_ext=LAGS_EXT, freqs=FREQS, levels=LEVELS)
Y_train_solar = X_train_solar[['y']]
X_train_solar = X_train_solar.drop('y',axis=1)
model_solar = xgb.XGBRegressor(n_estimators=8, verbosity=2,subsample=.7,max_depth=4,objective=LOSS_SOLAR)
model_solar.fit(X_train_solar, Y_train_solar, verbose=True)

X_train_eolica = create_sarimax_df(Y_full_train_eolica, Y_full_train_solar, lags=LAGS, lags_ext=LAGS_EXT, freqs=FREQS, levels=LEVELS)
Y_train_eolica = X_train_eolica[['y']]
X_train_eolica = X_train_eolica.drop('y',axis=1)
model_eolica = xgb.XGBRegressor(n_estimators=30, verbosity=2,subsample=.7,max_depth=4,objective=LOSS_EOLICA)
model_eolica.fit(X_train_eolica, Y_train_eolica, verbose=True)
print(f'Fit: {time()-start:.1f} s')
start = time()
print("#### Starting XGBoost prediction")
def predict(Y_full_train_solar, Y_full_train_eolica, full_horizon, step_horizon, repr=1):
    n_laps = ceil(full_horizon/step_horizon)
    repr_laps = n_laps//repr
    for i in range(n_laps):
        if (i+1)%repr_laps==0:
            print(f'Lap {i+1}/{n_laps}')
        if -full_horizon + (i+1)*step_horizon == 0:
            horizon = len(Y_full_test_solar.iloc[-full_horizon + i*step_horizon:])
        else:
            horizon = len(Y_full_test_solar.iloc[-full_horizon + i*step_horizon:-full_horizon + (i+1)*step_horizon])
        if LAGS is not None:
            assert horizon <= min(LAGS)

        future_Y_solar = extend_df(Y_full_train_solar, pd.Series(np.ones(horizon)))
        X_train_solar = create_sarimax_df(future_Y_solar, df_ext=Y_full_train_eolica, lags=LAGS, lags_ext=LAGS_EXT, freqs=FREQS, levels=LEVELS)
        X_train_solar = X_train_solar.drop('y',axis=1)
        forecast = model_solar.predict(X_train_solar.iloc[-horizon:])
        Y_full_train_solar = extend_df(Y_full_train_solar, forecast)
        
        future_Y_eolica = extend_df(Y_full_train_eolica, pd.Series(np.ones(horizon)))
        X_train_eolica = create_sarimax_df(future_Y_eolica, df_ext=Y_full_train_solar, lags=LAGS, lags_ext=LAGS_EXT, freqs=FREQS, levels=LEVELS)
        X_train_eolica = X_train_eolica.drop('y',axis=1)
        forecast = model_eolica.predict(X_train_eolica.iloc[-horizon:])
        Y_full_train_eolica = extend_df(Y_full_train_eolica, forecast)
    return Y_full_train_solar, Y_full_train_eolica

Y_full_train_solar, Y_full_train_eolica = predict(Y_full_train_solar,Y_full_train_eolica, FULL_HORIZON, STEP_HORIZON)
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