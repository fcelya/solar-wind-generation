import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from test_module import test_results
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import warnings
import xgboost as xgb
from sklearn.svm import SVR
from datetime import datetime

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

def create_arima_df(df, lags=[24*365], freqs=[24, 24*365], levels=None):
    df = create_fourier(df[['y']], freqs=freqs, levels=levels)
    for l in lags:
        df[f'lag_{l}'] = df['y'].shift(l)
    return df.dropna()

def create_sarimax_df(df, df_ext, lags=[24*365], lags_ext=[1,24,365*24], freqs=[24, 24*365], levels=None):
    df = create_fourier(df[['y']], freqs=freqs, levels=levels)
    df = df.merge(df_ext, how='left', left_index=True, right_index=True)
    df = df.rename(columns={
        'y_x': 'y',
        'y_y': 'x',
    })
    for l in lags:
        df[f'lag_{l}'] = df['y'].shift(l)

    for l in lags_ext:
        df[f"xlag_{l}"] = df['x'].shift(l)
    
    df = df.drop(['x'], axis=1)
    return df.dropna()

def extend_df(df, new_y, freq='H'):
    last_timestamp = df.index[-1]
    new_timestamps = pd.date_range(start=last_timestamp, periods=len(new_y)+1, freq=freq)[1:]
    new_data = pd.DataFrame({'y': new_y})
    new_data.index=new_timestamps
    new_df = pd.concat([df[['y']], new_data])
    new_df = new_df.asfreq('H')
    return new_df

df_eolica = read_dataset(data='eolica')
df_solar = read_dataset(data='solar')
# df = df.iloc[-24*365*3:]
# df_exog = df_exog.iloc[-24*365*3:]
df_eolica['y'] = np.clip(df_eolica['y'], 0.0001, np.inf)
df_eolica['y'], lbc_eolica = boxcox(df_eolica['y'])
df_solar['y'] = np.clip(df_solar['y'], 0.0001, np.inf)
df_solar['y'], lbc_solar = boxcox(df_solar['y'])

STEP_HORIZON = 1
FULL_HORIZON = 24*365

LAGS=[1,2,3,4,24]
LAGS_EXT=[1,2,3,4,24]
FREQS=[24,24*365]
LEVELS=[2,3]

Y_full_train_eolica = df_eolica.iloc[:-FULL_HORIZON][['y']]
Y_full_test_eolica = df_eolica.iloc[-FULL_HORIZON:][['y']]
Y_full_train_solar = df_solar.iloc[:-FULL_HORIZON][['y']]
Y_full_test_solar = df_solar.iloc[-FULL_HORIZON:][['y']]

X_train_solar = create_sarimax_df(Y_full_train_solar, Y_full_train_eolica, lags=LAGS, lags_ext=LAGS_EXT, freqs=FREQS, levels=LEVELS)
Y_train_solar = X_train_solar[['y']]
X_train_solar = X_train_solar.drop('y',axis=1)
model_solar = SVR(C=1.0, epsilon=0.2)
model_solar.fit(X_train_solar, Y_train_solar)

X_train_eolica = create_sarimax_df(Y_full_train_eolica, Y_full_train_solar, lags=LAGS, lags_ext=LAGS_EXT, freqs=FREQS, levels=LEVELS)
Y_train_eolica = X_train_eolica[['y']]
X_train_eolica = X_train_eolica.drop('y',axis=1)
model_eolica = SVR(C=1.0, epsilon=0.2)
model_eolica.fit(X_train_eolica, Y_train_eolica)

print("#### Starting SVM prediction")
n_laps = ceil(FULL_HORIZON/STEP_HORIZON)
repr_laps = n_laps//10
for i in range(n_laps):
    if (i+1)%repr_laps==0:
        print(f'Lap {i+1}/{n_laps}')
    if -FULL_HORIZON + (i+1)*STEP_HORIZON == 0:
        horizon = len(Y_full_test_solar.iloc[-FULL_HORIZON + i*STEP_HORIZON:])
    else:
        horizon = len(Y_full_test_solar.iloc[-FULL_HORIZON + i*STEP_HORIZON:-FULL_HORIZON + (i+1)*STEP_HORIZON])

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

print('\n###### Solar:')
Y_full_train_solar['y'] = inv_boxcox(Y_full_train_solar['y'], lbc_solar)
Y_full_test_solar['y'] = inv_boxcox(Y_full_test_solar['y'], lbc_solar)
test_results(Y_full_test_solar['y'].reset_index(drop=True), Y_full_train_solar.iloc[-FULL_HORIZON:]['y'].reset_index(drop=True))

print('\n###### Eolica:')
Y_full_train_eolica['y'] = inv_boxcox(Y_full_train_eolica['y'], lbc_eolica)
Y_full_test_eolica['y'] = inv_boxcox(Y_full_test_eolica['y'], lbc_eolica)
test_results(Y_full_test_eolica['y'].reset_index(drop=True), Y_full_train_eolica.iloc[-FULL_HORIZON:]['y'].reset_index(drop=True))

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
plt.savefig('svm_solar.png', dpi=300, bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize = (20, 7))
Y_hat_df = Y_full_test_eolica.merge(Y_full_train_eolica, how='left', left_index=True, right_index=True)
plot_df = Y_hat_df
plot_df[['y_x', 'y_y']].plot(ax=ax, linewidth=2)
ax.set_title('Eolica', fontsize=22)
ax.set_ylabel('Factor capacidad', fontsize=20)
ax.set_xlabel('Timestamp [t]', fontsize=20)
ax.legend(['Real','Prediction'])
ax.grid()
plt.savefig('svm_eolica.png', dpi=300, bbox_inches='tight')

# Save results
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_solar = f'svm_solar_{timestamp}.csv'
save_eolica = f'svm_eolica_{timestamp}.csv'
Y_full_train_solar.to_csv(f'./results/{save_solar}')
Y_full_train_eolica.to_csv(f'./results/{save_eolica}')