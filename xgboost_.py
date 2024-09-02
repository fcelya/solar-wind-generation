import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from test_module import test_results
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import warnings
import xgboost as xgb

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

df = read_dataset(data='solar')
df_exog = read_dataset(data='eolica')
# df = df.iloc[-24*365*3:]
# df_exog = df_exog.iloc[-24*365*3:]
df['y'] = np.clip(df['y'], 0.0001, np.inf)
df['y'], lambda_boxcox = boxcox(df['y'])
df_exog['y'] = boxcox(df_exog['y'], lambda_boxcox)

STEP_HORIZON = 1
FULL_HORIZON = 24*180

LAGS=[1,2,3,4,5,6,24]
LAGS_EXT=[1]
FREQS=[24,24*365]
LEVELS=[2,3]

Y_full_train = df.iloc[:-FULL_HORIZON]['y']
Y_full_test = df.iloc[-FULL_HORIZON:][['y']]
Y_train = pd.DataFrame(Y_full_train)

X_train = create_sarimax_df(Y_train, df_exog, lags=LAGS, lags_ext=LAGS_EXT, freqs=FREQS, levels=LEVELS)
Y_train = X_train[['y']]
X_train = X_train.drop('y',axis=1)
# model = ARIMA(X_train['y'], exog=X_train[X_train.columns.difference(['y'])], order=ORDER)
# results = model.fit()
model = xgb.XGBRegressor(n_estimators=1000, verbosity=2,subsample=.9,max_depth=6)
model.fit(X_train, Y_train, verbose=True)

n_laps = ceil(FULL_HORIZON/STEP_HORIZON)
repr_laps = n_laps//20
for i in range(n_laps):
    if (i+1)%repr_laps==0:
        print(f'Lap {i+1}/{n_laps}')
    if -FULL_HORIZON + (i+1)*STEP_HORIZON == 0:
        horizon = len(Y_full_test.iloc[-FULL_HORIZON + i*STEP_HORIZON:])
    else:
        horizon = len(Y_full_test.iloc[-FULL_HORIZON + i*STEP_HORIZON:-FULL_HORIZON + (i+1)*STEP_HORIZON])

    assert horizon <= min(LAGS)

    future_Y = extend_df(Y_train, pd.Series(np.ones(horizon)))
    X_train = create_sarimax_df(future_Y, df_ext=df_exog, lags=LAGS, lags_ext=LAGS_EXT, freqs=FREQS, levels=LEVELS)
    X_train = X_train.drop('y',axis=1)

    forecast = model.predict(X_train.iloc[-horizon:])
    Y_train = extend_df(Y_train, forecast)
    # results.extend(Y_train)
    # results = results.append(Y_train.iloc[-horizon:], exog=X_train.iloc[-horizon:], refit=False)

Y_train['y'] = inv_boxcox(Y_train['y'], lambda_boxcox)
Y_full_test['y'] = inv_boxcox(Y_full_test['y'], lambda_boxcox)
test_results(Y_full_test['y'].reset_index(drop=True), Y_train.iloc[-FULL_HORIZON:]['y'].reset_index(drop=True))
# Plot predictions
fig, ax = plt.subplots(1, 1, figsize = (20, 7))
Y_hat_df = Y_full_test.merge(Y_train, how='left', left_index=True, right_index=True)
# plot_df = pd.concat([Y_train_df, Y_hat_df]).set_index('ds')
plot_df = Y_hat_df
# plot_df[['y', 'NHITS']].plot(ax=ax, linewidth=2)
plot_df[['y_x', 'y_y']].plot(ax=ax, linewidth=2)
ax.set_title('Solar', fontsize=22)
ax.set_ylabel('Factor capacidad', fontsize=20)
ax.set_xlabel('Timestamp [t]', fontsize=20)
ax.legend(['Real','Prediction'])
ax.grid()

plt.savefig('xgboost.png', dpi=300, bbox_inches='tight')
# plt.show()
pass