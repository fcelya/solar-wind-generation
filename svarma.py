import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from test_module import test_results
from statsmodels.tsa.statespace.varmax import VARMAX
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import warnings

warnings.filterwarnings('ignore')

def read_dataset(name='data/factor_capacidad_conjunto.csv'):
    df = pd.read_csv(name,index_col=0,parse_dates=True)
    return df

def create_fourier(df, freqs=[24,24*365]):
    T = len(df)
    t = np.arange(0,T)
    result_df = df.copy()
    for f in freqs:
        sin = np.sin(2*np.pi*t/f)
        result_df[f'sen_{f}'] = sin
        cos = np.cos(2*np.pi*t/f)
        result_df[f'cos_{f}'] = cos
    return result_df

def create_arima_df(df, lags=[24*365], freqs=[24, 24*365]):
    df = create_fourier(df[['solar','eolica']], freqs=freqs)
    for l in lags:
        for y in ('solar','eolica'):
            df[f'lag_{y}_{l}'] = df[y].shift(l)
    return df.dropna()

def extend_df(df, new_y, freq='H'):
    last_timestamp = df.index[-1]
    new_timestamps = pd.date_range(start=last_timestamp, periods=len(new_y)+1, freq=freq)[1:]
    new_data = pd.DataFrame({'y': new_y})
    new_data.index=new_timestamps
    new_df = pd.concat([df[['y']], new_data])
    new_df = new_df.asfreq('H')
    return new_df

df = read_dataset()
df = np.clip(df, 0.0001, np.inf)
df['solar'], lambda_boxcox_solar = boxcox(df['solar'])
df['eolica'], lambda_boxcox_eolica = boxcox(df['eolica'])

STEP_HORIZON = 24
FULL_HORIZON = 24*365

LAGS=[24]
FREQS=[24,24*365]
ORDER=(2,2)

Y_full_train = df.iloc[:-FULL_HORIZON][['solar','eolica']]
Y_full_test = df.iloc[-FULL_HORIZON:][['solar','eolica']]
Y_train = pd.DataFrame(Y_full_train)

X_train = create_arima_df(Y_train, lags=LAGS, freqs=FREQS)
model = VARMAX(X_train[['solar','eolica']], exog=X_train[X_train.columns.difference(['solar','eolica'])], order=ORDER)
results = model.fit()

n_laps = ceil(FULL_HORIZON/STEP_HORIZON)
for i in range(n_laps):
    print(f'Lap {i+1}/{n_laps}')
    if -FULL_HORIZON + (i+1)*STEP_HORIZON == 0:
        horizon = len(Y_full_test.iloc[-FULL_HORIZON + i*STEP_HORIZON:])
    else:
        horizon = len(Y_full_test.iloc[-FULL_HORIZON + i*STEP_HORIZON:-FULL_HORIZON + (i+1)*STEP_HORIZON])

    assert horizon <= min(LAGS)

    future_Y = extend_df(Y_train, pd.Series(np.ones(horizon)))
    X_train = create_arima_df(future_Y, lags=LAGS, freqs=FREQS)
    X_train = X_train[X_train.columns.difference(['solar','eolica'])]

    forecast = results.get_forecast(steps=horizon, exog=X_train.iloc[-horizon:])
    forecast = forecast.predicted_mean
    Y_train = extend_df(Y_train, forecast)
    # results.extend(Y_train)
    results = results.append(Y_train.iloc[-horizon:], exog=X_train.iloc[-horizon:], refit=False)

Y_train['solar'] = inv_boxcox(Y_train['solar'], lambda_boxcox_solar)
Y_full_test['eolica'] = inv_boxcox(Y_full_test['eolica'], lambda_boxcox_eolica)
test_results(Y_full_test['solar'].reset_index(drop=True), Y_train.iloc[-FULL_HORIZON:]['solar'].reset_index(drop=True))
print(results.summary())
test_results(Y_full_test['eolica'].reset_index(drop=True), Y_train.iloc[-FULL_HORIZON:]['eolica'].reset_index(drop=True))
print(results.summary())
# Plot predictions
fig, ax = plt.subplots(1, 1, figsize = (20, 7))
Y_hat_df = Y_full_test.merge(Y_train, how='left', left_index=True, right_index=True)
# plot_df = pd.concat([Y_train_df, Y_hat_df]).set_index('ds')
plot_df = Y_hat_df
# plot_df[['y', 'NHITS']].plot(ax=ax, linewidth=2)
plot_df[['eolica_x', 'eolica_y']].plot(ax=ax, linewidth=2)
ax.set_title('Eolica', fontsize=22)
ax.set_ylabel('Factor capacidad', fontsize=20)
ax.set_xlabel('Timestamp [t]', fontsize=20)
ax.legend(['Real','Prediction'])
ax.grid()

plt.savefig('svarma.png', dpi=300, bbox_inches='tight')
# plt.show()
pass