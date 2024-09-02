import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sktime.forecasting.arima import AutoARIMA
import matplotlib.pyplot as plt

def read_dataset(name='data/factor_capacidad_conjunto.csv',data='solar'):
    df = pd.read_csv(name,index_col=0,parse_dates=True)
    df.index.freq = 'H'
    return df[data]

# Split data and declare panel dataset
Y_df = read_dataset(data='eolica')
Y_df = Y_df.iloc[-1000:]
# print('Read dataset')
# Y_train_df = Y_df[Y_df.index<='2023-01-01']
# Y_test_df = Y_df[Y_df.index>'2023-01-01']
Y_train_df = Y_df.iloc[:-200]
Y_test_df = Y_df.iloc[-200:]

# Fit and predict with NBEATS and NHITS models
horizon = len(Y_test_df)
forecaster = AutoARIMA(sp=24, d=0, max_p=2, max_q=2, suppress_warnings=True)  
print('Fitting models...')
forecaster.fit(Y_train_df)  
print('Predicting models...')
Y_hat_df = forecaster.predict(fh=[i for i in range(1,horizon+1)])
print('Plotting...')
# Plot predictions
fig, ax = plt.subplots(1, 1, figsize = (20, 7))
plt_df = pd.concat([Y_train_df.iloc[-horizon:], Y_test_df])
ax.plot(plt_df.index, plt_df.values)
ax.plot(Y_test_df.index, Y_hat_df)
ax.set_title('Eolica', fontsize=22)
ax.set_ylabel('Factor capacidad', fontsize=20)
ax.set_xlabel('Timestamp [t]', fontsize=20)
ax.legend(['Real','Prediction'])
ax.grid()

plt.savefig('test.png', dpi=300, bbox_inches='tight')