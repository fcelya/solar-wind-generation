import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import matplotlib.pyplot as plt

def read_dataset(name='data/factor_capacidad_conjunto.csv',data='solar'):
    df = pd.read_csv(name,index_col=0,parse_dates=True)
    df.index.freq = 'H'
    return df[data]

# Split data and declare panel dataset
Y_df = read_dataset(data='eolica')
print('Read dataset')
Y_train_df = Y_df[Y_df.index<='2023-01-01']
Y_test_df = Y_df[Y_df.index>'2023-01-01']

# Fit and predict with NBEATS and NHITS models
horizon = len(Y_test_df)
sf = StatsForecast(
    models = [AutoARIMA(season_length = 24)],
    freq='h'
)

print('Fitting models...')
sf.fit(Y_train_df.values)
print('Predicting models...')
Y_hat_df = sf.predict(h=horizon, level=[95])
print('Plotting...')
# Plot predictions
fig, ax = plt.subplots(1, 1, figsize = (20, 7))
Y_hat_df = Y_test_df.merge(Y_hat_df, how='left', on=['unique_id', 'ds'])
plot_df = pd.concat([Y_train_df, Y_hat_df]).set_index('ds')

plot_df[['y', 'NBEATS', 'NHITS']].plot(ax=ax, linewidth=2)

ax.set_title('AirPassengers Forecast', fontsize=22)
ax.set_ylabel('Monthly Passengers', fontsize=20)
ax.set_xlabel('Timestamp [t]', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()

plt.savefig('test.png', dpi=300, bbox_inches='tight')