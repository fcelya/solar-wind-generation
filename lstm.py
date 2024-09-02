import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
from neuralforecast.losses.pytorch import RMSE
import matplotlib.pyplot as plt
import torch
import numpy as np
from math import ceil
from test_module import test_results

def read_dataset(name='data/factor_capacidad_conjunto.csv',data='solar'):
    df = pd.read_csv(name,index_col=0,parse_dates=True)
    df = df.rename(columns={
        data: 'y'
    })
    df['ds'] = df.index
    df.loc[:,'unique_id'] = 0
    return df[['y', 'ds', 'unique_id']]

def extend_df(df, new_y, unique_id, freq='H'):
    last_timestamp = df['ds'].iloc[-1]
    new_timestamps = pd.date_range(start=last_timestamp, periods=len(new_y)+1, freq=freq)[1:]
    new_data = pd.DataFrame({'ds': new_timestamps, 'y': new_y, 'unique_id':unique_id})
    new_df = pd.concat([df, new_data], ignore_index=True).reset_index(drop=True)
    return new_df


# Split data and declare panel dataset
Y_df = read_dataset(data='eolica')
# Y_df = Y_df.iloc[-24*365*3:]

STEP_HORIZON = 24*365//6
LAG = 4*STEP_HORIZON
FULL_HORIZON = 24*365

Y_full_train = Y_df.iloc[:-FULL_HORIZON]
Y_full_test = Y_df.iloc[-FULL_HORIZON:]
Y_train = pd.DataFrame(Y_full_train)

horizon = STEP_HORIZON
lstm_params={
    'encoder_n_layers': 2,
    'encoder_hidden_size': 200,
    'decoder_hidden_size': 200,
    'context_size': LAG,
    # 'hist_exog_list': Y_train.columns.difference(['y','ds','unique_id']),
    'loss':RMSE(),
    'max_steps': 1000,
}
with torch.no_grad():
    models = [LSTM(
        h=horizon,
        **lstm_params)]
    nf = NeuralForecast(models=models, freq='H', local_scaler_type='boxcox')
    nf.fit(df=Y_train, verbose=True)

n_laps = ceil(FULL_HORIZON/STEP_HORIZON)
for i in range(n_laps):
    print(f'Lap {i+1}/{n_laps}')
    # Fit and predict with NBEATS and NHITS models
    if -FULL_HORIZON + (i+1)*STEP_HORIZON == 0:
        horizon = len(Y_full_test.iloc[-FULL_HORIZON + i*STEP_HORIZON:])
    else:
        horizon = len(Y_full_test.iloc[-FULL_HORIZON + i*STEP_HORIZON:-FULL_HORIZON + (i+1)*STEP_HORIZON])
    Y_future = extend_df(Y_train, pd.Series(np.ones(horizon)), unique_id=0)
    Y_hat_df = nf.predict(df=Y_train, futr_df=Y_future[['unique_id','ds']], verbose=False).reset_index()
    Y_train = extend_df(Y_train, Y_hat_df['LSTM'], unique_id=0)
Y_train['y'] = np.clip(Y_train['y'],0,np.inf)
test_results(Y_full_test['y'].reset_index(drop=True), Y_train.iloc[-FULL_HORIZON:]['y'].reset_index(drop=True))

# Plot predictions
fig, ax = plt.subplots(1, 1, figsize = (20, 7))
Y_hat_df = Y_full_test.merge(Y_train, how='left', on=['unique_id', 'ds'])
# plot_df = pd.concat([Y_train_df, Y_hat_df]).set_index('ds')
plot_df = Y_hat_df
# plot_df[['y', 'NHITS']].plot(ax=ax, linewidth=2)
plot_df[['y_x', 'y_y']].plot(ax=ax, linewidth=2)
ax.set_title('Solar', fontsize=22)
ax.set_ylabel('Factor capacidad', fontsize=20)
ax.set_xlabel('Timestamp [t]', fontsize=20)
ax.legend(['Real','Prediction'])
ax.grid()

# plt.savefig('test.png', dpi=300, bbox_inches='tight')
plt.show()
pass