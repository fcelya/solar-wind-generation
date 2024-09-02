import pandas as pd
import numpy as np
from math import ceil
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
import matplotlib.pyplot as plt
import torch
from test_module import test_results
from sera_loss import SERALoss

def read_dataset(name='data/factor_capacidad_conjunto.csv',variable='solar',add_dateinfo=False):
    variables = set(['solar','eolica'])
    exog = list(variables - set([variable]))[0]
    df = pd.read_csv(name,index_col=0,parse_dates=True)
    vars = list(df.columns)
    assert len(vars)==2
    df = df.rename(columns={
        variable:'y',
        exog:'x'
    })
    df['ds'] = pd.to_datetime(df.index)
    if add_dateinfo:
        df['day'] = df['ds'].dt.dayofyear
        df['hour'] = df['ds'].dt.hour
        futr_exog = ['day','hour']
    else:
        futr_exog=None
    df.loc[:,'unique_id'] = 0
    return df, futr_exog

def extend_df(df, new_y, freq='H', futr_exog=False):
    last_timestamp = df['ds'].iloc[-1]
    new_timestamps = pd.date_range(start=last_timestamp, periods=len(new_y[list(new_y.keys())[0]])+1, freq=freq)[1:]
    if futr_exog:
        df_new = pd.DataFrame({'ds': new_timestamps, 'y': new_y['y'], 'x': new_y['x'], 'unique_id':0, 'day':new_timestamps.dayofyear, 'hour':new_timestamps.hour})
    else:
        df_new = pd.DataFrame({'ds': new_timestamps, 'y': new_y['y'], 'x': new_y['x'], 'unique_id':0})
    new_df = pd.concat([df, df_new], ignore_index=True).reset_index(drop=True)
    return new_df


# Split data and declare panel dataset
Y_df_solar, _ = read_dataset(variable='solar')
Y_df_eolica, _ = read_dataset(variable='eolica')

STEP_HORIZON = 24*365//16
LAG = 4*STEP_HORIZON
FULL_HORIZON = 24*365

Y_full_train_solar = Y_df_solar.iloc[:-FULL_HORIZON].copy()
Y_full_train_eolica = Y_df_eolica.iloc[:-FULL_HORIZON].copy()
# if futr_exog:
#     id_vars = ['ds'] + futr_exog
# else:
#     id_vars = ['ds']
Y_full_test_solar = Y_df_solar.iloc[-FULL_HORIZON:]
Y_train_solar = Y_full_train_solar.copy()
Y_full_test_eolica = Y_df_eolica.iloc[-FULL_HORIZON:]
Y_train_eolica = Y_full_train_eolica.copy()

horizon = STEP_HORIZON
nhits_params = {
    'n_freq_downsample':[365*24,24,1],
    'n_pool_kernel_size': [2,2,1], # same size as n_freq_downsample
    'pooling_mode': 'MaxPool1d',#, 'AvgPool1d'
    'scaler_type': 'identity',#standard, robust, minmax, minmax1, invariant, revin
    'mlp_units': [[512,512],[512,512],[512,512]],
    'max_steps': 200,
    # 'futr_exog_list': futr_exog,
    'hist_exog_list': ['x'],
    'loss':SERALoss()
}
with torch.no_grad():
    models = [NHITS(
        input_size=LAG,
        h=horizon,
        **nhits_params)]
    nf_solar = NeuralForecast(models=models, freq='H', local_scaler_type='boxcox')
    nf_solar.fit(df=Y_train_solar, verbose=True)
    nf_eolica = NeuralForecast(models=models, freq='H', local_scaler_type='boxcox')
    nf_eolica.fit(df=Y_train_eolica, verbose=True)

n_laps = ceil(FULL_HORIZON/STEP_HORIZON)
for i in range(n_laps):
    print(f'Lap {i+1}/{n_laps}')
    if -FULL_HORIZON + (i+1)*STEP_HORIZON == 0:
        horizon = len(Y_full_test_solar.iloc[-FULL_HORIZON + i*STEP_HORIZON:])
    else:
        horizon = len(Y_full_test_solar.iloc[-FULL_HORIZON + i*STEP_HORIZON:-FULL_HORIZON + (i+1)*STEP_HORIZON])
        if horizon==0:
            break
    new_y = pd.Series(np.ones(horizon))
    Y_future_solar = extend_df(Y_train_solar.copy(), {'y':new_y, 'x':new_y})
    Y_hat_df_solar = nf_solar.predict(df=Y_train_solar,futr_df=Y_future_solar.drop(['y'], axis=1), verbose=True).reset_index()
    new_y_solar = Y_hat_df_solar['NHITS']
    Y_future_eolica= extend_df(Y_train_eolica.copy(), {'y':new_y, 'x':new_y})
    Y_hat_df_eolica = nf_eolica.predict(df=Y_train_eolica,futr_df=Y_future_eolica.drop(['y'], axis=1), verbose=True).reset_index()
    new_y_eolica = Y_hat_df_eolica['NHITS']

    Y_train_solar = extend_df(Y_train_solar, {'y':new_y_solar, 'x':new_y_eolica})
    Y_train_eolica = extend_df(Y_train_eolica, {'x':new_y_solar, 'y':new_y_eolica})
Y_train_solar['y'] = np.clip(Y_train_solar['y'],0,np.inf)
Y_train_eolica['y'] = np.clip(Y_train_eolica['y'],0,np.inf)
print('##### Solar:')
test_results(Y_full_test_solar['y'][-FULL_HORIZON:].reset_index(drop=True), Y_train_solar['y'][-FULL_HORIZON:].reset_index(drop=True))
print('##### Eolica:')
test_results(Y_full_test_solar['y'][-FULL_HORIZON:].reset_index(drop=True), Y_train_solar['y'][-FULL_HORIZON:].reset_index(drop=True))
