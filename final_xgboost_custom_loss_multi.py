import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import warnings
import xgboost as xgb
from datetime import datetime
from time import time
# from sera_loss import SERALossXGBoost
from test_module import save_results, Test
from scipy.interpolate import CubicHermiteSpline


warnings.filterwarnings('ignore')

def read_dataset(name='data/factor_capacidad.csv',data='solar'):
    df = pd.read_csv(name,index_col=0,parse_dates=True)
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

def sera_loss(y_true, y_pred,min_p=5,max_p=95,sense="+"):
    mini = np.min(y_true)-1e-3
    maxi = np.max(y_true)+1e-3
    semi1 = np.percentile(y_true, min_p)
    semi2 = np.percentile(y_true, max_p)
    if sense=='+':
        pchip = CubicHermiteSpline([mini,semi1,semi2,maxi],[0,0,1,1],[0,0,0,0])
    elif sense=='-':
        pchip = CubicHermiteSpline([mini,semi1,semi2,maxi],[1,1,0,0],[0,0,0,0])
    else:
        raise ValueError("sense must be either '+' or '-'")
    pchip_v = np.vectorize(pchip)
    phis = pchip_v(y_true)

    grad = 2*(y_pred - y_true)*phis
    hess = 2*phis
    
    return grad, hess

class SERA():
    def __init__(self, y_true, y_pred, dt=.001, min_p=10, max_p=90,sense="+"):
        self.y_true = y_true.to_numpy()
        self.y_pred = y_pred.to_numpy()
        self.dt = dt
        self.phis = self.calc_phis(min_p, max_p)
        if sense != '+' and sense != '-':
            raise ValueError("sense must be '+' or '-'")
        self.sense = sense

    def calc_phis(self, min_p,max_p):
        mini = np.min(self.y_true)-1e-3
        maxi = np.max(self.y_true)+1e-3
        semi1 = np.percentile(self.y_true, min_p)
        semi2 = np.percentile(self.y_true, max_p)
        if self.sense == '-':
            pchip = CubicHermiteSpline([mini,semi1,semi2,maxi],[1,1,0,0],[0,0,0,0])
        elif self.sense=='+':
            pchip = CubicHermiteSpline([mini,semi1,semi2,maxi],[0,0,1,1],[0,0,0,0])
        pchip_v = np.vectorize(pchip)
        return pchip_v(self.y_true) # We reutrn the phi(y_i) for every y_i

    def calc_SER(self, t):
        I = self.phis >= t
        return ((self.y_true-self.y_pred)**2)@I.T
    
    def calc_SERA(self):
        ts = np.linspace(0,1,int(1/self.dt))
        sers = np.vectorize(self.calc_SER)(ts)
        return sers.sum()*self.dt

class SERALossXGBoost(object):
    def __init__(self,y_true,min_p=10,max_p=90,sense='-'):
        self.min_p=min_p
        self.max_p=max_p
        mini = np.min(y_true)-1e-3
        maxi = np.max(y_true)+1e-3
        semi1 = np.percentile(y_true, self.min_p)
        semi2 = np.percentile(y_true, self.max_p)
        if sense=='-':
            pchip = CubicHermiteSpline([mini,semi1,semi2,maxi],[1,1,0,0],[0,0,0,0])
        elif sense=='+':
            pchip = CubicHermiteSpline([mini,semi1,semi2,maxi],[0,0,1,1],[0,0,0,0])
        else:
            raise ValueError("sense must be '+' or '-'")
        pchip_v = np.vectorize(pchip)
        self.phis = pchip_v(y_true)

    def sera_loss(self,y_true, y_pred):
        grad = 2*(y_pred - y_true)*self.phis
        hess = 2*self.phis
        
        return grad, hess

STEP_HORIZON = 1
FULL_HORIZON = 24*365
LAGS=[1,2,3,4,12,24,24*2]
LAGS_EXT=[1,2,3,4,12,24,24*2]
FREQS=[24,24*365]
LEVELS=[2,4]

N_ESTIMATORS = 8
SUBSAMPLE = .7
MAX_DEPTH = 4

start = time()
df_wind = read_dataset(data='wind')
df_solar_th = read_dataset(data='solar_th')
df_solar_pv = read_dataset(data='solar_pv')

# df_wind['wind'] = np.clip(df_wind['wind'], 0.0001, np.inf)
# df_wind['wind'], lbc_eolica = boxcox(df_wind['wind'])
# df_solar_th['solar_th'] = np.clip(df_solar_th['solar_th'], 0.0001, np.inf)
# df_solar_th['solar_th'], lbc_solar_th = boxcox(df_solar_th['solar_th'])
# df_solar_pv['solar_pv'] = np.clip(df_solar_pv['solar_pv'], 0.0001, np.inf)
# df_solar_pv['solar_pv'], lbc_solar_pv = boxcox(df_solar_pv['solar_pv'])

Y_full_train_wind = df_wind.iloc[:-FULL_HORIZON][['wind']]
Y_full_test_wind = df_wind.iloc[-FULL_HORIZON:][['wind']]
Y_full_train_solar_th = df_solar_th.iloc[:-FULL_HORIZON][['solar_th']]
Y_full_test_solar_th = df_solar_th.iloc[-FULL_HORIZON:][['solar_th']]
Y_full_train_solar_pv = df_solar_pv.iloc[:-FULL_HORIZON][['solar_pv']]
Y_full_test_solar_pv = df_solar_pv.iloc[-FULL_HORIZON:][['solar_pv']]

MIN_P=10
MAX_P=90
LOSS_SOLAR_PV = 'reg:squarederror'
LOSS_SOLAR_TH = 'reg:squarederror'

print(f'Preparation: {time()-start:.1f} s')
start = time()
df_ext = pd.merge(Y_full_train_solar_th,Y_full_train_wind,how='outer',left_index=True,right_index=True)
X_train_solar_pv = create_sarimax_df(Y_full_train_solar_pv, df_ext, lags=LAGS, lags_ext=LAGS_EXT, freqs=FREQS, levels=LEVELS)
Y_train_solar_pv = X_train_solar_pv[['solar_pv']]
X_train_solar_pv = X_train_solar_pv.drop('solar_pv',axis=1)
model_solar_pv = xgb.XGBRegressor(n_estimators=N_ESTIMATORS, verbosity=2,subsample=SUBSAMPLE,max_depth=MAX_DEPTH,objective=LOSS_SOLAR_PV)
model_solar_pv.fit(X_train_solar_pv, Y_train_solar_pv, verbose=True)

df_ext = pd.merge(Y_full_train_solar_pv,Y_full_train_wind,how='outer',left_index=True,right_index=True)
X_train_solar_th = create_sarimax_df(Y_full_train_solar_th, df_ext, lags=LAGS, lags_ext=LAGS_EXT, freqs=FREQS, levels=LEVELS)
Y_train_solar_th = X_train_solar_th[['solar_th']]
X_train_solar_th = X_train_solar_th.drop('solar_th',axis=1)
model_solar_th = xgb.XGBRegressor(n_estimators=N_ESTIMATORS, verbosity=2,subsample=SUBSAMPLE,max_depth=MAX_DEPTH,objective=LOSS_SOLAR_TH)
model_solar_th.fit(X_train_solar_th, Y_train_solar_th, verbose=True)

df_ext = pd.merge(Y_full_train_solar_pv,Y_full_train_solar_th,how='outer',left_index=True,right_index=True)
X_train_wind = create_sarimax_df(Y_full_train_wind, df_ext, lags=LAGS, lags_ext=LAGS_EXT, freqs=FREQS, levels=LEVELS)
Y_train_wind = X_train_wind[['wind']]
X_train_wind = X_train_wind.drop('wind',axis=1)
loss_class_eolica = SERALossXGBoost(Y_train_wind['wind'],MIN_P,MAX_P)
LOSS_WIND = loss_class_eolica.sera_loss #'reg:squarederror'
model_wind = xgb.XGBRegressor(n_estimators=N_ESTIMATORS, verbosity=2,subsample=SUBSAMPLE,max_depth=MAX_DEPTH,objective=LOSS_WIND)
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

print(f'Prediction: {time()-start:.1f} s')

# Y_full_train_wind = invert_box_cox(Y_full_train_wind,lbc_wind)
# Y_full_test_wind = invert_box_cox(Y_full_test_wind,lbc_wind)
# Y_full_train_solar_pv = invert_box_cox(Y_full_train_solar_pv,lbc_solar_pv)
# Y_full_test_solar_pv = invert_box_cox(Y_full_test_solar_pv,lbc_solar_pv)
# Y_full_train_solar_th = invert_box_cox(Y_full_train_solar_th,lbc_solar_th)
# Y_full_test_solar_th = invert_box_cox(Y_full_test_solar_th,lbc_solar_th)

df_obs = Y_full_test_solar_pv[['solar_pv']].merge(Y_full_test_solar_th[['solar_th']],left_index=True,right_index=True).merge(Y_full_test_wind[['wind']],left_index=True,right_index=True)
df_pred = Y_full_train_solar_pv[['solar_pv']].merge(Y_full_train_solar_th[['solar_th']],left_index=True,right_index=True).merge(Y_full_train_wind[['wind']],left_index=True,right_index=True)
df_pred = df_pred.iloc[-FULL_HORIZON:,:]

model = 'xgboost'
params = {
    'lags': LAGS,
    'lags_ext': LAGS_EXT,
    'freqs': FREQS,
    'levels': LEVELS,
    'n_estimators':N_ESTIMATORS, 
    'subsample':SUBSAMPLE,
    'max_depth':MAX_DEPTH,
    'objective':{
        'solar_pv': LOSS_SOLAR_PV,
        'solar_th': LOSS_SOLAR_TH,
        'wind': LOSS_WIND,
        }
}
experiment = {
    'step_horizon': STEP_HORIZON,
    'full_horizon': FULL_HORIZON,
    'obs': df_obs
}
prediction = df_pred
model_objects = {
    'solar_pv': model_solar_pv,
    'solar_th': model_solar_th,
    'wind': model_wind,
}

save_results(model,params,experiment,prediction,model_objects)

test = Test(df_obs,df_pred)
test.save_test_results('xgboost')
df_test_results = test.get_results_df()
print(df_test_results)