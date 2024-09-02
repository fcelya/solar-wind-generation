from hyperopt import hp, fmin, tpe, Trials
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from test_module import test_results
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import warnings

warnings.filterwarnings('ignore')

class cSARIMAX():
    def __init__(self, full_horizon, step_horizon, params, endog='solar', exog='eolica'):
        self.full_horizon = full_horizon
        self.step_horizon = step_horizon
        self.n_laps = ceil(full_horizon//step_horizon)
        self.lags = params['lags']
        self.lags_ext=params['lags_ext']
        self.freqs=params['freqs']
        self.order=(params['ar_order'], params['i_order'], params['ma_order'])
        self.endog = endog
        self.exog = exog
        self.df_endog, self.lambda_bc_endog = self.read_dataset(data=endog)
        self.df_exog, self.lambda_bc_exog = self.read_dataset(data=exog)
        self.result_tests = None
        self.create_x_train()
        self.create_model()

    
    def read_dataset(self, name='data/factor_capacidad_conjunto.csv',data='solar', last=None):
        df = pd.read_csv(name,index_col=0,parse_dates=True)
        df = df.rename(columns={
            data: 'y'
        })
        df = df[['y']]
        
        if last:
            df = df.iloc[-last:]
        df['y'] = np.clip(df['y'], 0.0001, np.inf)
        df['y'], lambda_boxcox = boxcox(df['y'])
        return df, lambda_boxcox
    
    def create_x_train(self):
        self.Y_full_train = self.df_endog.iloc[:-self.full_horizon]['y']
        self.Y_full_test = self.df_endog.iloc[-self.full_horizon:][['y']]
        self.Y_train = pd.DataFrame(self.Y_full_train)

        self.X_train = self.create_sarimax_df(self.Y_train, self.df_exog)

    def create_model(self):   
        self.model = ARIMA(self.X_train['y'], exog=self.X_train[self.X_train.columns.difference(['y'])], order=self.order)

    def create_fourier(self, df, levels=None):
        if levels is not None:
            assert len(levels) == len(self.freqs)
        else:
            levels = [1 for _ in self.freqs]

        T = len(df)
        t = np.arange(0,T)
        result_df = df.copy()
        for i, f in enumerate(self.freqs):
            for l in range(1, levels[i]+1):
                sin = np.sin(2*np.pi*t*l/f)
                result_df[f'sen_f{f}_l{l}'] = sin
                cos = np.cos(2*np.pi*t*l/f)
                result_df[f'cos_f{f}_l{l}'] = cos
        return result_df

    def create_arima_df(self, df):
        df = self.create_fourier(self.df[['y']], freqs=self.freqs)
        for l in self.lags:
            df[f'lag_{l}'] = df['y'].shift(l)
        return df.dropna()

    def create_sarimax_df(self, df, df_ext):
        df = self.create_fourier(df[['y']])
        df = df.merge(df_ext, how='left', left_index=True, right_index=True)
        df = df.rename(columns={
            'y_x': 'y',
            'y_y': 'x',
        })
        for l in self.lags:
            df[f'lag_{l}'] = df['y'].shift(l)

        for l in self.lags_ext:
            df[f"xlag_{l}"] = df['x'].shift(l)
        
        df = df.drop(['x'], axis=1)
        return df.dropna()

    def extend_df(self, df, new_y, freq='H'):
        last_timestamp = df.index[-1]
        new_timestamps = pd.date_range(start=last_timestamp, periods=len(new_y)+1, freq=freq)[1:]
        new_data = pd.DataFrame({'y': new_y})
        new_data.index=new_timestamps
        new_df = pd.concat([df[['y']], new_data])
        new_df = new_df.asfreq('H')
        return new_df
    
    def fit(self):
        self.results = self.model.fit()
    
    def prediction_loop(self):
        for i in range(self.n_laps):
            print(f'Lap {i+1}/{self.n_laps}')
            if -self.full_horizon + (i+1)*self.step_horizon == 0:
                horizon = len(self.Y_full_test.iloc[-self.full_horizon + i*self.step_horizon:])
            else:
                horizon = len(self.Y_full_test.iloc[-self.full_horizon + i*self.step_horizon:-self.full_horizon + (i+1)*self.step_horizon])

            assert horizon <= min(self.lags)

            future_Y = self.extend_df(self.Y_train, pd.Series(np.ones(horizon)))
            self.X_train = self.create_sarimax_df(future_Y, df_ext=self.df_exog)
            self.X_train = self.X_train[self.X_train.columns.difference(['y'])]

            forecast = self.results.get_forecast(steps=horizon, exog=self.X_train.iloc[-horizon:])
            forecast = forecast.predicted_mean
            self.Y_train = self.extend_df(self.Y_train, forecast)
            # results.extend(Y_train)
            self.results = self.results.append(self.Y_train.iloc[-horizon:], exog=self.X_train.iloc[-horizon:], refit=False)
        
        self.Y_train['y'] = inv_boxcox(self.Y_train['y'], self.lambda_bc_endog)
        self.Y_full_test['y'] = inv_boxcox(self.Y_full_test['y'], self.lambda_bc_endog)

    def display_results(self):
        print(self.results.summary())

    def test_results(self):
        self.result_tests = test_results(self.Y_full_test['y'].reset_index(drop=True), self.Y_train.iloc[-self.full_horizon:]['y'].reset_index(drop=True))

    def get_rmse(self):
        if self.result_tests is None:
            self.test_results()
        return self.result_tests[0][0]
    
    def get_mae(self):
        if self.result_tests is None:
            self.test_results()
        return self.result_tests[0][1]
    
    def get_bias(self):
        if self.result_tests is None:
            self.test_results()
        return self.result_tests[0][2]

    def plot_predictions(self, save_fig=True, show_fig=True, fig_name='sarimax.png'):
        # Plot predictions
        fig, ax = plt.subplots(1, 1, figsize = (20, 7))
        Y_hat_df = self.Y_full_test.merge(self.Y_train, how='left', left_index=True, right_index=True)
        plot_df = Y_hat_df
        plot_df[['y_x', 'y_y']].plot(ax=ax, linewidth=2)
        ax.set_title(self.endog, fontsize=22)
        ax.set_ylabel('Factor capacidad', fontsize=20)
        ax.set_xlabel('Timestamp [t]', fontsize=20)
        ax.legend(['Real','Prediction'])
        ax.grid()
        if save_fig:
            plt.savefig(fig_name, dpi=300, bbox_inches='tight')
        if show_fig:
            plt.show()
# Define the hyperparameter space
            
space = {
    'ar_order': hp.choice('ar_order', [0,1,2,3,4,5,6,7,8]),
    'i_order': hp.choice('i_order', [0,1,2]),
    'ma_order': hp.choice('ma_order', [0,1,2,3,4,5,6,7,8]),
    'lags': hp.choice('lags', [[24], [24*365], [24, 24*365]]),
    'lags_ext': hp.choice('lags_ext', [[24], [24*365], [24, 24*365]]),
    'freqs': hp.choice('freqs', [[24], [24*365], [24, 24*365]])

}
FULL_HORIZON=24*365
STEP_HORIZON=24

# Define the objective function
def objective(params):

    # GET Y_TRAIN, CREATE MODEL WITH PARAMETERS AND FIT
    sarimax = cSARIMAX(FULL_HORIZON, STEP_HORIZON, params)
    sarimax.fit()

    # PREDICT
    sarimax.prediction_loop()

    # GET Y_TEST AND EVALUATE
    rmse = sarimax.get_rmse()

    return rmse

# Run the optimization
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)

print('Best hyperparameters: ', best)
