import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cramervonmises, ecdf, entropy, rankdata, genextreme
from sklearn.neighbors import KernelDensity
from copulas.univariate import GaussianKDE
from copulas.multivariate import GaussianMultivariate
import time
from datetime import datetime
import os.path
import pickle as pkl

def save_results(model:str, params:dict, experiment:dict, prediction:pd.DataFrame,model_objects:dict=None,save_path='testing'):
    """
    model: String. Name of the model.
    params: Dictionary. Parameters used to characterize the model.
    experiment: Dictionary. Keys are ['step_horizon':int,'full_horizon':int,'obs':pd.DataFrame]
    prediction: pd.DataFrame. DataFrame of predicitons
    """
    results = {}
    results['model'] = model
    results['params'] = params
    results['experiment'] = experiment
    results['prediction'] = prediction
    if model_objects: results['model_objects'] = model_objects
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_name = os.path.join(save_path,f"{model.lower()}_{timestamp}.pkl")
    with open(save_name,'wb') as f:
        pkl.dump(results,f)
    return

def load_results(path):
    with open(path,'rb') as f:
        results = pkl.load(f)
    df_obs = results['experiment']['obs']
    df_pred = results['prediction']
    return df_obs, df_pred, results

class Test():
    def __init__(self, df_obs, df_pred,verbose=True):
        if set(df_obs.columns)!=set(df_pred.columns):
            raise ValueError("Column names of df_obs and df_pred must be the same")
        self.df_obs_full = df_obs.copy()
        self.df_pred = df_pred.copy()
        self.df_obs = df_obs[df_obs.index.isin(set(df_pred.index))]
        if len(self.df_obs) != len(self.df_pred):
            raise ValueError("df_pred and df_obs have incongruent indeces")
        self.columns = df_obs.columns
        self.v = verbose

        self.testing_version = '001'

        self.get_results()

    def get_results(self):
        if self.v: t1 = time.time()
        results = {}
        results['cramer_von_mises'] = self.cramer_von_mises()
        results['kl_divergence'] = self.kl_divergence()
        results['acf_dist'] = self.acf_dist()
        results['ccmd'] = self.ccmd()
        results['ccf_dist'] = self.ccf_dist()
        results['cvar'] = self.cvar()
        results['tail_dependence_coef'] = self.tail_dependence_coefficient()
        results['return_level_dist'] = self.return_level()
        self.results = results
        if self.v: print(f"Calculated all metrics in {time.time()-t1:.1f} s")

    def get_uniform_results(self):
        results = {}
        results['cramer_von_mises'] = self.results['cramer_von_mises']
        results['kl_divergence'] = self.results['kl_divergence']
        for i in ['xi','rho']:
            results[f'acf_dist_{i}'] = {k:self.results['acf_dist'][k][i] for k in self.columns}
        results['ccmd'] = {k:self.results['ccmd'] for k in self.columns}
        for c in self.columns:
            for i in ['xi','rho']:
                results[f'ccf_dist_{c}_{i}'] = {k:self.results['ccf_dist'][(c,k)][i] for k in set(self.columns)-set([c])}
                results[f'ccf_dist_{c}_{i}'].update({c:self.results['acf_dist'][c][i]})
                # In these, c series is the original and the rest are lagged and their relationship is stuidied
        results['cvar_pos'] = {c:self.results['cvar'][(c,1)] for c in self.columns}
        results['cvar_neg'] = {c:self.results['cvar'][(c,-1)] for c in self.columns}
        results['tail_dependence_coef_pos'] = {c:self.results['tail_dependence_coef'][(c,1)] for c in self.columns}
        results['tail_dependence_coef_neg'] = {c:self.results['tail_dependence_coef'][(c,-1)] for c in self.columns}
        results['return_level_dist_pos'] = {c:self.results['return_level_dist'][(c,1)] for c in self.columns}
        results['return_level_dist_neg'] = {c:self.results['return_level_dist'][(c,-1)] for c in self.columns}
        self.uniform_results = results

    def get_results_df(self):
        self.get_uniform_results()
        df = pd.DataFrame(columns=self.columns)
        for k in self.uniform_results.keys():
            s = pd.Series(self.uniform_results[k], name=k)
            df = df._append(s)
        self.results_df = df
        return df
    
    def save_test_results(self,name='',path='testing'):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(path,f'test_{self.testing_version}_{name}_{timestamp}.pkl')
        with open(file_name,'wb') as f:
            pkl.dump(self.results,f)
        return
    
    def load_test_results(self,file):
        with open(file,'rb') as f:
            results = pkl.load(f)
        self.results = results
        return
        
    def cramer_von_mises(self):
        results = {}
        for c in self.columns:
            cdf = ecdf(self.df_obs[c])
            cdf = cdf.cdf
            res = cramervonmises(self.df_pred[c], cdf.evaluate)
            results[c] = res.statistic
        if self.v: print("Calculated cramer von mises")
        return results
    
    def kl_divergence(self, grid_samples=5000):
        results = {}
        for c in self.columns:
            kde_observed = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(self.df_obs[c].values.reshape(-1, 1))
            kde_predicted = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(self.df_pred[c].values.reshape(-1, 1))
            common_grid = np.linspace(min(self.df_obs[c].min(), self.df_pred[c].min()),
                                    max(self.df_obs[c].max(), self.df_pred[c].max()), grid_samples).reshape(-1, 1)

            log_density_observed = kde_observed.score_samples(common_grid)
            log_density_predicted = kde_predicted.score_samples(common_grid)
            density_observed = np.exp(log_density_observed)
            density_predicted = np.exp(log_density_predicted)
            density_predicted += 1e-10
            kl_divergence_kde = entropy(density_observed, density_predicted)
            results[c]=kl_divergence_kde
        
        if self.v: print("Calculated KL divergence")
        return results
    
    def _new_corr_coef(self,df, x='x', y='y'):
        n = len(df)
        df = df.sort_values(by=x, ascending=True)
        rank = df[y].rank(method='min')
        coef = 1 - 3*sum(abs(rank-rank.shift(1))[1:])/(n**2-1)
        return coef

    def _calc_lcf(self,series_org, series_lag, n_lags=24*3):
        assert len(series_org) == len(series_lag)
        rhos = []
        xis = []
        for lag in range(n_lags + 1):
            shifted_series = series_lag.shift(lag)
            rho = series_org.corr(shifted_series)
            df = pd.DataFrame({'x':shifted_series, 'y':series_org})
            df = df.dropna()
            xi = self._new_corr_coef(df)
            rhos.append(rho)
            xis.append(xi)
        return rhos, xis

    def _calc_cf_dist(self,cf_obs,cf_pred):
        cf_obs = np.array(cf_obs)
        cf_pred = np.array(cf_pred)
        w = np.abs(cf_obs)
        return np.sqrt((w*(cf_obs-cf_pred)**2).sum()/w.sum())

    def acf_dist(self):
        results = {}
        # lcs structure: {series: ((xi_obs, xi_pred),(rho_obs,rho_pred))}
        for c in self.columns:
            lc_rhos_pred, lc_xis_pred = self._calc_lcf(self.df_pred[c],self.df_pred[c])
            lc_rhos_obs, lc_xis_obs = self._calc_lcf(self.df_obs[c],self.df_obs[c])
            acfd_xi = self._calc_cf_dist(lc_xis_obs,lc_xis_pred)
            acfd_rho = self._calc_cf_dist(lc_rhos_obs, lc_rhos_pred)
            results[c] = {'xi':acfd_xi,'rho':acfd_rho}
        if self.v: print("Calculated acf distance")
        return results
    
    def _calc_mat_dist(self,m1,m2):
        n = len(m1)
        res = 0
        for i in range(1,n):
            for j in range(0,n-1):
                res += (m1[i][j] - m2[i][j])**2
        res /= (n*(n-1)/2)
        return res**.5

    def ccmd(self):
        dist_obs = GaussianMultivariate(distribution={
            'solar_pv':GaussianKDE,
            'solar_th':GaussianKDE,
            'wind':GaussianKDE,
        })
        dist_obs.fit(self.df_obs.sample(7500,random_state=42))
        params_obs = dist_obs.to_dict()
        dist_pred = GaussianMultivariate(distribution={
            'solar_pv':GaussianKDE,
            'solar_th':GaussianKDE,
            'wind':GaussianKDE,
        })
        dist_pred.fit(self.df_pred.sample(7500,random_state=42))
        params_pred = dist_pred.to_dict()
        ccmd = self._calc_mat_dist(params_obs['correlation'],params_pred['correlation'])
        if self.v: print("Calculated ccmd")
        return ccmd
    
    def ccf_dist(self):
        lcs = {}
        for c1 in self.columns:
            for c2 in self.columns:
                if c1==c2: continue
                lc_rhos_pred, lc_xis_pred = self._calc_lcf(self.df_pred[c1],self.df_pred[c2])
                lc_rhos_obs, lc_xis_obs = self._calc_lcf(self.df_obs[c1],self.df_obs[c2])
                acfd_xi = self._calc_cf_dist(lc_xis_obs,lc_xis_pred)
                acfd_rho = self._calc_cf_dist(lc_rhos_obs, lc_rhos_pred)
                lcs[(c1,c2)] = {'xi':acfd_xi,'rho':acfd_rho}
        if self.v: print("Calculated ccf distance")
        return lcs

    def _calc_cvar(self,values,level=.95,sign=1):
        """
        values: pd.Series with values
        level: The percentage in per1 basis from which the cvar is to be calculated
        sign: 1 or -1. Whether the cvar is to be calculated from the level up or down
        """
        if sign == -1:
            level = 1 - level
        var = np.percentile(values, 100 * level)
        if sign == -1:
            tail_losses = values[values <= var]
        elif sign == 1:
            tail_losses = values[values >= var]
        else:
            raise ValueError("Sign must be either 1 or -1")
        cvar = tail_losses.mean()
        return cvar
    
    def cvar(self):
        results = {}
        for c in self.columns:
            for sign in [-1,1]:
                cvar_pred = self._calc_cvar(self.df_pred[c],sign=sign)
                cvar_obs = self._calc_cvar(self.df_obs[c],sign=sign)
                results[(c,sign)] = cvar_pred/cvar_obs - 1
        if self.v: print("Calculated cvar")
        return results

    def _calc_tdc(self, series1, series2, tail=1, q=0.05):
        if len(series1) != len(series2):
            raise ValueError(f"Length of series 1 and series 2 must be the same, currently it is {len(series1)} and {len(series2)}")
        # Convert to ranks for empirical copula
        u = rankdata(series1) / (len(series1) + 1)
        v = rankdata(series2) / (len(series2) + 1)
        # Set the threshold based on the quantile
        if tail == 1:
            threshold_u = np.quantile(u, 1 - q)
            threshold_v = np.quantile(v, 1 - q)
        elif tail == -1:
            threshold_u = np.quantile(u, q)
            threshold_v = np.quantile(v, q)
        else:
            raise ValueError("Tail must be either '1' or '-1'.")
        
        # Count how many observations are above the thresholds
        if tail == 1:
            extreme_u = (u >= threshold_u)
            extreme_v = (v >= threshold_v)
        else:
            extreme_u = (u <= threshold_u)
            extreme_v = (v <= threshold_v)

        # Estimate the tail dependence coefficient
        lambda_tail = np.sum(extreme_u & extreme_v)/np.sum(extreme_v)
        
        return lambda_tail

    def tail_dependence_coefficient(self):
        results = {}
        for c in self.columns:
            l_up = self._calc_tdc(self.df_obs[c],self.df_pred[c],tail=1,q=.05)
            l_down = self._calc_tdc(self.df_obs[c],self.df_pred[c],tail=-1,q=.05)
            results[(c,1)] = l_up
            results[(c,-1)] = l_down
        if self.v: print("Calculated tail dependence coefficient")
        return results
    
    def __calc_rl(self,mu,std,xi,T=10*52):
        return mu+std/xi*((-np.log(1-1/T))**(-xi)-1)

    def _calc_rld(self,obs,pred,sign=1):
        if sign==1:
            obs_max = obs.resample('w').max()
            pred_max = pred.resample('w').max()
        elif sign==-1:
            obs_max = obs.resample('w').min()
            pred_max = pred.resample('w').min()
        else:
            raise ValueError("sign must be either 1 or -1")
        obs_params = genextreme.fit(obs_max)
        pred_params = genextreme.fit(pred_max)
        obs_rl = self.__calc_rl(*obs_params)
        pred_rl = self.__calc_rl(*pred_params)
        return pred_rl/obs_rl - 1
    
    def return_level(self):
        results = {}
        for c in self.columns:
            for sign in [1,-1]:
                res = self._calc_rld(self.df_obs[c],self.df_pred[c],sign)
                results[(c,sign)] = res
        if self.v: print("Calculated return level distance")
        return results
    

if __name__=='__main__':
    def read_dataset(name='data/factor_capacidad.csv'):
        df = pd.read_csv(name,index_col=0,parse_dates=True)
        return df
    df = read_dataset()
    df_pred = df.iloc[-24*365*2:-24*365]
    df_obs = df.iloc[-24*365*3:-24*365*2]
    df_obs.index = df_pred.index
    test = Test(df_obs,df_pred,verbose=True)
    res_df = test.get_results_df()
    test.save_test_results('test_set')
    print(res_df)


    # import pickle as pkl
    # with open('testing/xgboost_multi_2024-10-04_20-55-34.pkl','rb') as f:
    #     pred_results = pkl.load(f)
    # combined_obs = pred_results['results']['train']['solar_pv'].merge(pred_results['results']['train']['solar_th'],left_index=True,right_index=True).merge(pred_results['results']['train']['wind'],left_index=True,right_index=True)
    # combined_pred = pred_results['results']['test']['solar_pv'].merge(pred_results['results']['test']['solar_th'],left_index=True,right_index=True).merge(pred_results['results']['test']['wind'],left_index=True,right_index=True)
    # test = Test(combined_obs,combined_pred,verbose=True)
    # res_df = test.get_results_df()
    # print(res_df)