import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.stattools import jarque_bera
from scipy.interpolate import CubicHermiteSpline

class SERA():
    def __init__(self, y_true, y_pred, dt=.001, min_p=5, max_p=10):
        self.y_true = y_true.to_numpy()
        self.y_pred = y_pred.to_numpy()
        self.dt = dt
        self.phis = self.calc_phis(min_p, max_p)

    def calc_phis(self, min_p,max_p):
        mini = np.min(self.y_true)-1e-3
        maxi = np.max(self.y_true)+1e-3
        semi1 = np.percentile(self.y_true, min_p)
        semi2 = np.percentile(self.y_true, max_p)
        pchip = CubicHermiteSpline([mini,semi1,semi2,maxi],[1,1,0,0],[0,0,0,0])
        pchip_v = np.vectorize(pchip)
        return pchip_v(self.y_true) # We reutrn the phi(y_i) for every y_i

    def calc_SER(self, t):
        I = self.phis >= t
        return ((self.y_true-self.y_pred)**2)@I.T
    
    def calc_SERA(self):
        ts = np.linspace(0,1,int(1/self.dt))
        sers = np.vectorize(self.calc_SER)(ts)
        return sers.sum()*self.dt


def test_results(y_true_complete, y_pred_complete, disp_error=True, disp_resid=False, test_horizon=None, min_p=5,max_p=10):
    """
    y_true: pd.Series()
    y_pred: pd.Series()
    """
    if test_horizon:
        y_true = y_true_complete[-test_horizon:].copy()
        y_pred = y_pred_complete[-test_horizon:].copy()
    else:
        y_true = y_true_complete
        y_pred = y_pred_complete

    residuals = y_true - y_pred

    rmse = np.sqrt(((y_true - y_pred)**2).mean())
    mae = np.abs(y_true - y_pred).mean()
    bias = (y_true - y_pred).mean()
    s = SERA(y_true, y_pred, min_p=min_p, max_p=max_p)
    sera = s.calc_SERA()
    corr = np.corrcoef(y_true, y_pred)
    corr = corr[0,1]
    df = pd.DataFrame(data={'true':y_true,'pred':y_pred})
    df.sort_values(by='true',ascending=True)
    min5 = np.percentile(y_true,5)
    df = df.loc[y_true<=min5,:]
    rmse95 = ((df['true']-df['pred'])**2).mean()**.5

    lm, lmpvalue, fstatistic, fpvalue = het_white(residuals, pd.DataFrame({'x':y_true, 'const':np.ones(len(y_true))}))
    dw = durbin_watson(residuals)
    jb, jbpvalue, _, _ = jarque_bera(residuals)

    if disp_error:
        print(f'--- FORECASTING ACCURACY ---')
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"SERA [{min_p}%, {max_p}%]: {sera:.4f}")
        print(f"Bias: {bias:.4f}")
        print(f"Correlation: {corr:.4f}")
        print(f"RMSE at lower 5%: {rmse95:.4f}")
    if disp_resid:
        print("--- RESIDUALS ANALYSIS ---")
        print(f"White test p-values: {lmpvalue:.3f}, {fpvalue:.3f} (heteroskedasticity of results)")
        print(f"Durbin-Watson statistic: {dw:.4f} (serial correlation of residuals 0-4)")
        print(f"Jarque-Berra p-values: {jbpvalue:.4f} (normality of residuals)")
    
    return rmse, mae, corr, sera, bias, rmse95



def test_results_multivariate(y_true_complete, y_pred_complete, disp_results=True, test_horizon=None):
    """
    y_true: pd.DataFrame()
    y_pred: pd.DataFrame()
    """
    for u in pd.unique(y_true_complete['unique_id']):
        print(f"Results for '{u}'")
        y_true = y_true_complete[y_true_complete['unique_id']==u]['y']
        y_pred = y_pred_complete[y_pred_complete['unique_id']==u]['y']
        test_results(y_true,y_pred,disp_results=disp_results,test_horizon=test_horizon)
        # residuals = y_true - y_pred

        # rmse = np.sqrt(((y_true - y_pred)**2).mean())
        # mae = np.abs(y_true - y_pred).mean()
        # bias = (y_true - y_pred).mean()
        # s = SERA(y_true, y_pred)
        # sera = s.calc_SERA()
        # corr = np.corrcoef(y_true, y_pred)
        # corr = corr[0,1]

        # lm, lmpvalue, fstatistic, fpvalue = het_white(residuals, pd.DataFrame({'x':y_true, 'const':np.ones(len(y_true))}))
        # dw = durbin_watson(residuals)
        # jb, jbpvalue, _, _ = jarque_bera(residuals)

        # if disp_results:
        #     print(f'--- FORECASTING ACCURACY ---')
        #     print(f"RMSE: {rmse:.4f}")
        #     print(f"MAE:  {mae:.4f}")
        #     print(f"SERA {{(y-.5)**3}}:  {sera:.4f}")
        #     print(f"Bias: {bias:.4f}")
        #     print(f"Correlation: {corr:.4f}")
        #     print("--- RESIDUALS ANALYSIS ---")
        #     print(f"White test p-values: {lmpvalue:.3f}, {fpvalue:.3f} (heteroskedasticity of results)")
        #     print(f"Durbin-Watson statistic: {dw:.4f} (serial correlation of residuals 0-4)")
        #     print(f"Jarque-Berra p-values: {jbpvalue:.4f} (normality of residuals)")
    
    return