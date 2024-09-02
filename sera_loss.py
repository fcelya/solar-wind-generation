import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss, _WeightedLoss
from neuralforecast.losses.pytorch import BasePointLoss
import numpy as np
from scipy.interpolate import CubicHermiteSpline


class SERALoss(BasePointLoss):
    def __init__(self,horizon_weight=None,dt=.001, exp=12):
        super(SERALoss, self).__init__(horizon_weight=horizon_weight, outputsize_multiplier=1, output_names=[""])
        self.dt = dt
        self.exp=exp

    def calc_phis(self, targets):
        self.phis = targets.detach().clone()
        self.phis -= 1
        self.phis = self.phis**self.exp
        mmax = torch.max(self.phis)
        mmin = torch.min(self.phis)
        self.phis -= mmin
        self.phis /= (mmax-mmin)
    
    def calc_SER(self, t):
        mask = self.phis > t
        y_true = self.targets[mask]
        y_pred = self.predictions[mask]
        return ((y_true - y_pred)**2).sum()
    
    def forward(self, predictions, targets):
        self.calc_phis(targets)
        self.targets = targets
        self.predictions = predictions
        ts = torch.arange(0, 1, self.dt)
        batched_calc_SER = torch.vmap(torch.vectorize)
        sers = batched_calc_SER(ts)
        return sers.sum() * self.dt
    
class SERALossXGBoost(object):
    def __init__(self,y_true,min_p=5,max_p=10):
        self.min_p=min_p
        self.max_p=max_p
        mini = np.min(y_true)-1e-3
        maxi = np.max(y_true)+1e-3
        semi1 = np.percentile(y_true, self.min_p)
        semi2 = np.percentile(y_true, self.max_p)
        pchip = CubicHermiteSpline([mini,semi1,semi2,maxi],[1,1,0,0],[0,0,0,0])
        pchip_v = np.vectorize(pchip)
        self.phis = pchip_v(y_true)

    def sera_loss(self,y_true, y_pred):
        grad = 2*(y_pred - y_true)*self.phis
        hess = 2*self.phis
        
        return grad, hess


# class SERALoss(_Loss):
#     def __init__(self, dt=.001, exp=10, size_average=None, reduce=None, reduction: str = 'mean'):
#         super(SERALoss, self).__init__(size_average, reduce, reduction)
#         self.dt = dt
#         if exp%2!=0:
#             exp+=1
#         self.exp=exp

#     def calc_phis(self, targets):
#         self.phis = targets.detach().clone()
#         self.phis -= 1
#         self.phis = self.phis**self.exp
#         mmax = torch.max(self.phis)
#         mmin = torch.min(self.phis)
#         self.phis -= mmin
#         self.phis /= (mmax-mmin)
    
#     def calc_SER(self, t):
#         mask = self.phis > t
#         y_true = self.targets[mask]
#         y_pred = self.predictions[mask]
#         return ((y_true - y_pred)**2).sum()
    
#     def forward(self, predictions, targets):
#         self.calc_phis(targets)
#         self.targets = targets
#         self.predictions = predictions
#         ts = torch.arange(0, 1, self.dt)
#         batched_calc_SER = torch.vmap(torch.vectorize)
#         sers = batched_calc_SER(ts)
#         return sers.sum() * self.dt