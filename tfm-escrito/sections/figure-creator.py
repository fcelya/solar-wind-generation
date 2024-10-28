import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import os

BASE_PATH = os.path.abspath(os.path.join(__file__,'..','..','..'))
DATA_PATH = os.path.join(BASE_PATH,'testing')
SAVE_PATH = os.path.join(BASE_PATH,'tfm-escrito','assets')

with open(os.path.join(DATA_PATH,'nhits_univariate_2024-10-11_12-37-56.pkl'),'rb') as f:
# with open(os.path.join(DATA_PATH,'sarimax_2024-10-14_03-25-38.pkl'),'rb') as f:
    data_base = pkl.load(f)
    df_pred_base = data_base['prediction']
    df_obs_base = data_base['experiment']['obs']

fig, ax = plt.subplots(2)

START_LOC_W = 24*30*7
LEN_W = 24*7
START_LOC_Y = 0
LEN_Y = 24*365

index = df_pred_base.index[START_LOC_W:START_LOC_W+LEN_W]
ax[0].plot(index,df_pred_base.loc[index,'solar_pv'],label='Modeled',alpha=.7)
ax[0].plot(index,df_obs_base.loc[index,'solar_pv'],label='Observed',alpha=.7)
ax[0].legend()
ax[0].tick_params(axis='x',rotation=30)

index = df_pred_base.index[START_LOC_Y:START_LOC_Y+LEN_Y]
ax[1].plot(index,df_pred_base.loc[index,'solar_pv'],label='Modeled',alpha=.7)
ax[1].plot(index,df_obs_base.loc[index,'solar_pv'],label='Observed',alpha=.7)
ax[1].legend()
ax[1].tick_params(axis='x',rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH,'nhits-solar-pv.png'), bbox_inches='tight')


with open(os.path.join(DATA_PATH,'itransformer_multivariate_2024-10-24_14-11-54.pkl'),'rb') as f:
    data_base = pkl.load(f)
    df_pred_base = data_base['prediction']
    df_obs_base = data_base['experiment']['obs']


fig, ax = plt.subplots(2)

index = df_pred_base.index[START_LOC_W:START_LOC_W+LEN_W]
ax[0].plot(index,df_pred_base.loc[index,'solar_th'],label='Modeled',alpha=.7)
ax[0].plot(index,df_obs_base.loc[index,'solar_th'],label='Observed',alpha=.7)
ax[0].legend()
ax[0].tick_params(axis='x',rotation=30)

index = df_pred_base.index[START_LOC_Y:START_LOC_Y+LEN_Y]
ax[1].plot(index,df_pred_base.loc[index,'solar_th'],label='Modeled',alpha=.7)
ax[1].plot(index,df_obs_base.loc[index,'solar_th'],label='Observed',alpha=.7)
ax[1].legend()
ax[1].tick_params(axis='x',rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH,'itransformer-solar-th.png'), bbox_inches='tight')



with open(os.path.join(DATA_PATH,'varmax_2024-10-27_21-23-27.pkl'),'rb') as f:
    data_base = pkl.load(f)
    df_pred_base = data_base['prediction']
    df_obs_base = data_base['experiment']['obs']


fig, ax = plt.subplots(2)

index = df_pred_base.index[START_LOC_W:START_LOC_W+LEN_W]
ax[0].plot(index,df_pred_base.loc[index,'wind'],label='Modeled',alpha=.7)
ax[0].plot(index,df_obs_base.loc[index,'wind'],label='Observed',alpha=.7)
ax[0].legend()
ax[0].tick_params(axis='x',rotation=30)

index = df_pred_base.index[START_LOC_Y:START_LOC_Y+LEN_Y]
ax[1].plot(index,df_pred_base.loc[index,'wind'],label='Modeled',alpha=.7)
ax[1].plot(index,df_obs_base.loc[index,'wind'],label='Observed',alpha=.7)
ax[1].legend()
ax[1].tick_params(axis='x',rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH,'varmax-wind.png'), bbox_inches='tight')


with open(os.path.join(DATA_PATH,'nbeats_univariate_2024-10-11_22-37-32.pkl'),'rb') as f:
    data_base_2 = pkl.load(f)
    df_pred_base_2 = data_base_2['prediction']
    df_obs_base_2 = data_base_2['experiment']['obs']


fig, ax = plt.subplots(2)

index = df_pred_base.index[START_LOC_W:START_LOC_W+LEN_W]
ax[0].plot(index,df_pred_base.loc[index,'wind'],label='Modeled VARMAX',alpha=.7)
ax[0].plot(index,df_pred_base_2.loc[index,'wind'],label='Modeled N-BEATS',alpha=.7)
ax[0].plot(index,df_obs_base.loc[index,'wind'],label='Observed',alpha=.7)
ax[0].legend()
ax[0].tick_params(axis='x',rotation=30)

index = df_pred_base.index[START_LOC_Y:START_LOC_Y+LEN_Y]
ax[1].plot(index,df_pred_base.loc[index,'wind'],label='Modeled VARMAX',alpha=.7)
ax[1].plot(index,df_pred_base_2.loc[index,'wind'],label='Modeled N-BEATS',alpha=.7)
ax[1].plot(index,df_obs_base.loc[index,'wind'],label='Observed',alpha=.7)
ax[1].legend()
ax[1].tick_params(axis='x',rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH,'varmax-nbeats-wind.png'), bbox_inches='tight')
pass