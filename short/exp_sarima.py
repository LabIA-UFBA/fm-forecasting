# basic
import os
import pickle
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
# pre processing
from sklearn import preprocessing as pre
# NN
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import MSELoss
from torch_geometric.nn import GCNConv
# val and plot
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from loguru import logger as log
#from ..val import calculate_metrics
# plot
import matplotlib.pyplot as plt
# foundation model
from functools import reduce
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX

log.add("sarima_short_time.log")

import pmdarima as pm
plt.style.use("seaborn-v0_8-whitegrid")

SEED = 1345
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(SEED)
#plt.style.use('seaborn-whitegrid')
#pd.set_option('display.float_format', '{:.16f}'.format)
warnings.filterwarnings('ignore')

sb = pd.read_parquet("/home/marcos/loader_03-04_2024.parquet")

sbx = sb.query("index <= '2024-04-30 23:59:59'")
sby = sb.query("index  > '2024-04-30 23:59:59'")

pred_len = abs(sbx.shape[0] - sb.shape[0])

nodes = sb.columns

scores_error = {'node': [], 'mae': [], 'mse': [], 'r2': [], 'mape': []}

for node in tqdm(nodes):

    serie = sbx[node]

    log.info(f"run node {node} size serie: {serie.shape} pred_len: {pred_len}")

    modelo_auto = pm.auto_arima(
        serie,
        start_p=0, max_p=3,
        start_q=0, max_q=3,
        d=None,            # auto determina o melhor d
        seasonal=True,
        start_P=0, max_P=2,
        start_Q=0, max_Q=2,
        D=None,            # auto determina o melhor D
        m=40,              # frequência sazonal, ex: 24 para dados horários
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True      # mais rápido
    )

    log.info("Resultados pos grid search")
    log.info(modelo_auto.summary())

    log.info(f"fit model")
    modelo = SARIMAX(serie,
                    order=modelo_auto.order,
                    seasonal_order=modelo_auto.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)

    resultado = modelo.fit()
    log.info(f"forecasting")
    forecast = resultado.get_forecast(steps=pred_len)
    media_prevista = forecast.predicted_mean

    y_true = sby[node].values
    y_pred = media_prevista

    scores_error['node'].append(node)
    scores_error['mse'].append(mean_squared_error(y_true, y_pred))
    scores_error['mae'].append(mean_absolute_error(y_true, y_pred))
    scores_error['r2'].append(r2_score(y_true, y_pred))
    scores_error['mape'].append(mean_absolute_percentage_error(y_true, y_pred))

df_results = pd.DataFrame(scores_error)
df_results["model"] = "SARIMA"
df_results.to_parquet(f"results/SARIMA-short-time.parquet", index=False)

log.info("Done.")