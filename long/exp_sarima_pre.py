# basic
import os
import pickle
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
# NN
import torch
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
from statsmodels.tsa.statespace.sarimax import SARIMAX
model_name = "SARIMA"
log.add("sarima_long_time.log")

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

sbx = sb.query("index <= '2024-03-31 23:59:59'")
sby = sb.query("index > '2024-03-31 23:59:59'")
pred_len = abs(sbx.shape[0] - sb.shape[0])
nodes = sb.columns

scores_error = {'node': [], 'mae': [], 'mse': [], 'r2': [], 'mape': []}
targets ={}

for node in tqdm(nodes):

    log.info(f"run node {node}")

    serie_train = sbx[node]

    # save results
    with open(f"sarima_best_params/{node}-grid-search-results.pkl", "rb") as f:
        modelo_auto = pickle.load(f)
    
    modelo = SARIMAX(serie_train,
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


    targets[node] = {"input":  serie_train, 
                     'true': y_true,
                     'pred': y_pred}
    
with open(f'results/{model_name}-long-time-targets.pkl', 'wb') as f:
    pickle.dump(targets, f)

df_results = pd.DataFrame(scores_error)
df_results["model"] = "SARIMA"
df_results.to_parquet(f"results/SARIMA-long-time.parquet", index=False)

log.info("Done.")