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
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
model_name = "ETS"


log.add(f"log/{model_name}.log")

plt.style.use("seaborn-v0_8-whitegrid")

SEED = 1345
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(SEED)
warnings.filterwarnings('ignore')

sb = pd.read_parquet("/home/marcos/loader_03-04_2024.parquet")

sbx = sb.query("index <= '2024-03-31 23:59:59'")
sby = sb.query("index > '2024-03-31 23:59:59'")
pred_len = abs(sbx.shape[0] - sb.shape[0])
nodes = sb.columns

scores_error = {'node': [], 'mae': [], 'mse': [], 'r2': [], 'mape': []}
forecastings = {}
sazonalidade =  7 * 40

for node in tqdm(nodes):

    log.info(f"run node {node}")

    serie_train = sbx[node]
    serie_test = sby[node]

    modelo = ExponentialSmoothing(
        serie_train,
        #trend='add',                # ou 'mul' se for melhor
        seasonal='add',             # ou 'mul'
        seasonal_periods=sazonalidade
    )

    resultado = modelo.fit()

    y_pred = resultado.forecast(steps=pred_len)
    y_true = serie_test.values

    forecastings[f"pred_{node}"] = y_pred
    forecastings[f"true_{node}"] = y_true

    scores_error['node'].append(node)
    scores_error['mse'].append(mean_squared_error(y_true, y_pred))
    scores_error['mae'].append(mean_absolute_error(y_true, y_pred))
    scores_error['r2'].append(r2_score(y_true, y_pred))
    scores_error['mape'].append(mean_absolute_percentage_error(y_true, y_pred))

df_results = pd.DataFrame(scores_error)
df_results["model"] = model_name
df_results.to_parquet(f"results/{model_name}-long-time.parquet", index=False)

df_forecastings = pd.DataFrame(forecastings)
df_forecastings.to_parquet(f"results/forecasting-{model_name}-long-time.parquet", index=False)

log.info(f"Done {model_name}.")