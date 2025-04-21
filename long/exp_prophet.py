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
from prophet import Prophet

log.add("Prophet.log")

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

for node in tqdm(nodes):

    log.info(f"run node {node}")

    serie = sbx[node].reset_index()
    serie.columns = ['ds', 'y']  # formato exigido pelo Prophet

    modelo = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive'  # ou 'multiplicative' se necessário
    )

    modelo.fit(serie)

    # criar datas futuras para previsão
    future = modelo.make_future_dataframe(periods=pred_len, freq="30min")  # ajusta freq se necessário
    forecast = modelo.predict(future)

    y_pred = forecast['yhat'].iloc[-pred_len:].values
    y_true = sby[node].values

    scores_error['node'].append(node)
    scores_error['mse'].append(mean_squared_error(y_true, y_pred))
    scores_error['mae'].append(mean_absolute_error(y_true, y_pred))
    scores_error['r2'].append(r2_score(y_true, y_pred))
    scores_error['mape'].append(mean_absolute_percentage_error(y_true, y_pred))

df_results = pd.DataFrame(scores_error)
df_results["model"] = "Prophet"
df_results.to_parquet(f"results/PROPHET-long-time.parquet", index=False)

log.info("Done Prophet.")