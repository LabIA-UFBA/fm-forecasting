# basic
import os
import pickle
import warnings
import traceback
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
#from torchmetrics.regression import R2Score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from loguru import logger as log
from val import calculate_metrics
# plot
import matplotlib.pyplot as plt
# foundation model
from functools import reduce
from prophet import Prophet

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


def load_datasets(filepath):
    """Carrega os datasets de arquivos pickle."""
    try:
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        return dataset
    except IOError as e:
        log.error(f"Erro ao carregar o dataset: {e}")
    except pickle.PickleError as e:
        log.error(f"Erro ao desserializar o dataset: {e}")
        traceback.print_exception(e)


model_name = "prophet"
#
# input parameters long
#

sb = pd.read_parquet("/home/marcos/loader_03-04_2024.parquet")
sbx = sb.query("index <= '2024-03-31 23:59:59'")
sby = sb.query("index > '2024-03-31 23:59:59'")


#
# Input parameters
#
fpath_root = "/mnt/data/marcos/data/node_regression_bus/data_split/"
test_dataset = load_datasets(f'{fpath_root}test.pkl')
inference_size = test_dataset[0].y.shape[1]

nodes = [53,  365,  382,  666,  701, 1326, 1404, 1569, 1916, 2617]

stops = {53: '125960550',
         365: '230565994',
         382: '258781031',
         666: '43768720',
         701: '44072192',
         1326: '44783654',
         1404: '44783914',
         1569: '44784438',
         1916: '45833547',
         2617: '47568123'}




scores_error = {'node': [], 'batch': [], 'mae': [], 'mse': [], 'r2': [], 'mape': []}
targets = {}
cost, time = 0, 0
dfs = []

sazonalidade = 7 * 40

for node in tqdm(nodes):

    dfs_pred = []   
    targets[node] = []

    log.info("Fit prophet model...")

    #
    # fit model
    #
    serie = sbx[stops[node]].reset_index()
    serie.columns = ['ds', 'y']  # formato Prophet
    serie['cap'] = serie['y'].max() * 1.2  
    serie['floor'] = 0   


    modelo = Prophet(growth='logistic')
    modelo.fit(serie)


    log.info("Forecasting prophet model...")

    for time, snapshot in enumerate(test_dataset):
        snapshot.to('cpu')

        y_true = snapshot.y[node,:].cpu().data.numpy()
        #
        # forecasting
        #

        future = modelo.make_future_dataframe(periods=y_true.shape[0], freq='30min')
        future['cap'] = serie['cap'].iloc[0]  # mesmo cap usado no treino
        future['floor'] = 0

        forecast = modelo.predict(future)
        y_pred = forecast['yhat'].iloc[-y_true.shape[0]:].values

  
        
        
        scores_error['node'].append(stops[node])
        scores_error['batch'].append(time)
        scores_error['mse'].append(mean_squared_error(y_true, y_pred))
        scores_error['mae'].append(mean_absolute_error(y_true, y_pred))
        scores_error['r2'].append(r2_score(y_true, y_pred))
        scores_error['mape'].append(mean_absolute_percentage_error(y_true, y_pred))
        
        targets[node].append({"input":  snapshot.x[node,:].cpu().numpy(), 
                              'true': y_true,
                              'pred': y_pred,
                              'node': stops[node]
                             })

log.info("save forecasting....")
#
# save forecastings
#
with open(f'results/{model_name}-targets.pkl', 'wb') as f:
    pickle.dump(targets, f)


log.info("save scores....")
#
# save scores
#
df_results = pd.DataFrame(scores_error)
df_results["model"] = model_name
df_results.to_parquet(f"results/{model_name}-batch.parquet", index=False)