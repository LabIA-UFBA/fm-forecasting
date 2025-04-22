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

#from chronos import ChronosPipeline
from chronos import BaseChronosPipeline
from chronos import ChronosPipeline, ChronosBoltPipeline


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


# model_name = "chronos-t5-small"
# pipeline = BaseChronosPipeline.from_pretrained(
#     f"amazon/{model_name}",  # use "amazon/chronos-bolt-small" for the corresponding Chronos-Bolt model
#     device_map="cuda",  # use "cpu" for CPU inference
#     torch_dtype=torch.bfloat16,
# )

model_name = "chronos-t5-base"
pipeline = ChronosPipeline.from_pretrained(
    f"amazon/{model_name}",
    device_map="cuda",  # use "cpu" for CPU inference and "mps" for Apple Silicon
    torch_dtype=torch.bfloat16,
)


scores_error = {'node': [], 'batch': [], 'mae': [], 'mse': [], 'r2': [], 'mape': []}
targets = {}
cost, time = 0, 0
dfs = []

for node in tqdm(nodes):
    dfs_pred = []   
    targets[node] = []
    for time, snapshot in enumerate(test_dataset):
        snapshot.to('cpu')

        #
        # (alterar aqui o modelo)
        #
        forecast = pipeline.predict(context=torch.tensor(snapshot.x[node,:]),
                                    prediction_length=inference_size,
                                    limit_prediction_length=False,
                                    num_samples=1)
        #
        y_hat = forecast.view(-1)
        #
        
        quantiles, mean = pipeline.predict_quantiles(
            context=torch.tensor(snapshot.x[node,:]),
            prediction_length=inference_size,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        
        # nao alterar mais abaixo

        cost = cost + torch.mean((y_hat-snapshot.y)**2)
        y_true = snapshot.y.cpu().data.numpy()
        y_pred = y_hat.cpu().data.numpy()
        
        scores_error['node'].append(stops[node])
        scores_error['batch'].append(time)
        scores_error['mse'].append(mean_squared_error(y_true[node,:], y_pred))
        scores_error['mae'].append(mean_absolute_error(y_true[node,:], y_pred))
        scores_error['r2'].append(r2_score(y_true[node,:], y_pred))
        scores_error['mape'].append(mean_absolute_percentage_error(y_true[node,:], y_pred))
        
        targets[node].append({"input": snapshot.x[node,:].cpu().numpy(), 
                              'true': snapshot.y[node,:].cpu().data.numpy(),
                              'pred': y_pred,
                              'node': stops[node], 
                              'quantiles': quantiles,
                              'mean': mean,
                             })
        
    cost = cost / (time+1)
    cost = cost.item()
    log.info(f"node: {node} MSE test: {cost:.4f}")


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
df_results.to_parquet(f"results/cronos-{model_name}-batch.parquet", index=False)