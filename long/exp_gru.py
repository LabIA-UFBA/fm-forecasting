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
from chronos import ChronosPipeline

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


def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])

    return np.array(X), np.array(y)


class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, h = self.gru(x)  # h: (1, batch, hidden)
        out = self.fc(h.squeeze(0))  # (batch, hidden) -> (batch, 1)
        return out
    
#
# parametros
# 

nodes = sb.columns

scores_error = {'node': [], 'mae': [], 'mse': [], 'r2': [], 'mape': []}

total_points = sb.shape[0]
t = np.arange(total_points)

limite_marco = 31 * 40
limite_abril = limite_marco + (30 * 40)

future_steps = sb.shape[0] - limite_marco

# model params

input_size = 1       # entrada por passo de tempo
hidden_size = 32     # tamanho do hidden state da GRU
seq_len = 40         # número de passos anteriores usados para prever
train_points = limite_marco  # 1 mês
future_steps = future_steps  # prever 2 meses
device = 'cuda' if torch.cuda.is_available() else 'cpu'


for node in tqdm(nodes):

    serie = sb[node]
 
    # data
    X_train, y_train = create_sequences(serie[:train_points], seq_len)

    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(device)  # (batch, seq_len, 1)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)   # (batch, 1)

    # model
    model = GRUNet(input_size=1, hidden_size=hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    log.info("train....")
    epochs = 500
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    log.info("forecasting....")
    model.eval()
    history = list(serie[:train_points])
    predictions = []

    for _ in range(future_steps):
        seq_input = torch.tensor(history[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        with torch.no_grad():
            next_val = model(seq_input).item()
        predictions.append(next_val)
        history.append(next_val)
    

    y_true = sb[node][limite_marco:].values
    y_pred = predictions

    scores_error['node'].append(node)
    scores_error['mse'].append(mean_squared_error(y_true, y_pred))
    scores_error['mae'].append(mean_absolute_error(y_true, y_pred))
    scores_error['r2'].append(r2_score(y_true, y_pred))
    scores_error['mape'].append(mean_absolute_percentage_error(y_true, y_pred))

df_results = pd.DataFrame(scores_error)
df_results["model"] = "GRU"
df_results.to_parquet(f"GRU-long-time.parquet", index=False)

log.info("Done.")