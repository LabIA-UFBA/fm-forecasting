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

log.add("mlp_short_time.log")
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

def create_windows(data, input_size):
    X, y = [], []
    for i in range(len(data) - input_size):
        X.append(data[i:i+input_size])
        y.append(data[i+input_size])

    return np.array(X), np.array(y)


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)
    
#
# parametros
# 

nodes = sb.columns

scores_error = {'node': [], 'mae': [], 'mse': [], 'r2': [], 'mape': []}

total_points = sb.shape[0]
t = np.arange(total_points)

limite_marco = 31 * 40
limite_abril = limite_marco + (30 * 40)

future_steps = sb.shape[0] - limite_abril

input_size = 40  # tamanho da janela de entrada
train_points = limite_abril  # 1 mês
future_steps = future_steps  
device = 'cuda' if torch.cuda.is_available() else 'cpu'
targets = {}


for node in tqdm(nodes):

    serie = sb[node]
 
    X_train, y_train = create_windows(serie[:train_points], input_size)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

    # model
    model = MLP(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    log.info("train")
    epochs = 500
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    log.info("forecasting....")
    # forecasting
    model.eval()
    history = list(serie[:train_points])  # Começa com o primeiro mês
    predictions = []

    for _ in range(future_steps):
        x_input = torch.tensor(history[-input_size:], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x_input).item()
        predictions.append(pred)
        history.append(pred)  # feedback do próprio modelo
    

    y_true = sb[node][limite_abril:].values
    y_pred = predictions

    scores_error['node'].append(node)
    scores_error['mse'].append(mean_squared_error(y_true, y_pred))
    scores_error['mae'].append(mean_absolute_error(y_true, y_pred))
    scores_error['r2'].append(r2_score(y_true, y_pred))
    scores_error['mape'].append(mean_absolute_percentage_error(y_true, y_pred))

    targets[node] = {"input":  X_train, 
                     'true': y_true,
                     'pred': y_pred}


with open(f'results/mlp-short-time-targets.pkl', 'wb') as f:
    pickle.dump(targets, f)

df_results = pd.DataFrame(scores_error)
df_results["model"] = "MLP"
df_results.to_parquet(f"results/MLP-short-time.parquet", index=False)

log.info("Done.")