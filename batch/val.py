import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

def mape_fix(y_true, y_pred, epsilon=1e-3):
    """
    Calcula o MAPE corrigido para evitar valores menores que 1.

    Args:
        y_true (array-like): Valores reais.
        y_pred (array-like): Valores preditos.
        epsilon (float): Limite mínimo para o denominador.

    Returns:
        float: MAPE corrigido.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Evita divisões por valores muito pequenos
    denominator = np.maximum(np.abs(y_true), epsilon)
    
    # Calcula o erro absoluto percentual
    error = np.abs((y_true - y_pred) / denominator)

    # Retorna o MAPE em percentual
    return np.mean(error) * 100


def calculate_metrics(
    true, 
    pred, 
    verbose=False
) -> dict:
    """
    Calculate and optionally print various regression metrics.

    Parameters:
    true (array-like): Array of true values.
    pred (array-like): Array of predicted values.
    verbose (bool): If True, print the calculated metrics. Default is False.

    Returns:
    tuple: A tuple containing MSE, MAE, RMSE, MAPE, and R2 scores.
    """
    rmse = mean_squared_error(true, pred, squared=False)
    mse = mean_squared_error(true, pred, squared=True)
    mae = mean_absolute_error(true, pred)
    mape = mean_absolute_percentage_error(true+1, pred+1) 
    r2 = r2_score(true, pred)
    
    if verbose:
        print(f'--- Regression Metrics ---\n'
              f'Mean Squared Error (MSE): {mse:.4f}\n'
              f'Mean Absolute Error (MAE): {mae:.4f}\n'
              f'Root Mean Squared Error (RMSE): {rmse:.4f}\n'
              f'Mean Absolute Percentage Error (MAPE): {mape:.4f}\n'
              f'R-squared (R2): {r2:.4f}')
        
    return {'mse': mse, 'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}