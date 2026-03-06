import torch


def mean_squared_error(y_true, y_pred):    
    mse = torch.mean((y_true - y_pred) ** 2)
    return mse.item()

def mean_absolute_error(y_true, y_pred):
    mae = torch.mean(torch.abs(y_true - y_pred))
    return mae.item()

# def r2_score(y_true, y_pred):
#     y_true_mean = torch.mean(y_true, dim=1, keepdim=True)
#     total_sum_squares = torch.sum((y_true - y_true_mean) ** 2, dim=1)
#     residual_sum_squares = torch.sum((y_true - y_pred) ** 2, dim=1)
#     r2 = torch.mean(1 - (residual_sum_squares / (total_sum_squares + 1e-8)))
#     return r2.item()

def r2_score(y_true, y_pred):
    total_sum_squares = torch.sum((y_true - torch.mean(y_true)) ** 2)
    residual_sum_squares = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - (residual_sum_squares / (total_sum_squares + 1e-8))
    return r2.item()

def mean_absolute_percentage_error(y_true, y_pred):
    mape = torch.mean(torch.abs((y_true - y_pred) / y_true))
    return mape.item()


