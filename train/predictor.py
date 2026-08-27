from sklearn.metrics import (r2_score, 
                             root_mean_squared_error as rmse, 
                             mean_absolute_error as mae)


from train import utils

import torch
from torch.nn import Module
from torch_geometric.loader import DataLoader

from typing import Tuple

import numpy as np

class Predictor():
    def __init__(self):

        self.metric_func = {'Rp': utils.rp_calc,
                            'Rs': utils.rs_calc,
                            'R2': r2_score,
                            'RMSE': rmse,
                            'MAE': mae}


    @staticmethod
    def predict(model: Module, 
                test: DataLoader, 
                model_params=None) -> Tuple[np.ndarray, np.ndarray]:

        device = next(model.parameters()).device

        if model_params is not None:
            model.load_state_dict(model_params)

        model.eval()

        with torch.no_grad():

            y_test = []
            y_hat = []
                               
            for batch in test:
                batch = batch.to(device)
                          
                y_true = batch.y
                y = model(batch)
        
                y_hat.append(y.detach().cpu().numpy())
                y_test.append(y_true.detach().cpu().numpy())

        y_hat = np.concatenate([a.flatten() for a in y_hat])
        y_test = np.concatenate([a.flatten() for a in y_test])
        
        return y_test, y_hat

    def calc_perf_stats(self, y_test, y_hat):

        stats_line = 'Test Prediction Stats: '
        metrics_line = ''
        metrics_dict = dict()

        for k, v in self.metric_func.items():

            value = v(y_test, y_hat)
            metrics_line += f"{k} = {value:.3f} | "

            metrics_dict[k] = float(value)

        stats_line += metrics_line + '\n'
        print(stats_line)

        return metrics_dict