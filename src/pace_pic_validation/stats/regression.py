import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
import numpy as np
import ast
import pandas as pd



def RobustRegression(x,y,):

    #transform
    x_ = np.array(x).reshape(-1, 1)
    y_ = np.array(y)

    #empty?
    if x_.size == 0:
        raise ValueError("Input arrays x is empty.")
    if y_.size == 0:
        raise ValueError("Input arrays y is empty.")

    #model
    X = sm.add_constant(x_)
    robust_model = sm.RLM(y_, X, M=sm.robust.norms.HuberT())
    results_robust = robust_model.fit()
    y_pred = results_robust.predict(X)

    #stats
    r2 = r2_score(y, y_pred)
    rmse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    params = results_robust.params
    intercept = params[0]
    coefficient = params[1]
    if intercept > 0:
        model= f'PIC = {coefficient:.3f} * IS + {intercept:.3f}'
    else:
        model = f'PIC = {coefficient:.3f} * IS {intercept:.3f}'

    #output
    print(results_robust.summary())
    print(f'Model: {model}')
    print(f"Robust Regression R²: {r2:.4f}")
    print(f"Robust RMSE: {rmse:.4f}")
    print(f"Robust MAE: {mae:.4f}")

    return y_pred, model, intercept, coefficient, r2, rmse, mae