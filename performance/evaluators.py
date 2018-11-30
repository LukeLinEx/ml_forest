import numpy as np


def ravel_single_variable(x):
    if len(x.shape) > 1 and x.shape[1] == 1:
        x = x.ravel()

    return x


from sklearn.metrics import mean_squared_error


def rmse(fval, lval):
    f = ravel_single_variable(fval)
    l = ravel_single_variable(lval)

    rmse_ = np.sqrt(
        mean_squared_error(f, l)
    )

    return rmse_
