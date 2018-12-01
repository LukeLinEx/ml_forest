import numpy as np


def ravel_single_variable(x):
    if len(x.shape) > 1 and x.shape[1] == 1:
        x = x.ravel()

    return x


from sklearn.metrics import mean_squared_error


class Evaluator(object):
    def __call__(self, fval, lval):
        raise NotImplementedError


class RMSE(Evaluator):
    def __call__(self, fval, lval):
        f = ravel_single_variable(fval)
        l = ravel_single_variable(lval)

        rmse_ = np.sqrt(
            mean_squared_error(f, l)
        )

        return rmse_

class LRMSE_pred_been_trans(Evaluator):
    def __init__(self, shift=0):
        self.shift = shift

    def __call__(self, fval, lval):
        f = ravel_single_variable(fval)
        lval = np.log(lval + self.shift)
        l = ravel_single_variable(lval)

        rmse_ = np.sqrt(mean_squared_error(f, l))

        return rmse_


if __name__ == "__main__":
    import numpy as np

    a = np.array([1,2,3])
    b = np.array([1,2,3])
    c = np.array([3,2,1])

    evaluator = RMSE()
    print(type(evaluator).__name__)
    print(evaluator(a, b))
    print(evaluator(a, c))

    print("\n")

    shift = 1
    evaluator = LRMSE_pred_been_trans(shift)
    print(type(evaluator).__name__)
    print(evaluator(np.log(a+shift), b))
    print(evaluator(a, c))
