import xgboost as xgb
from ml_forest.core.elements.ftrans_base import FTransform


class XGBRegressorWithTuning(FTransform):
    def __init__(
            self,
            # These go to the XGB's booster
            objective="reg:linear", eval_metric="rmse", eta=0.3, gamma=0, max_depth=6, min_child_weight=1,
            subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_lambda=1, alpha=1, max_leaves=0,
            max_bin=256, seed=0,

            # These go to XGB Python API
            num_boost_round=10, early_stopping_rounds=50, feval=None,
            obj=None, maximize=None, verbose_eval=100):
        # TODO: write doc string
        """
        Update Doc String
        """
        super(XGBRegressorWithTuning, self).__init__(rise=1, tuning=True)
        args_passed = locals()
        essential_keys = [
            "seed", "maximize", "obj", "feval", "early_stopping_rounds", "num_boost_round", "max_bin",
            "max_leaves", "alpha", "reg_lambda", "colsample_bylevel", "colsample_bytree", "subsample",
            "min_child_weight", "max_depth", "gamma", "eta", "eval_metric", "objective"
        ]

        self.__essentials = {key: args_passed[key] for key in essential_keys}
        self.verbose_eval = verbose_eval

    def fit_singleton(self, x_train, y_train, x_valid, y_valid, x_test):
        xg_train = xgb.DMatrix(x_train, label=y_train)
        xg_valid = xgb.DMatrix(x_valid, label=y_valid)
        xg_test = xgb.DMatrix(x_test)

        # Params that go to XGB's booster
        pkeys = [
            "objective", "eval_metric", "eta", "gamma", "max_depth", "min_child_weight", "subsample",
            "colsample_bytree", "colsample_bylevel", "reg_lambda", "alpha", "max_leaves", "max_bin",
            "seed"
        ]
        params = {key: self.__essentials[key] for key in pkeys}
        params["silent"] = 1

        model = xgb.train(
            params=params, dtrain=xg_train,
            evals=[(xg_train, 'train'), (xg_valid, 'eval')],

            feval=self.__essentials["feval"],
            num_boost_round=self.__essentials['num_boost_round'],
            early_stopping_rounds=self.__essentials["early_stopping_rounds"],
            verbose_eval=self.verbose_eval
        )

        tmp = model.predict(xg_test, ntree_limit=model.best_ntree_limit)
        return model, tmp

    def transform(self, fed_test_value):
        xg_test = xgb.DMatrix(fed_test_value)

        result = []

        for model in self.models.values():
            result.append(
                model.predict(xg_test)
            )
        output = result[0]
        i = 1
        for pred in result[1:]:
            output += pred
            i += 1
        output = output / i

        return output.reshape(-1, 1)
