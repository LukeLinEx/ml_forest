import xgboost as xgb
from ml_forest.core.elements.ftrans_base import FTransform


class XGBClassifierWithTuning(FTransform):
    def __init__(
            self, objective="multi:softprob", eval_metric="mlogloss", eta=0.3, gamma=0, max_depth=6, min_child_weigh=1,
            subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_lambda=1, alpha=0, max_leaves=0, max_bin=256,

            num_boost_round=10, feval=None, predict_proba=False, early_stopping_rounds=50,

            num_class=None, obj=None, maximize=None, seed=0, verbose_eval=100):
        """

        ** TODO: not knowing yet how to add the 5 items into essentials
        :param early_stopping_rounds
        :param obj:
        :param feval:
        :param maximize:
        """

        super(XGBClassifierWithTuning, self).__init__(rise=1, tuning=True)

        # These go to the XGB's booster
        self.__essentials = {
            "objective":objective, "eval_metric":eval_metric, "eta":eta, "gamma":gamma,
            "max_depth":max_depth, "min_child_weigh":min_child_weigh, "subsample":subsample,
            "colsample_bytree":colsample_bytree, "colsample_bylevel":colsample_bylevel,
            "reg_lambda":reg_lambda, "alpha":alpha, "max_leaves":max_leaves, "max_bin":max_bin,

            # These go to XGB Python API
            "num_boost_round": num_boost_round, "feval": feval, "predict_proba": predict_proba,
            "early_stopping_rounds": early_stopping_rounds,

        }

        # parameters that doesn't go to essentials
        self._seed = seed
        self.verbose_eval = verbose_eval

    def fit_singleton(self, x_train, y_train, x_valid, y_valid, x_test):
        xg_train = xgb.DMatrix(x_train, label=y_train)
        xg_valid = xgb.DMatrix(x_valid, label=y_valid)
        xg_test = xgb.DMatrix(x_test)

        pkeys = [
            "objective", "eval_metric", "eta", "gamma", "max_depth", "min_child_weigh", "subsample",
            "colsample_bytree", "colsample_bylevel", "reg_lambda", "alpha", "max_leaves", "max_bin",
            "num_boost_round", "feval", "predict_proba", "early_stopping_rounds"
        ]

        params = {key:self.__essentials[key] for key in pkeys}

        params["silent"] = 1
        params["seed"] = self._seed

        num_boost_round = self.__essentials['num_boost_round']
        model = xgb.train(
            params=params, dtrain=xg_train, num_boost_round=num_boost_round,
            evals=[(xg_train, 'train'), (xg_valid, 'eval')],
            feval = self.__essentials["feval"],
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
                model.predict(xg_test, ntree_limit=model.best_ntree_limit)
            )
        output = result[0]
        i = 1
        for pred in result[1:]:
            output += pred
            i += 1
        output = output / i

        return output.reshape(-1, 1)
