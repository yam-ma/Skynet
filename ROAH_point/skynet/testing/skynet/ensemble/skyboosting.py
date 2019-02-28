from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier

from skynet.base import SkyMLBase, fit, predict
from skynet.data_handling import get_init_features
from skynet.data_handling import get_init_vis_level
from skynet.data_handling import get_init_response


class SkyGradientBoosting(SkyMLBase, GradientBoostingClassifier):
    def __init__(self, loss="deviance", learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion="friedman_mse", min_samples_split=2,
                 min_samples_leaf=1, max_depth=3):
        SkyMLBase.__init__(self)
        GradientBoostingClassifier.__init__(self, loss=loss, learning_rate=learning_rate,
                                            n_estimators=n_estimators, subsample=subsample,
                                            criterion=criterion, min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf, max_depth=max_depth)

        self.vis_level = get_init_vis_level()
        self.feature_ = get_init_features()
        self.target_ = get_init_response()

    def fit(self, X=None, y=None, sample_weight=None, monitor=None):
        if X is None:
            X = self.X_train

        if y is None:
            y = self.y_train

        fit(super(SkyMLBase, self), X, y, sample_weight, monitor=monitor)

    def predict(self, X=None):
        if X is None:
            X = self.X_test

        return predict(super(SkyMLBase, self), X)

    def _make_estimator(self, append=True, **kwargs):
        pass


class SkyXGBoosting(SkyMLBase, XGBClassifier):
    def __init__(self, max_depth=3, learning_rate=0.1,
                 n_estimators=100, silent=True,
                 objective="binary:logistic", booster='gbtree',
                 n_jobs=1, nthread=None, gamma=0, min_child_weight=1,
                 max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, random_state=0, seed=None, missing=None, **kwargs):
        SkyMLBase.__init__(self)
        XGBClassifier.__init__(self, max_depth=max_depth, learning_rate=learning_rate,
                               n_estimators=n_estimators, silent=silent,
                               objective=objective, booster=booster,
                               n_jobs=n_jobs, nthread=nthread, gamma=gamma,
                               min_child_weight=min_child_weight, max_delta_step=max_delta_step,
                               subsample=subsample, colsample_bytree=colsample_bytree,
                               colsample_bylevel=colsample_bylevel, reg_alpha=reg_alpha,
                               reg_lambda=reg_lambda, scale_pos_weight=scale_pos_weight,
                               base_score=base_score, random_state=random_state,
                               seed=seed, missing=missing, **kwargs)

    def fit(self, X=None, y=None, sample_weight=None, eval_set=None, eval_metric=None,
            early_stopping_rounds=None, verbose=True, xgb_model=None,
            sample_weight_eval_set=None):
        if X is None:
            X = self.X_train

        if y is None:
            y = self.y_train

        fit(super(SkyMLBase, self), X, y, sample_weight,
            eval_set=eval_set, eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds, verbose=verbose,
            xgb_model=xgb_model, sample_weight_eval_set=sample_weight_eval_set)

    def predict(self, X=None, output_margin=False, ntree_limit=None):
        if X is None:
            X = self.X_test

        return predict(super(SkyMLBase, self), X)
