from sklearn.ensemble import RandomForestClassifier

from skynet.base import SkyMLBase, fit, predict
from skynet.data_handling import get_init_features
from skynet.data_handling import get_init_vis_level
from skynet.data_handling import get_init_response


class SkyRandomForest(SkyMLBase, RandomForestClassifier):
    def __init__(self, n_estimators=10, criterion="gini", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, max_features="auto",
                 n_jobs=1, random_state=None):
        SkyMLBase.__init__(self)
        RandomForestClassifier.__init__(self, n_estimators=n_estimators, criterion=criterion,
                                        max_depth=max_depth, min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf, max_features=max_features,
                                        n_jobs=n_jobs, random_state=random_state)

        self.vis_level = get_init_vis_level()
        self.feature_ = get_init_features()
        self.target_ = get_init_response()

    def fit(self, X=None, y=None, sample_weight=None):
        if type(self.max_features) == "int":
            if self.max_features > X.shape[1]:
                self.reset(n_estimators=self.n_estimators, criterion=self.criterion,
                           max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                           min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                           n_jobs=self.n_jobs, random_state=self.random_state)
        if X is None:
            X = self.X_train

        if y is None:
            y = self.y_train

        fit(super(SkyMLBase, self), X, y, sample_weight)

    def predict(self, X=None):
        if X is None:
            X = self.X_test

        return predict(super(SkyMLBase, self), X)
