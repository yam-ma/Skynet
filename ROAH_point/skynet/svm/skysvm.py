from sklearn.svm import SVC

from skynet.base import SkyMLBase, fit, predict
from skynet.data_handling import get_init_features
from skynet.data_handling import get_init_vis_level
from skynet.data_handling import get_init_response


class SkySVM(SkyMLBase, SVC):
    def __init__(self, kernel="rbf", gamma="auto", C=1.0, random_state=None):
        SkyMLBase.__init__(self)
        SVC.__init__(self, kernel=kernel, gamma=gamma, C=C, random_state=random_state)

        self.vis_level = get_init_vis_level()
        self.feature_ = get_init_features()
        self.target_ = get_init_response()

    def fit(self, X=None, y=None, sample_weight=None):
        if X is None:
            X = self.X_train

        if y is None:
            y = self.y_train

        fit(super(SkyMLBase, self), X, y, sample_weight)

    def predict(self, X=None):
        if X is None:
            X = self.X_test

        return predict(super(SkyMLBase, self), X)
