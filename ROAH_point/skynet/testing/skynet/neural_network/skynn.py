from sklearn.neural_network import MLPClassifier

from skynet.base import SkyMLBase, fit, predict
from skynet.data_handling import get_init_features
from skynet.data_handling import get_init_vis_level
from skynet.data_handling import get_init_response


class SkyMLP(SkyMLBase, MLPClassifier):
    def __init__(self, hidden_layer_sizes=(256,), activation="relu", solver="adam",
                 alpha=0.0001, batch_size="auto", learning_rate="constant", learning_rate_init=0.001,
                 random_state=None, early_stopping=True, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        SkyMLBase.__init__(self)
        MLPClassifier.__init__(self, hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                               solver=solver, alpha=alpha, batch_size=batch_size, learning_rate=learning_rate,
                               learning_rate_init=learning_rate_init, random_state=random_state,
                               early_stopping=early_stopping, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

        self.vis_level = get_init_vis_level()
        self.feature_ = get_init_features()
        self.target_ = get_init_response()

    def fit(self, X=None, y=None):
        if X is None:
            X = self.X_train

        if y is None:
            y = self.y_train

        fit(super(SkyMLBase, self), X, y)

    def predict(self, X=None):
        if X is None:
            X = self.X_test

        return predict(super(SkyMLBase, self), X)
