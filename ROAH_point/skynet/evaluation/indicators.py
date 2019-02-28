import numpy as np

from sklearn.metrics import classification_report, accuracy_score
from skynet.datasets import learning_data


class Evaluator(object):
    def __init__(self):
        self.accuracy_ = None
        self.threat_score_ = None
        self.rmse_ = None

    def evaluate(self, p, y, threshold=None, conf_mat=False, eval_index=False):
        label = np.array(list(learning_data.get_init_vis_level().keys()))
        if label.min() != 0. or label.max() != 1.:
            if threshold is None:
                threshold = int(len(label) / 2)

            p = np.where(p > threshold, 0, 1)
            y = np.where(y > threshold, 0, 1)

        pt00 = np.where((p == 0.) & (y == 0.))[0]
        pt11 = np.where((p == 1.) & (y == 1.))[0]
        pt01 = np.where((p == 0.) & (y == 1.))[0]
        pt10 = np.where((p == 1.) & (y == 0.))[0]

        if conf_mat:
            confusion_matrix(len(pt11), len(pt00), len(pt10), len(pt01))

        self.accuracy_ = accuracy_score(y, p)
        self.threat_score_ = len(pt11) / (len(pt10) + len(pt11) + len(pt01))
        self.rmse_ = rmse(y, p)

        if eval_index:
            print(
                "Accuracy     : {:.3}\n"
                "Thread Score : {:.3}\n"
                "RMSE         : {:.3}\n".format(self.accuracy_, self.threat_score_, self.rmse_)
            )

        print(classification_report(y_true=y, y_pred=p))

        return self.threat_score_, self.accuracy_, self.rmse_


def confusion_matrix(tp, tn, fp, fn, name="Model"):
    print("Positive = \033[31m{}\033[0m, Negative = \033[31m{}\033[0m"
          .format(tp + fn, tn + fp))
    print("\033[34m%s\033[0m\n"
          "---------------------------------------\n"
          "  Act\Pre  |      +      |      -    \n"
          "---------------------------------------\n"
          "     +     |     %s     |     %s     \n"
          "---------------------------------------\n"
          "     -     |     %s     |     %s     \n"
          "---------------------------------------\n"
          % (name,
             str(tp).center(3),
             str(fn).center(3),
             str(fp).center(3),
             str(tn).center(3)))


def rmse(t, y):
    return np.sqrt((t - y) ** 2).mean()
