from skynet.feature_selection.filter import pearson_correlation
from skynet.feature_selection.filter import relief
from skynet.feature_selection.wrapper import WrapperSelector
from skynet.svm import SkySVM
from skynet.ensemble import SkyGradientBoosting
from skynet.ensemble import SkyXGBoosting
from skynet.ensemble import SkyRandomForest
from skynet.neural_network import SkyMLP


class FilterWrapperSelector(WrapperSelector):
    def __init__(self, classifier=SkySVM(), nof=10, param_grid=None,
                 filter_method="corr", depth=2):
        if filter_method == "corr":
            filter_method = pearson_correlation
        elif filter_method == "relief":
            filter_method = relief

        super(FilterWrapperSelector, self).__init__(
            classifier=classifier, nof=nof, param_grid=param_grid,
        )

        self.filter = filter_method
        self.depth = depth


def main():
    import os
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from skynet import SKYNET_PATH, DATA_PATH
    from skynet.data_handling import read_learning_data
    from skynet.data_handling.preprocessing import PreProcessor
    from skynet.data_handling import get_init_response
    from skynet.data_handling import split_time_series

    icao = "RJFK"

    train = read_learning_data(DATA_PATH + "/skynet/train_%s.pkl" % icao)
    test = read_learning_data(DATA_PATH + "/skynet/test_%s.pkl" % icao)

    conf1 = {"svm": {"classifier": SkySVM(),
                     "param_grid": {"C": [1, 10, 100, 1000, 10000],
                                    "kernel": ["rbf"],
                                    "gamma": [0.0001, 0.001, 0.01, 0.1, 1]},
                     },
             "gb": {"classifier": SkyGradientBoosting(),
                    "param_grid": {"n_estimators": [10, 100],
                                   "min_samples_split": [2, 10, 100],
                                   "min_samples_leaf": [1, 10, 50],
                                   "max_depth": [3, 4, 5, 6],
                                   "subsample": [0.8, 1.0]},
                    },
             "xgb": {"classifier": SkyXGBoosting(),
                     "param_grid": {"n_estimators": [10, 100],
                                    "min_samples_split": [2, 10, 100],
                                    "min_samples_leaf": [1, 10, 50],
                                    "max_depth": [3, 4, 5, 6],
                                    "subsample": [0.8, 1.0]}
                     },
             "forest": {"classifier": SkyRandomForest(),
                        "param_grid": {"n_estimators": [10, 100],
                                       "min_samples_split": [2, 10, 100],
                                       "min_samples_leaf": [1, 10, 50]
                                       },
                        },
             "nn": {"classifier": SkyMLP(),
                    "param_grid": {"alpha": [0.0001, 0.001, 0.01, 0.1]},
                    },
             "nof": 50,
             "filter_method": "corr",
             "depth": 50
             }
    # 期間別のfeature selection

    data = pd.concat([train, test]).reset_index(drop=True)

    preprocess = PreProcessor(norm=False, binary=False)
    preprocess.fit(data.iloc[:, :-1], data.iloc[:, -1])

    data = pd.concat([preprocess.X_train, preprocess.y_train], axis=1)

    sp = split_time_series(data, level="month", period=2)

    for model in ["svm", "xgb", "gb", "forest", "nn"]:
        print(model)
        print()
        os.makedirs(SKYNET_PATH + "/feature_selection/filter_wrapper/scores/%s" % model, exist_ok=True)
        m = {}
        for key in sp:
            ext = sp[key]

            target = get_init_response()
            feats = [f for f in ext.keys() if not (f in target + ["date"])]

            X = ext[feats]
            ss = StandardScaler()
            X = pd.DataFrame(ss.fit_transform(X), columns=X.keys())
            y = ext[target]

            selector = FilterWrapperSelector(classifier=conf1[model]["classifier"],
                                             nof=conf1["nof"],
                                             param_grid=conf1[model]["param_grid"],
                                             filter_method=conf1["filter_method"],
                                             depth=conf1["depth"])
            selector.fit(X=X, y=y)
            m[key] = selector.score
            m[key].to_csv(SKYNET_PATH + "/feature_selection/filter_wrapper/scores/%s/%s_%s.csv"
                          % (model, conf1["filter_method"], key),
                          index=False)


if __name__ == "__main__":
    main()
