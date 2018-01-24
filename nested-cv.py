from time import time

import numpy as np
import pandas as pd
import distributed.joblib  # noqa
from distributed import Client

from sklearn.datasets import fetch_covtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.joblib import parallel_backend
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.utils import check_array


RANDOM_STATE = 13


def load_data():
    # Load dataset
    print("Loading dataset...")
    data = fetch_covtype(download_if_missing=True, shuffle=True,
                         random_state=RANDOM_STATE)
    X = check_array(data['data'], dtype=np.float32, order='C')
    y = (data['target'] != 1).astype(np.int)

    # Create train-test split (as [Joachims, 2006])
    print("Creating train-test split...")
    n_train = 522911
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    # Standardize first 10 features (the numerical ones)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    mean[10:] = 0.0
    std[10:] = 1.0
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    return X_train, X_test, y_train, y_test


ESTIMATORS = {
    'RandomForest': RandomForestClassifier(n_estimators=100),
}

X_train, X_test, y_train, y_test = load_data()


if __name__ == '__main__':
    import sys

    try:
        SCHEDULER_ADDRESS = sys.argv[1]
        client = Client(SCHEDULER_ADDRESS)
    except IndexError:
        print("Provide a scheduler address")
        sys.exit(1)

    BACKENDS = [
        # ('threading', {}),
        ('dask.distributed', {
            'scheduler_host': SCHEDULER_ADDRESS,
            'scatter': [X_train, y_train],
        }),
        # ('loky', {}),
    ]

    n_jobs_grid = [1, -1]

    print("Dataset statistics:")
    print("===================")
    print("%s %d" % ("number of features:".ljust(25), X_train.shape[1]))
    print("%s %d" % ("number of classes:".ljust(25), np.unique(y_train).size))
    print("%s %s" % ("data type:".ljust(25), X_train.dtype))
    print("%s %d (pos=%d, neg=%d, size=%dMB)"
          % ("number of train samples:".ljust(25),
             X_train.shape[0], np.sum(y_train == 1),
             np.sum(y_train == 0), int(X_train.nbytes / 1e6)))
    print("%s %d (pos=%d, neg=%d, size=%dMB)"
          % ("number of test samples:".ljust(25),
             X_test.shape[0], np.sum(y_test == 1),
             np.sum(y_test == 0), int(X_test.nbytes / 1e6)))

    error, train_time, test_time = {}, {}, {}
    for backend, backend_kwargs in BACKENDS:
        for n_jobs_outer in n_jobs_grid:
            for n_jobs_inner in n_jobs_grid:
                clf = RandomForestClassifier(random_state=RANDOM_STATE,
                                             n_estimators=100)
                param_grid = {
                    'max_features': [5, 25, 54],
                    'min_samples_split': [2, 5, 10],
                }
                gs = GridSearchCV(clf, param_grid, cv=5, n_jobs=n_jobs_inner)
                name = '%s,%s,%s' % (backend, n_jobs_outer, n_jobs_inner)

                print("Training with {}...".format(name), end="")

                with parallel_backend(backend, **backend_kwargs):
                    time_start = time()
                    cv_gs = cross_validate(gs, X=X_train, y=y_train, cv=5,
                                           return_train_score=True,
                                           n_jobs=n_jobs_outer)
                    train_time[name] = time() - time_start
                    error[name] = cv_gs['test_score'].mean()

                print("done")
                df = pd.DataFrame(cv_gs)
                df.to_csv("{}.csv".format(name))

    print("{:<25} | {}".format("Backend", "Train Time"))
    print("-" * 44)
    for name in sorted(error, key=error.get):
        print("{:<25} | {}".format(name, train_time[name]))
