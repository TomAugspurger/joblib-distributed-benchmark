"""
Run distributed joblib benchmarks.
"""
import argparse
import sys
from time import time

import distributed.joblib  # noqa
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.externals.joblib import parallel_backend
from sklearn.metrics import zero_one_loss
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.utils import check_array


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-s', '--scheduler-address',
                        help='Address for the scheduler like "tcp://...:8786"')
    parser.add_argument('-b', '--benchmark', choices=['basic', 'nested'])

    for backend in ['threading', 'dask', 'loky']:
        parser.add_argument(f'--{backend}', dest=backend, action='store_true')
        parser.add_argument(f'--no-{backend}', dest=backend,
                            action='store_false')

        parser.set_defaults(**{backend: True})

    return parser.parse_args(args)


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


def basic(scheduler_address, backends):
    ESTIMATORS = {
        'RandomForest': RandomForestClassifier(n_estimators=100),
        'ExtraTreesClassifier': ExtraTreesClassifier(n_estimators=100)
    }

    X_train, X_test, y_train, y_test = load_data()
    BACKENDS = build_backends(backends, scheduler_address, X_train, y_train)

    print("Dataset statistics:")
    print("===================")
    print("%s %d" % ("number of features:".ljust(25), X_train.shape[1]))
    print("%s %d" % ("number of classes:".ljust(25), np.unique(y_train).size))
    print("%s %s" % ("data type:".ljust(25), X_train.dtype))
    print("%s %d (pos=%d, neg=%d, size=%dMB)"
          % ("number of train samples:".ljust(25),
             X_train.shape[0], np.sum(y_train == 1),
             np.sum(y_train == 0), int(X_train.nbytes / 1e6)))
    print("%s %d (pos=%d, ackend)neg=%d, size=%dMB)"
          % ("number of test samples:".ljust(25),
             X_test.shape[0], np.sum(y_test == 1),
             np.sum(y_test == 0), int(X_test.nbytes / 1e6)))

    print()
    print("Training Classifiers")
    print("====================")
    error, train_time, test_time = {}, {}, {}
    for est_name, estimator in sorted(ESTIMATORS.items()):
        for backend, backend_kwargs in BACKENDS:
            print("Training %s with %s backend... " % (est_name, backend),
                  end="")
            estimator_params = estimator.get_params()

            estimator.set_params(**{p: RANDOM_STATE
                                    for p in estimator_params
                                    if p.endswith("random_state")})

            if "n_jobs" in estimator_params:
                estimator.set_params(n_jobs=-1)

            # Key for the results
            name = '%s, %s' % (est_name, backend)

            with parallel_backend(backend, **backend_kwargs):
                time_start = time()
                estimator.fit(X_train, y_train)
                train_time[name] = time() - time_start

            time_start = time()
            y_pred = estimator.predict(X_test)
            test_time[name] = time() - time_start

            error[name] = zero_one_loss(y_test, y_pred)

            print("done")

    print()
    print("Classification performance:")
    print("===========================")
    print("%s %s %s %s"
          % ("Classifier  ", "train-time", "test-time", "error-rate"))
    print("-" * 44)
    for name in sorted(error, key=error.get):
        print("%s %s %s %s" % (name,
                               ("%.4fs" % train_time[name]),
                               ("%.4fs" % test_time[name]),
                               ("%.4f" % error[name])))

    print()


def build_backends(backends, scheduler_host, X_train, y_train):
    BACKENDS = []

    while backends:
        backend = backends.pop()
        if backend == 'threading':
            BACKENDS.append(('threading', {}))
        elif backend == 'dask.distributed':
            BACKENDS.append(('dask.distributed', {
                'scheduler_host': scheduler_host,
                'scatter': [X_train, y_train]
            }))
        elif backend == 'loky':
            BACKENDS.append(('loky', {}))
        else:
            raise ValueError(f"Bad backend {backend}")

    return BACKENDS


def nested(scheduler_address, backends):
    X_train, X_test, y_train, y_test = load_data()
    BACKENDS = build_backends(backends, scheduler_address, X_train, y_train)
    n_jobs_grid = [-1, 1]

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

    error, train_time = {}, {}
    for backend, backend_kwargs in BACKENDS:
        for n_jobs_outer in n_jobs_grid:
            for n_jobs_inner in n_jobs_grid:
                clf = RandomForestClassifier(random_state=RANDOM_STATE,
                                             n_estimators=10,
                                             n_jobs=-1)
                param_grid = {
                    'max_features': [4, 8, 12],
                    'min_samples_split': [2, 5],
                }
                gs = GridSearchCV(clf, param_grid, cv=5, n_jobs=n_jobs_inner,
                                  verbose=2)
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


def main(args=None):
    args = parse_args(args)

    backends = []
    if args.loky:
        backends.append('loky')

    if args.threading:
        backends.append('threading')

    if args.dask:
        backends.append('dask.distributed')

    if args.benchmark == 'basic':
        basic(args.scheduler_address, backends)

    elif args.benchmark == 'nested':
        nested(args.scheduler_address, backends)


if __name__ == '__main__':
    sys.exit(main(None))
