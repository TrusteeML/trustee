"""Shared fixtures for the Trustee test suite."""

import matplotlib
import numpy as np
import pytest
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Tests must never try to open a window.
matplotlib.use("Agg")

RANDOM_STATE = 0


@pytest.fixture(autouse=True)
def deterministic_random():
    """Seed the global RNG before every test.

    Trustee.fit() samples with np.random.choice and calls train_test_split without a
    random_state, so both draw from NumPy's global state. Without this, results differ
    run to run and any assertion on fidelity or agreement is flaky.
    """
    np.random.seed(RANDOM_STATE)


@pytest.fixture(scope="session")
def iris():
    """The iris dataset as plain numpy arrays."""
    return datasets.load_iris(return_X_y=True)


@pytest.fixture(scope="session")
def iris_frame():
    """The iris dataset as a (DataFrame, Series) pair."""
    return datasets.load_iris(return_X_y=True, as_frame=True)


@pytest.fixture(scope="session")
def iris_bunch():
    """The full iris bunch, for its `target_names` / `feature_names`."""
    return datasets.load_iris()


@pytest.fixture(scope="session")
def iris_split(iris):
    X, y = iris
    return train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)


@pytest.fixture(scope="session")
def diabetes():
    """A regression dataset as plain numpy arrays."""
    return datasets.load_diabetes(return_X_y=True)


@pytest.fixture(scope="session")
def blackbox(iris_split):
    """A fitted classification blackbox for Trustee to explain."""
    X_train, _, y_train, _ = iris_split
    return RandomForestClassifier(n_estimators=10, random_state=RANDOM_STATE).fit(X_train, y_train)


@pytest.fixture(scope="session")
def fitted_classifier(iris):
    """A plain fitted DecisionTreeClassifier, for tree-utility tests."""
    X, y = iris
    return DecisionTreeClassifier(random_state=RANDOM_STATE).fit(X, y)


@pytest.fixture(scope="session")
def fitted_regressor(diabetes):
    """A plain fitted DecisionTreeRegressor, for tree-utility tests."""
    X, y = diabetes
    return DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=4).fit(X, y)
