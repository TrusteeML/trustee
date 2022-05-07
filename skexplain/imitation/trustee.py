import abc
import torch
import functools
import numpy as np
import pandas as pd
from copy import deepcopy

from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


from skexplain.helpers import get_dt_info, prune_index, get_dt_dict


def _check_if_trained(func):
    """
    Check whether the Trustee is already fitted and self.best_student exists
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.best_student:
            raise ValueError("No student models have been trained yet. Please fit() Trustee explainer first.")
        return func(self, *args, **kwargs)

    return wrapper


class Trustee(abc.ABC):
    """
    Implements the Trust-oriented Decision Tree Extraction (Trustee) algorithm to train
    student model based on observations from an Expert model.
    """

    def __init__(self, expert, student_class, logger=None):
        """Init method"""
        self.log = logger.log if logger else print
        self.expert = expert
        self.students = []
        self.student_class = student_class

        self.best_student = None
        self.features = None
        self.nodes = None
        self.branches = None

    @abc.abstractmethod
    def score(self, y_true, y_pred):
        """Score function for student models"""

    def fit(
        self,
        X,
        y,
        max_leaf_nodes=None,
        max_depth=None,
        ccp_alpha=0.0,
        train_size=0.7,
        num_iter=100,
        num_samples=2000,
        samples_size=None,
        use_features=None,
        predict_method_name="predict",
        optimization="fidelity",  # for comparative purposes only
        aggregate=True,  # for comparative purposes only
        verbose=False,
    ):
        """Trains Decision Tree Regressor to imitate Expert model."""
        if verbose:
            self.log(f"Initializing training dataset using {self.expert} as expert model")

        if len(X) != len(y):
            raise ValueError("Features (X) and target (y) values should have the same length.")

        features = X
        targets = getattr(self.expert, predict_method_name)(X)

        if hasattr(targets, "shape") and len(targets.shape) >= 2:
            targets = targets.argmax(axis=-1)

        student = self.student_class(
            random_state=0,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            ccp_alpha=ccp_alpha,
        )

        if verbose:
            self.log(f"Expert model score: {self.score(y, targets)}")
            self.log(f"Initializing Trustee loop with {num_iter} iterations")

        # Trustee loop
        for i in range(num_iter):
            if verbose:
                self.log("#" * 10, f"Iteration {i}/{num_iter}", "#" * 10)

            dataset_size = len(features)
            size = int(int(len(X)) * samples_size) if samples_size else num_samples
            # Step 1: Sample predictions from training dataset
            if verbose:
                self.log(
                    "Sampling {} points from training dataset with ({}, {}) entries".format(
                        size, len(features), len(targets)
                    )
                )

            samples_idxs = np.random.choice(dataset_size, size=size, replace=False)

            if isinstance(features, pd.DataFrame) and isinstance(targets, pd.Series):
                X_iter, y_iter = features.iloc[samples_idxs], targets.iloc[samples_idxs]
            elif isinstance(features, np.ndarray) and isinstance(targets, np.ndarray):
                X_iter, y_iter = features[samples_idxs], targets[samples_idxs]
            elif torch.is_tensor(features) and torch.is_tensor(targets):
                X_iter, y_iter = features[samples_idxs].clone().detach(), targets[samples_idxs].clone().detach()
            else:
                X_iter, y_iter = np.array(features)[samples_idxs], np.array(targets)[samples_idxs]

            X_train, X_test, y_train, y_test = train_test_split(X_iter, y_iter, train_size=train_size)

            X_train_student = X_train
            X_test_student = X_test
            if use_features is not None:
                if isinstance(X_train, pd.DataFrame):
                    X_train_student = X_train.iloc[:, use_features]
                    X_test_student = X_test.iloc[:, use_features]
                elif isinstance(X_train, np.ndarray):
                    X_train_student = np.reshape(X_train[:, [use_features]], (X_train.shape[0], -1))
                    X_test_student = np.reshape(X_test[:, [use_features]], (X_test.shape[0], -1))
                elif torch.is_tensor(X_train):
                    X_train_student = X_train[:, [use_features]].clone().detach()
                    X_test_student = X_test[:, [use_features]].clone().detach()
                else:
                    X_train_student = np.array(X_train)[:, [use_features]]
                    X_test_student = np.array(X_test)[:, [use_features]]

            # Step 2: Traing DecisionTreeRegressor with sampled data
            student.fit(X_train_student, y_train)
            student_pred = student.predict(X_test_student)

            if verbose:
                self.log(
                    "Student model {} trained with depth {} and {} leaves:".format(
                        i, student.get_depth(), student.get_n_leaves()
                    )
                )
                self.log(f"Student model score: {self.score(y_test, student_pred)}")

            # Step 3: Use expert model predictions to aggregate original dataset
            expert_pred = getattr(self.expert, predict_method_name)(X_test)
            if hasattr(expert_pred, "shape") and len(expert_pred.shape) >= 2:
                # expert_pred = expert_pred.ravel()
                expert_pred = expert_pred.argmax(axis=-1)

            if aggregate:
                if isinstance(features, pd.DataFrame) and isinstance(targets, pd.Series):
                    features = features.append(X_test)
                    targets = targets.append(expert_pred)
                elif torch.is_tensor(features) and torch.is_tensor(targets):
                    features = torch.cat((features, X_test), 0)
                    targets = torch.cat((targets, expert_pred), 0)
                else:
                    features = np.append(features, X_test, axis=0)
                    targets = np.append(targets, expert_pred, axis=0)

            if optimization == "accuracy":
                # Step 4: Calculate reward based on Decistion Tree Classifier accuracy
                reward = self.score(y_test, student_pred)
            else:
                # Step 4: Calculate reward based on Decistion Tree Classifier fidelity to the Expert model
                reward = self.score(expert_pred, student_pred)

            if verbose:
                self.log(f"Student model {i} fidelity: {reward}")

            # Step 5: Somehow incorporate that reward onto training process?
            # - Maybe just store the highest reward possible and use that as output?
            self.students.append((deepcopy(student), reward, i))

        self.best_student = self.explain()[0]

    def explain(self):
        """Returns explainable model that best imitates Expert model, based on calculated rewards."""
        if not self.students:
            raise ValueError("No student models have been trained yet. Please fit() Trustee explaimer first.")

        return max(self.students, key=lambda item: item[1])

    @_check_if_trained
    def get_students(self):
        """Returns list of all (student, reward) obtained during the training process."""
        return self.students

    @_check_if_trained
    def get_n_features(self):
        """Returns number of features used in the top student model."""

        if not self.features:
            self.features, self.nodes, self.branches = get_dt_info(self.best_student)

        return len(self.features.keys())

    @_check_if_trained
    def get_n_classes(self):
        """Returns number of classes used in the top student model."""

        return self.best_student.tree_.n_classes[0]

    @_check_if_trained
    def get_samples_sum(self):
        """Returns the sum of all samples in all non-leaf nodes in best student model."""

        left = self.best_student.tree_.children_left
        right = self.best_student.tree_.children_right
        samples = self.best_student.tree_.n_node_samples

        return np.sum([n_samples if left[node] != right[node] else 0 for node, n_samples in enumerate(samples)])

    @_check_if_trained
    def get_top_branches(self, top_k=10):
        """Returns list of top branches of the best student."""

        if not self.branches:
            self.features, self.nodes, self.branches = get_dt_info(self.best_student)

        return sorted(self.branches, key=lambda p: p["samples"], reverse=True)[:top_k]

    @_check_if_trained
    def get_top_features(self, top_k=10):
        """Returns list of top features of the best student."""

        if not self.features:
            self.features, self.nodes, self.branches = get_dt_info(self.best_student)

        return sorted(self.features.items(), key=lambda p: p[1]["samples"], reverse=True)[:top_k]

    @_check_if_trained
    def get_top_nodes(self, top_k=10):
        """Returns list of top nodes of the best student."""

        if not self.nodes:
            self.features, self.nodes, self.branches = get_dt_info(self.best_student)

        return sorted(
            self.nodes, key=lambda p: p["samples"] * abs(p["gini_split"][0] - p["gini_split"][1]), reverse=True
        )[:top_k]

    @_check_if_trained
    def get_samples_by_level(self):
        """Returns list of samples by level of the best student."""

        if not self.nodes:
            self.features, self.nodes, self.branches = get_dt_info(self.best_student)

        samples_by_level = list(np.zeros(self.best_student.get_depth() + 1))
        nodes_by_level = list(np.zeros(self.best_student.get_depth() + 1).astype(int))
        for node in self.nodes:
            samples_by_level[node["level"]] += node["samples"]
            # nodes_by_level[node["level"]] += 1

        for node in self.branches:
            samples_by_level[node["level"]] += node["samples"]
            nodes_by_level[node["level"]] += 1

        return samples_by_level

    @_check_if_trained
    def get_leaves_by_level(self):
        """Returns list of leaves by level of the best student."""

        if not self.branches:
            self.features, self.nodes, self.branches = get_dt_info(self.best_student)

        leaves_by_level = list(np.zeros(self.best_student.get_depth() + 1).astype(int))
        for node in self.branches:
            leaves_by_level[node["level"]] += 1

        return leaves_by_level

    @_check_if_trained
    def prune(self, top_k=10, max_impurity=0.10):
        """Prunes and returns the best student model explanation from the list of students."""

        top_branches = self.get_top_branches(top_k=top_k)
        prunned_student = deepcopy(self.best_student)

        nodes_to_keep = set({})
        for branch in top_branches:
            for (node, _, _, _) in branch["path"]:
                if self.best_student.tree_.impurity[node] > max_impurity:
                    nodes_to_keep.add(node)

        for node in self.nodes:
            if node["idx"] not in nodes_to_keep:
                prune_index(prunned_student, node["idx"], 0)

        # update classifier with prunned model
        prunned_student.tree_.__setstate__(get_dt_dict(prunned_student))

        return prunned_student


class ClassificationTrustee(Trustee):
    """
    Implements the Trust-oriented Decision Tree Extraction (Trustee) algorithm to train a student Decision Tree Classifier
    based on observations from an Expert classification model.
    """

    def __init__(self, expert, logger=None):
        """Init method"""
        super().__init__(expert, student_class=DecisionTreeClassifier, logger=logger)

    def score(self, y_true, y_pred, average="macro"):
        """Score function for student models"""
        return f1_score(y_true, y_pred, average=average)


class RegressionTrustee(Trustee):
    """
    Implements the Trust-oriented Decision Tree Extraction (Trustee) algorithm to train a student Decision Tree Regressor
    based on observations from an Expert regression model.
    """

    def __init__(self, expert, logger=None):
        """Init method"""
        super().__init__(expert=expert, student_class=DecisionTreeRegressor, logger=logger)

    def score(self, y_true, y_pred):
        """Score function for student models"""
        return r2_score(y_true, y_pred)
