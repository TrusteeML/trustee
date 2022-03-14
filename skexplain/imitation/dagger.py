from abc import ABC
from copy import deepcopy

import torch
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class Dagger(ABC):
    """
    Implements the Dataset Aggregation (Dagger) algorithm to train
    student model based on observations from an Expert model.
    """

    def __init__(self, expert, student_class, logger=None):
        """Init method"""
        self.log = logger.log if logger else print
        self.expert = expert
        self.students = []
        self.student_class = student_class

    def score(self, y_true, y_pred):
        pass

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
            # targets = targets.ravel()

        student = self.student_class(
            random_state=0,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            ccp_alpha=ccp_alpha,
        )

        if verbose:
            self.log(f"Expert model score: {self.score(y, targets)}")
            self.log(f"Initializing Dagger loop with {num_iter} iterations")

        # Dagger loop
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
                X_iter, y_iter = (
                    features[samples_idxs].clone().detach(),
                    targets[samples_idxs].clone().detach(),
                )
            else:
                X_iter, y_iter = (
                    np.array(features)[samples_idxs],
                    np.array(targets)[samples_idxs],
                )

            X_train, X_test, y_train, y_test = train_test_split(X_iter, y_iter, train_size=train_size)

            # Step 2: Traing DecisionTreeRegressor with sampled data
            student.fit(X_train, y_train)
            student_pred = student.predict(X_test)

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

    def explain(self):
        """Returns explainable model that best imitates Expert model, based on calculated rewards."""
        return max(self.students, key=lambda item: item[1])

    def get_students(self):
        """Returns list of all (student, reward) obtained during the training process."""
        return self.students


class ClassificationDagger(Dagger):
    """
    Implements the Dataset Aggregation (Dagger) algorithm to train a student Decision Tree Classifier
    based on observations from an Expert classification model.
    """

    def __init__(self, expert, logger=None):
        """Init method"""
        super().__init__(expert, student_class=DecisionTreeClassifier, logger=logger)

    def score(self, y_true, y_pred, average="macro"):
        """Score function for student models"""
        return f1_score(y_true, y_pred, average=average)


class RegressionDagger(Dagger):
    """
    Implements the Dataset Aggregation (Dagger) algorithm to train a student Decision Tree Regressor
    based on observations from an Expert regression model.
    """

    def __init__(self, expert, logger=None):
        """Init method"""
        super().__init__(expert=expert, student_class=DecisionTreeRegressor, logger=logger)

    def score(self, y_true, y_pred):
        """Score function for student models"""
        return r2_score(y_true, y_pred)
