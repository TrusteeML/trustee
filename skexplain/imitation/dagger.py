import random
from abc import ABC
from copy import deepcopy

import numpy as np

import pandas as pd
from sklearn.metrics import classification_report, f1_score, r2_score
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
        max_iter=100,
        num_samples=2000,
        verbose=False,
    ):
        """Trains Decision Tree Regressor to imitate Expert model."""

        if verbose:
            self.log(
                "Initializing training dataset using {} as expert model".format(
                    self.expert
                )
            )

        features = X
        targets = self.expert.predict(X)

        if len(targets.shape) >= 2:
            targets = targets.argmax(axis=-1)
            # targets = targets.ravel()

        student = self.student_class(
            random_state=0,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            ccp_alpha=ccp_alpha,
        )

        if verbose:
            self.log("Expert model score: {}".format(self.score(y, targets)))
            self.log("Initializing Dagger loop with {} iterations".format(max_iter))

        # Dagger loop
        for i in range(max_iter):
            if verbose:
                self.log("#" * 10, "Iteration {}/{}".format(i, max_iter), "#" * 10)

            # Step 1: Sample predictions from training dataset
            #   - Some distribution? Or do I just use the dataset? Why not use all of it?
            #   - Sample randomly every time to generate different trees?
            if verbose:
                self.log(
                    "Sampling {} points from training dataset with {}/{} entries".format(
                        num_samples, len(features), len(targets)
                    )
                )
                self.log("Dataset {} entries".format(len(list(zip(features, targets)))))

            X_iter, y_iter = zip(
                *random.sample(
                    list(
                        zip(
                            features.values
                            if isinstance(features, pd.DataFrame)
                            else features,
                            targets,
                        )
                    ),
                    num_samples,
                )
            )
            X_iter, y_iter = np.array(X_iter), np.array(y_iter)

            X_train, X_test, y_train, y_test = train_test_split(
                X_iter, y_iter, train_size=0.7
            )

            self.log(X_iter)
            self.log("X_train X_test", X_train.shape, X_test.shape)
            self.log("y_train y_test", y_train.shape, y_test.shape)

            # Step 2: Traing DecisionTreeRegressor with sampled data
            student.fit(X_train, y_train)
            student_pred = student.predict(X_test)

            if verbose:
                self.log(
                    "Student model {} trained with depth {} and {} leaves:".format(
                        i, student.get_depth(), student.get_n_leaves()
                    )
                )
                self.log(
                    "Student model score: {}".format(self.score(y_test, student_pred))
                )

            # Step 3: Use expert model predictions to aggregate original dataset
            expert_pred = self.expert.predict(X_test)
            if len(expert_pred.shape) >= 2:
                # expert_pred = expert_pred.ravel()
                expert_pred = expert_pred.argmax(axis=-1)

            features = np.append(features, X_test, axis=0)
            targets = np.append(targets, expert_pred, axis=0)

            # Step 4: Calculate reward based on Decistion Tree Classifier fidelity to the Expert model
            reward = self.score(expert_pred, student_pred)
            if verbose:
                self.log("Student model {} fidelity: {}".format(i, reward))

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
        self.log("\n{}".format(classification_report(y_true, y_pred, digits=3)))
        return f1_score(y_true, y_pred, average=average)


class RegressionDagger(Dagger):
    """
    Implements the Dataset Aggregation (Dagger) algorithm to train a student Decision Tree Regressor
    based on observations from an Expert regression model.
    """

    def __init__(self, expert, logger=None):
        """Init method"""
        super().__init__(
            expert=expert, student_class=DecisionTreeRegressor, logger=logger
        )

    def score(self, y_true, y_pred):
        """Score function for student models"""
        return r2_score(y_true, y_pred)
