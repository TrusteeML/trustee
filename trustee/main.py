"""
Trustee
====================================
The core module of the Trustee project
"""
import abc
import functools
import numpy as np
import pandas as pd

from copy import deepcopy


from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from trustee.utils.tree import get_dt_info, top_k_prune


def _check_if_trained(func):
    """
    Checks whether the Trustee is already fitted and self._best_student exists

    Parameters
    ----------
    func
        Function to apply decorator to.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if len(self._top_students) == 0:
            raise ValueError("No student models have been trained yet. Please fit() Trustee explainer first.")
        return func(self, *args, **kwargs)

    return wrapper


class Trustee(abc.ABC):
    """
    Base implementation the Trust-oriented Decision Tree Extraction (Trustee)
    algorithm to train student model based on observations from an Expert model.
    """

    def __init__(self, expert, student_class, logger=None):
        """
        Trustee constructor

        Parameters
        ----------
        expert
            The ML blackbox model to analyze.
        student_class
            Class of student to train based on blackbox model predictions
        logger (optional)
            A logger object

        """
        self.log = logger.log if logger else print
        self.expert = expert
        self.student_class = student_class

        self._students_by_iter = []
        self._top_students = []
        self._stable_students = []

        self._X_train = []
        self._X_test = []
        self._y_train = []
        self._y_test = []

        self._best_student = None
        self._features = None
        self._nodes = None
        self._branches = None

    @abc.abstractmethod
    def _score(self, y_true, y_pred):
        """Score function for student models"""

    def fit(
        self,
        X,
        y,
        top_k=10,
        max_leaf_nodes=None,
        max_depth=None,
        ccp_alpha=0.0,
        train_size=0.7,
        num_iter=50,
        num_stability_iter=5,
        num_samples=2000,
        samples_size=None,
        use_features=None,
        predict_method_name="predict",
        optimization="fidelity",  # for comparative purposes only
        aggregate=True,  # for comparative purposes only
        verbose=False,
    ):
        """
        Trains Decision Tree Regressor to imitate Expert model.

        Parameters
        ----------
        X
        y
        max_leaf_nodes
        max_depth
        ccp_alpha
        train_size
        num_iter
        num_samples
        samples_size
        use_features
        predict_method_name
        optimization
        aggregate
        verbose
        """
        if verbose:
            self.log(f"Initializing training dataset using {self.expert} as expert model")

        if len(X) != len(y):
            raise ValueError("Features (X) and target (y) values should have the same length.")

        # convert data to np array to facilitate processing
        X = pd.DataFrame(X)
        y = pd.Series(y)

        # split input array to train DTs and evaluate agreement
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(X, y, train_size=train_size)

        features = self._X_train
        targets = pd.Series(getattr(self.expert, predict_method_name)(self._X_train))

        if hasattr(targets, "shape") and len(targets.shape) >= 2:
            targets = targets.ravel()

        student = self.student_class(
            random_state=0, max_leaf_nodes=max_leaf_nodes, max_depth=max_depth, ccp_alpha=ccp_alpha
        )

        if verbose:
            self.log(f"Expert model score: {self._score(self._y_train, targets)}")
            self.log(f"Initializing Trustee outer-loop with {num_stability_iter} iterations")

        # Trustee outer-loop
        for i in range(num_stability_iter):
            self._students_by_iter.append([])
            if verbose:
                self.log("#" * 10, f"Outer-loop Iteration {i}/{num_stability_iter}", "#" * 10)
                self.log(f"Initializing Trustee inner-loop with {num_stability_iter} iterations")

            # Trustee inner-loop
            for j in range(num_iter):
                if verbose:
                    self.log("#" * 10, f"Inner-loop Iteration {j}/{num_iter}", "#" * 10)

                dataset_size = len(features)
                size = int(int(len(self._X_train)) * samples_size) if samples_size else num_samples
                # Step 1: Sample predictions from training dataset
                if verbose:
                    self.log(
                        f"Sampling {size} points from training dataset with ({len(features)}, {len(targets)}) entries"
                    )

                samples_idxs = np.random.choice(dataset_size, size=size, replace=False)
                X_iter, y_iter = features.iloc[samples_idxs], targets.iloc[samples_idxs]
                X_iter_train, X_iter_test, y_iter_train, y_iter_test = train_test_split(
                    X_iter, y_iter, train_size=train_size
                )

                X_train_student = X_iter_train
                X_test_student = X_iter_test
                if use_features is not None:
                    X_train_student = X_iter_train.iloc[:, use_features]
                    X_test_student = X_iter_test.iloc[:, use_features]

                # Step 2: Traing DecisionTreeRegressor with sampled data
                student.fit(X_train_student.values, y_iter_train.values)
                student_pred = student.predict(X_test_student.values)

                if verbose:
                    self.log(
                        f"Student model {i}-{j} trained with depth {student.get_depth()} and {student.get_n_leaves()} leaves:"
                    )
                    self.log(f"Student model score: {self._score(y_iter_test, student_pred)}")

                # Step 3: Use expert model predictions to aggregate original dataset
                expert_pred = pd.Series(getattr(self.expert, predict_method_name)(X_iter_test))
                if hasattr(expert_pred, "shape") and len(expert_pred.shape) >= 2:
                    expert_pred = expert_pred.ravel()

                if aggregate:
                    features = pd.concat([features, X_iter_test])
                    targets = pd.concat([targets, expert_pred])

                if optimization == "accuracy":
                    # Step 4: Calculate reward based on Decistion Tree Classifier accuracy
                    reward = self._score(y_iter_test, student_pred)
                else:
                    # Step 4: Calculate reward based on Decistion Tree Classifier fidelity to the Expert model
                    reward = self._score(expert_pred, student_pred)

                if verbose:
                    self.log(f"Student model {i}-{j} fidelity: {reward}")

                # Save student to list of iterations dt
                self._students_by_iter[i].append((deepcopy(student), reward))

            # Save student with highest fidelity to list of top students by iteration
            self._top_students.append(max(self._students_by_iter[i], key=lambda item: item[1]))

        # Get best overall student based on mean agreement
        self._best_student = self.explain(top_k=top_k)[0]

    @_check_if_trained
    def explain(self, top_k=10):
        """
        Returns explainable model that best imitates Expert model, based on calculated rewards.
        """
        # Return dt with highest mean agreement when pruned (with no thrshold)
        stable = self.get_stable(top_k=top_k, threshold=0, sort=False)
        return max(stable, key=lambda item: item[2])

    @_check_if_trained
    def get_stable(self, top_k=10, threshold=0.9, sort=True):
        """
        Filters out explanations from Trustee sability analysis with less than threshold agreement.

        Parameters
        ----------
        top_k = 10
        threshold = 0.9
        sort = True
        """
        if len(self._stable_students) == 0:
            agreement = []
            # Calculate pair-wise agreement of all top students generated during inner loop
            for i, _ in enumerate(self._top_students):
                agreement.append([])
                # Apply top-k prunning before calculating agreement
                base_tree = top_k_prune(self._top_students[i][0], top_k=top_k)
                for j, _ in enumerate(self._top_students):
                    # Apply top-k prunning before calculating agreement
                    iter_tree = top_k_prune(self._top_students[j][0], top_k=top_k)

                    iter_y_pred = iter_tree.predict(self._X_test.values)
                    base_y_pred = base_tree.predict(self._X_test.values)

                    agreement[i].append(self._score(iter_y_pred, base_y_pred))

                # Save complete dt, top-k prune dt, mean agreement and fidelity
                self._stable_students.append(
                    (
                        self._top_students[i][0],
                        base_tree,
                        np.mean(agreement[i]),
                        self._top_students[i][1],
                    )
                )

        stable = self._stable_students
        if threshold > 0:
            stable = filter(lambda item: item[2] >= threshold, stable)

        if sort:
            return sorted(stable, key=lambda item: item[2], reverse=True)

        return stable

    @_check_if_trained
    def get_all_students(self):
        """
        Returns list of all (student, reward) obtained during the inner-loop process.
        """
        return self._students_by_iter

    @_check_if_trained
    def get_top_students(self):
        """
        Returns list of top (students, reward) obtained during the outer-loop process.
        """
        return self._top_students

    @_check_if_trained
    def get_n_features(self):
        """
        Returns number of features used in the top student model.
        """
        if not self._features:
            self._features, self._nodes, self._branches = get_dt_info(self._best_student)

        return len(self._features.keys())

    @_check_if_trained
    def get_n_classes(self):
        """
        Returns number of classes used in the top student model.
        """
        return self._best_student.tree_.n_classes[0]

    @_check_if_trained
    def get_samples_sum(self):
        """
        Returns the sum of all samples in all non-leaf _nodes in best student model.
        """
        left = self._best_student.tree_.children_left
        right = self._best_student.tree_.children_right
        samples = self._best_student.tree_.n_node_samples

        return np.sum([n_samples if left[node] != right[node] else 0 for node, n_samples in enumerate(samples)])

    @_check_if_trained
    def get_top_branches(self, top_k=10):
        """
        Returns list of top _branches of the best student.
        """
        if not self._branches:
            self._features, self._nodes, self._branches = get_dt_info(self._best_student)

        return sorted(self._branches, key=lambda p: p["samples"], reverse=True)[:top_k]

    @_check_if_trained
    def get_top_features(self, top_k=10):
        """
        Returns list of top _features of the best student.
        """
        if not self._features:
            self._features, self._nodes, self._branches = get_dt_info(self._best_student)

        return sorted(self._features.items(), key=lambda p: p[1]["samples"], reverse=True)[:top_k]

    @_check_if_trained
    def get_top_nodes(self, top_k=10):
        """
        Returns list of top _nodes of the best student.
        """
        if not self._nodes:
            self._features, self._nodes, self._branches = get_dt_info(self._best_student)

        return sorted(
            self._nodes, key=lambda p: p["samples"] * abs(p["gini_split"][0] - p["gini_split"][1]), reverse=True
        )[:top_k]

    @_check_if_trained
    def get_samples_by_level(self):
        """
        Returns list of samples by level of the best student.
        """
        if not self._nodes:
            self._features, self._nodes, self._branches = get_dt_info(self._best_student)

        samples_by_level = list(np.zeros(self._best_student.get_depth() + 1))
        nodes_by_level = list(np.zeros(self._best_student.get_depth() + 1).astype(int))
        for node in self._nodes:
            samples_by_level[node["level"]] += node["samples"]

        for node in self._branches:
            samples_by_level[node["level"]] += node["samples"]
            nodes_by_level[node["level"]] += 1

        return samples_by_level

    @_check_if_trained
    def get_leaves_by_level(self):
        """
        Returns list of leaves by level of the best student.
        """
        if not self._branches:
            self._features, self._nodes, self._branches = get_dt_info(self._best_student)

        leaves_by_level = list(np.zeros(self._best_student.get_depth() + 1).astype(int))
        for node in self._branches:
            leaves_by_level[node["level"]] += 1

        return leaves_by_level

    @_check_if_trained
    def prune(self, top_k=10, max_impurity=0.10):
        """
        Prunes and returns the best student model explanation from the list of _students_by_iter.

        Parameters
        ----------
        top_k
        max_impurity
        """
        return top_k_prune(self._best_student, top_k=top_k, max_impurity=max_impurity)


class ClassificationTrustee(Trustee):
    """
    Implements the Trust-oriented Decision Tree Extraction (Trustee) algorithm to train
    a student Decision Tree Classifier based on observations from an Expert classification model.
    """

    def __init__(self, expert, logger=None):
        """
        Classification Trustee constructor

        Parameters
        ----------
        expert
            The ML blackbox model to analyze.
        student_class
            Class of student to train based on blackbox model predictions
        logger (optional)
            A logger object
        """
        super().__init__(expert, student_class=DecisionTreeClassifier, logger=logger)

    def _score(self, y_true, y_pred, average="macro"):
        """
        F1-score function for classification student models

        Parameters
        ----------
        y_true
        y_pred
        """
        return f1_score(y_true, y_pred, average=average)


class RegressionTrustee(Trustee):
    """
    Implements the Trust-oriented Decision Tree Extraction (Trustee) algorithm to train a
    student Decision Tree Regressor based on observations from an Expert regression model.
    """

    def __init__(self, expert, logger=None):
        """
        Regression Trustee constructor

        Parameters
        ----------
        expert
            The ML blackbox model to analyze.
        student_class
            Class of student to train based on blackbox model predictions
        logger (optional)
            A logger object
        """
        super().__init__(expert=expert, student_class=DecisionTreeRegressor, logger=logger)

    def _score(self, y_true, y_pred):
        """
        R2-score function for regression student models

        Parameters
        ----------
        y_true
        y_pred
        """
        return r2_score(y_true, y_pred)
