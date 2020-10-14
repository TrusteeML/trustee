import random
from abc import ABC
from copy import deepcopy

import numpy

from skexplain.dists import CategoricalGaussianMixtureDist
from sklearn.metrics import classification_report, f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from skmultiflow.trees import HoeffdingTreeClassifier


class Endear(ABC):
    """
        Implements the ENhanced DatasEt AggRegation (Endear) algorithm to train
        student model based on observations from an Expert model sampled from a GaussianMixtureDistribution.
    """

    def __init__(self, expert, student_class, is_cls=False, dist_class=CategoricalGaussianMixtureDist, logger=None):
        """ Init method """
        self.log = logger.log if logger else print
        self.expert = expert
        self.students = []
        self.dist = None
        self.dist_class = dist_class
        self.student_class = student_class
        self.is_cls = is_cls

    def score(self, y_true, y_pred):
        pass

    def fit(self, X, y, cat_feat_inds=[], numeric_feat_inds=[], max_leaf_nodes=None, max_depth=None, max_iter=100, num_samples=2000, n_components=10, verbose=False):
        """ Trains Decision Tree Regressor to imitate Expert model. """

        if verbose:
            self.log("Initializing training dataset using {} as expert model".format(self.expert))
            # self.log("Sampling {} points from {}".format(len(X), self.dist.__class__.__name__))

        features = X
        targets = self.expert.predict(X)

        if verbose:
            self.log("Initializing {} as distribution to sample points from".format(self.dist_class.__name__))

        self.dist = self.dist_class(cat_feat_inds, numeric_feat_inds)
        self.dist.fit(features, targets, n_components=n_components, is_cls=self.is_cls)

        student = self.student_class(random_state=0, max_leaf_nodes=max_leaf_nodes, max_depth=max_depth)

        if verbose:
            self.log("Expert model score: {}".format(self.score(y, targets)))
            self.log("Initializing Endear loop with {} iterations".format(max_iter))

        # Endear loop
        for i in range(max_iter):
            if verbose:
                self.log("#" * 10, "Iteration {}/{}".format(i, max_iter), "#" * 10)

            # Step 1: Sample predictions from training dataset
            #   - Some distribution? Or do I just use the dataset? Why not use all of it?
            #   - Sample randomly every time to generate different trees?
            if verbose:
                # self.log("Sampling {} points from {}".format(num_samples, self.dist.__class__.__name__))
                self.log("Sampling {} points from {} dataset".format(num_samples, len(features)))

            #
            # X_iter = self.dist.sample([], num_samples)
            # y_iter = self.expert.predict(X_iter)

            X_train, y_train = zip(*random.sample(list(zip(features, targets)), num_samples))
            X_test = self.dist.sample(len(X_train))
            y_test = self.expert.predict(X_test)

            # Step 2: Traing DecisionTreeRegressor with sampled data
            student.fit(X_train, y_train)
            student_pred = student.predict(X_test)

            if verbose:
                self.log("Student model {} trained with depth {} and {} leaves:"
                         .format(i, student.get_depth(), student.get_n_leaves()))
                self.log("Student model score: {}".format(self.score(y_test, student_pred)))

            # Step 3: Use expert model predictions to aggregate original dataset
            features = numpy.append(features, X_test, axis=0)
            targets = numpy.append(targets, y_test, axis=0)

            # Step 4: Calculate reward based on Decistion Tree Classifier fidelity to the Expert model
            reward = 0
            if verbose:
                reward = self.score(y_test, student_pred)
                self.log("Student model {} fidelity: {}".format(i, reward))

            # Step 5: Somehow incorporate that reward onto training process?
            # - Maybe just store the highest reward possible and use that as output?
            self.students.append((deepcopy(student), reward, i))

    def explain(self):
        """ Returns explainable model that best imitates Expert model, based on calculated rewards. """
        return max(self.students, key=lambda item: item[1])

    def get_students(self):
        """ Returns list of all (student, reward) obtained during the training process. """
        return self.students


class ClassificationEndear(Endear):
    """
        Implements the ENhanced DatasEt AggRegation (Endear) algorithm to train a student Decision Tree Classifier
        based on observations from an Expert classification model.
    """

    def __init__(self, expert, logger=None):
        """ Init method """
        super().__init__(expert, student_class=DecisionTreeClassifier, is_cls=True, logger=logger)

    def score(self, y_true, y_pred, average='macro'):
        """ Score function for student models """
        self.log("\n{}".format(classification_report(y_true, y_pred, digits=3)))
        return f1_score(y_true, y_pred, average=average)


class RegressionEndear(Endear):
    """
        Implements the ENhanced DatasEt AggRegation (Endear) algorithm to train a student Decision Tree Regressor
        based on observations from an Expert regression model.
    """

    def __init__(self, expert, logger=None):
        """ Init method """
        super().__init__(expert=expert, student_class=DecisionTreeRegressor, logger=logger)

    def score(self, y_true, y_pred):
        """ Score function for student models """
        return r2_score(y_true, y_pred)
