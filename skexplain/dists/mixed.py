import numpy as np

from skexplain.dists import CategoricalDist, GaussianMixtureDist

# Fields:
#  cat_feat_inds: list of (# of categorical features) lists, each of (# categories
#       per categorical feature) listing the column indices that correspond to that
#       feature in X)
#  numeric_feat_inds: list of column indices that correspond to numeric features in X


class CategoricalGaussianMixtureDist:
    def __init__(self, cat_feat_inds, numeric_feat_inds, logger=None):
        self.log = logger.log if logger else print
        self.cat_feat_inds = cat_feat_inds
        self.numeric_feat_inds = numeric_feat_inds
        self.n_cols = 0
        self.gm_dist = GaussianMixtureDist(logger=logger) if len(self.numeric_feat_inds) != 0 else None
        self.categorical_dists = []
        for feature_ind in cat_feat_inds:
            self.categorical_dists.append(CategoricalDist(logger=logger))

    def fit(self, X, y, n_components=1, is_cls=False):
        self.log("Fitting CategoricalGaussianMixtureDist")
        self.n_cols = np.shape(X)[1]

        for feature in range(len(self.cat_feat_inds)):
            self.categorical_dists[feature].fit(X[:, self.cat_feat_inds[feature]], y)

        if self.gm_dist:
            self.gm_dist.fit(X[:, self.numeric_feat_inds], y, n_components=n_components, is_cls=is_cls)

    def sample(self, n_points):
        xs = np.empty((n_points, self.n_cols))
        for feature in range(len(self.cat_feat_inds)):
            cat_dist = self.categorical_dists[feature]
            cat_feat_ind = self.cat_feat_inds[feature]
            cat_xs = cat_dist.sample(n_points)
            xs[:, cat_feat_ind] = cat_xs
        if not self.gm_dist is None:
            numeric_xs = self.gm_dist.sample(n_points)
            xs[:, self.numeric_feat_inds] = numeric_xs[0]
        return xs

    def mass(self):
        mass = 1
        for feature in range(len(self.cat_feat_inds)):
            cat_feat_ind = self.cat_feat_inds[feature]
            cat_dist = self.categorical_dists[feature]
            mass *= cat_dist.mass()
        if not self.gm_dist is None:
            mass *= self.gm_dist.mass()
        return mass
