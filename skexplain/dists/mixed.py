import numpy as np

from sklearn.dists.categorical import CategoricalDist, GaussianMixtureDist

# Fields:
#  cat_feat_inds: list of (# of categorical features) lists, each of (# categories
#       per categorical feature) listing the column indices that correspond to that
#       feature in X)
#  numeric_feat_inds: list of column indices that correspond to numeric features in X


class CategoricalGaussianMixtureDist:
    def __init__(self, X, cat_feat_inds, numeric_feat_inds, num_components, logger=None):
        self.log = logger.log if logger else print
        self.cat_feat_inds = cat_feat_inds
        self.numeric_feat_inds = numeric_feat_inds
        self.categorical_dist = []
        for feature_ind in cat_feat_inds:
            self.categorical_dist.append(CategoricalDist(X[:, feature_ind]))
        if len(numeric_feat_inds) == 0:
            self.gm_dist = None
        else:
            self.gm_dist = GaussianMixtureDist(X[:, numeric_feat_inds], num_components, logger=logger)
        self.num_cols = np.shape(X)[1]

    def sample(self, cons, nPts):
        xs = np.empty((nPts, self.num_cols))
        limits = CategoricalGaussianMixtureDist.consolidate_constraints(cons, self.num_cols)

        for feature in range(len(self.cat_feat_inds)):
            catDist = self.categorical_dist[feature]
            cat_feat_ind = self.cat_feat_inds[feature]
            cat_xs = catDist.sample([limits[cat_feature] for cat_feature in cat_feat_ind], nPts)
            xs[:, cat_feat_ind] = cat_xs
        if not self.gm_dist is None:
            numeric_xs = self.gm_dist.sample([limits[nF] for nF in self.numeric_feat_inds], nPts)
            xs[:, self.numeric_feat_inds] = numeric_xs
        return xs

    def mass(self, cons):
        limits = CategoricalGaussianMixtureDist.consolidate_constraints(cons, self.num_cols)
        mass = 1
        for feature in range(len(self.cat_feat_inds)):
            cat_feat_ind = self.cat_feat_inds[feature]
            catDist = self.categorical_dist[feature]
            mass *= catDist.mass([limits[cat_feature] for cat_feature in cat_feat_ind])
        if not self.gm_dist is None:
            mass *= self.gm_dist.mass([limits[nF] for nF in self.numeric_feat_inds])
        return mass

    # Takes constraints of type C and turns it into a list of lists, two
    # constraints per dimension
    # type params:
    #  C = [(InternalNode, bool)] (i.e., constraints are lists of internal nodes, and
    #                              a boolean indicating left, x[ind] <= thresh (True)
    #                              or right, x[ind] > thresh (False))
    #
    # params/returns:
    #  cons: C, constraints
    #  num_cols: int, dimension of data
    #  returns: [[float | np.inf | -np.inf]], each (sub)list specifies the two
    #    (upper and lower) bounds of the constraints of the constrained region for that dimension
    @staticmethod
    def consolidate_constraints(cons, num_cols):
        # make a num_cols-dimensional list of lists, each of which specifies an
        # upper and lower bound of the constrained region for that dimension
        limits = [[-np.inf, np.inf] for i in range(num_cols)]
        for con in cons:
            ind = con[0].ind
            thresh = con[0].thresh
            if con[1]:
                # upper limit
                cur_upper_lim = limits[ind][1]
                if thresh < cur_upper_lim:
                    limits[ind][1] = thresh
            else:
                # lower limit
                cur_lower_lim = limits[ind][0]
                if thresh > cur_lower_lim:
                    limits[ind][0] = thresh
        return limits
