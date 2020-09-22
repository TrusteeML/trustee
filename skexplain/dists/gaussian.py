import numpy as np

from sklearn.mixture import GaussianMixture

# Fits a Gaussian mixture model with the given number of components.
#
# fields:
#  X : np.array([nRows, num_cols]) (the example points)
#  num_components : int, number of components to fit


class GaussianMixtureDist:
    def __init__(self, X, num_components, logger=None):
        self.log = logger.log if logger else print
        self.gmm = GaussianMixture(n_components=num_components, covariance_type='diag', reg_covar=1e-1)
        self.num_cols = np.shape(X)[1]
        self.num_components = num_components
        # fit a Gaussian mixture model
        self.log('Fitting Gaussian mixture with {} components and {} columns ...'.format(
            str(num_components), str(self.num_cols)), INFO)
        self.gmm.fit(X)
        self.log('Done!', INFO)

    # Compute the probability of each component in the mixture
    def _compute_probs(self, limits):
        probabilities = np.zeros(self.num_components)
        for component in range(self.num_components):
            mean = self.gmm.means_[component]
            cov = self.gmm.covariances_[component]
            weight = self.gmm.weights_[component]
            probabilities[component] = constrained_gaussian_density(mean, cov, limits) * weight
        return probabilities

    # Integrate the probability density given the constraints.
    def mass(self, limits):
        probabilities = self._compute_probs(limits)
        return np.sum(probabilities)

    # Samples a random point from the fitted mixture model
    # given the constraints
    #
    # returns: np.array([nPts, self.num_cols])
    def sample(self, limits, nPts):
        # consolidate constraints
        #limits = consolidateConstraints(cons, self.num_cols)

        # compute probabilities
        probabilities = self._compute_probs(limits)

        # compute mass
        s = np.sum(probabilities)

        self.log(("probabilities per component:", probabilities), DEBUG)
        self.log(("limits:", limits), DEBUG)

        # density is zero (up to rounding) within these constraints
        xs = np.zeros((nPts, self.num_cols))
        if s == 0:
            return xs

        # normalize
        probabilities = probabilities / s

        # Count of how many points to sample
        chosenComponents = np.random.choice(self.num_components, size=nPts, p=probabilities)

        # Sample the points
        curIndex = 0
        for component in range(self.num_components):
            curCount = np.sum(chosenComponents == component)
            nextIndex = curIndex + curCount
            xs[curIndex:nextIndex, :] = sample_trunc_gaussian(
                self.gmm.means_[component], self.gmm.covariances_[component], limits, curCount)
            curIndex = nextIndex

        return xs

    # samples from a multidimensional truncated diagonal-covariance gaussian
    #
    # params/returns:
    #  mean: np.array([num_cols]), mean of Gaussian
    #  cov: np.array([num_cols]), diaganal elements of (assume diagonal) covariance matrix
    #    of Gaussian
    #  limits: [[float | np.inf | -np.inf]], a list of lists, each (sub)list specifies the two (upper and lower)
    #    bounds of the constraints
    #  nPts: number of points to sample
    #  returns: np.array([nPts, num_cols]), samples from the multivariate Gaussian
    @staticmethod
    def sample_trunc_gaussian(mean, cov, limits, nPts):
        num_cols = len(limits)
        samples = np.zeros((num_cols, nPts))
        for i in range(num_cols):
            limit = limits[i]
            mu = mean[i]
            std = np.sqrt(cov[i])
            a, b = (limit[0] - mu) / std, (limit[1] - mu) / std
            samples[i, :] = truncnorm.rvs(a, b, loc=mu, scale=std, size=nPts)
        return samples.transpose()
