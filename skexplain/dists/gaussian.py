import numpy as np

from scipy.stats import truncnorm
from skexplain.utils.funcs import integrate_truncated
from sklearn.metrics import classification_report, f1_score, r2_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

# Fits a Gaussian mixture model with the given number of components.
#
# fields:
#  X : np.array([nRows, n_cols]) (the example points)
#  n_components : int, number of components to fit


class GaussianMixtureDist:
    def __init__(self, logger=None):
        self.log = logger.log if logger else print
        self.gmm = None
        self.gmms = []
        self.n_cols = 0
        self.n_components = 1
        # fit a Gaussian mixture model

    def fit(self, X, y, n_components=1, is_cls=False):
        self.n_components = n_components
        self.n_cols = np.shape(X)[1]
        self.log('Fitting Gaussian mixture with {} components and {} columns ...'.format(
            str(self.n_components), str(self.n_cols)))
        self.X = X
        self.y = y

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

        reg_covars = {
            'spherical': 1e-6,
            'diag': 1e-1,
            'tied': 10,
            'full': 100
        }

        # fitting different covariances types to find out best
        self.gmms = [GaussianMixture(n_components=self.n_components, covariance_type=cov_type, reg_covar=reg_covars[cov_type])
                     for cov_type in ['spherical', 'diag', 'tied', 'full']]

        highest_score = np.NINF
        for gmm in self.gmms:
            gmm.fit(X_train, y_train)

            y_pred = gmm.predict(X_test)
            model_score = f1_score(y_test, y_pred, average="macro") if is_cls else r2_score(y_test, y_pred)
            self.log("covariances_ shape", gmm.covariances_.shape)
            self.log("covariance_type", gmm.covariance_type, "score", model_score)
            self.log("BIC", gmm.bic(X))
            if model_score > highest_score:
                highest_score = model_score
                self.gmm = gmm

        self.log("covariance_type", self.gmm.covariance_type)
        self.log("covariances_", self.gmm.covariances_)
        self.log("converged_", self.gmm.converged_)
        self.log("BIC", self.gmm.bic(X))
        self.log('Done!')

        # Integrate the probability density

    def mass(self):
        probabilities = self.predict_proba(self.X)
        return np.sum(probabilities)

    # Samples a random point from the fitted mixture model
    # returns: np.array([n_points, self.n_cols])
    def sample(self, n_points):
        return self.gmm.sample(n_points)
