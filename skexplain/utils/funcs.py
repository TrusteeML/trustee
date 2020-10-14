
from scipy.stats import norm, truncnorm

# Calculates integral under a 1-dimensional Gaussian with specified mean,
# standard deviation, and between the two ranges specified
# parameters/returns:
#  mean : float, mean of distribution
#  std: float, standard deviation of distribution
#  lower : float | -np.inf, lower limit of integral
#  upper : float | np.inf, upper limit of integral
#  return : float, area between the two limits


def integrate_truncated(mean, std, lower, upper):
    rv = norm(loc=mean, scale=std)
    if rv.cdf(upper) - rv.cdf(lower) == 0:
        print("equals zero", upper, lower, mean, std)
    return rv.cdf(upper) - rv.cdf(lower)
