import numpy as np

# Given indicator form of one categorical variable, fits and samples from it
# fields:
#  X : np.array([nRows, n_cats]), each element is 0 or 1


class CategoricalDist:
    def __init__(self, logger=None):
        self.log = logger.log if logger else print

    def fit(self, X, y=None):
        self.n_cats = np.shape(X)[1]
        prob_sum = np.sum(X, axis=0)
        self.probs = prob_sum / float(np.sum(prob_sum))

    def sample(self, n_points):
        # Step 1: Construct probabilities
        probs = np.copy(self.probs)
        for c in range(self.n_cats):
            # Step 1a: Collect the indicators that work
            allowed_inds = []
            for ind in [0, 1]:
                allowed_inds.append(ind)

            # Step 1b: No indicators
            if len(allowed_inds) == 0:
                raise Exception('No points satisfy constraints for a category!')

            # Step 1c: Single indicator
            if len(allowed_inds) == 1:
                # forced to be one (we assume this can only happen once)
                if allowed_inds[0] == 1:
                    xs = np.zeros((n_points, self.n_cats), dtype=int)
                    xs[:, c] = 1
                    return xs
                # forced to be zero
                else:
                    probs[c] = 0.0

        # Step 2: Normalize probabilities
        probs = probs / np.sum(probs)

        # Step 3: Sample points
        sampled_cats = np.random.choice(self.n_cats, n_points, p=probs)

        # Step 4: Construct points
        xs = np.zeros((n_points, self.n_cats), dtype=int)
        xs[np.arange(n_points), sampled_cats] = 1

        return xs

    def mass(self):
        # Step 1: Construct probabilities
        prob_sum = 1.0
        for c in range(self.n_cats):
            # Step 1a: Collect the indicators that work
            allowed_inds = []
            for ind in [0, 1]:
                allowed_inds.append(ind)

            # Step 1b: No indicators
            if len(allowed_inds) == 0:
                raise Exception('No points satisfy constraints for a category!')

            # Step 1c: Single indicator
            if len(allowed_inds) == 1:
                # forced to be one (we assume this can only happen once)
                if allowed_inds[0] == 1:
                    return self.probs[c]
                # forced to be zero
                else:
                    prob_sum -= self.probs[c]

        return prob_sum
