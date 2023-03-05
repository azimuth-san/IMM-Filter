import numpy as np
from scipy.stats import multivariate_normal


class IMMFilter():
    """IMM filter."""

    def __init__(self, kalman_filters, mode_proba, transition_mat):

        self.num_f = len(kalman_filters)
        self.kalman_filters = kalman_filters

        self.mode_proba = mode_proba
        self.transition_mat = transition_mat
        self.mixing_proba = np.zeros((self.num_f, self.num_f))

        self.ndim_aug = max([kf.x_post.shape[0] for kf in self.kalman_filters])
        self.x_post = np.zeros(self.ndim_aug)
        self.P_post = np.zeros((self.ndim_aug, self.ndim_aug))

        self.likelihood = np.zeros(self.num_f)

    def init_posterior(self, x=None, P=None):
        """Initialize the postrerior distribution of each kalman filter."""

        if x is not None:
            if isinstance(x, (list, tuple)):
                for x_, kf in zip(x, self.kalman_filters):
                    assert kf.x_post.shape == x_.shape
                    kf.init_posterior(x=x)
            else:
                for kf in self.kalman_filters:
                    ndim = kf.x_post.shape[0]
                    assert x.shape[0] >= ndim
                    kf.init_posterior(x=np.copy(x[:ndim]))

        if P is not None:
            if isinstance(P, (list, tuple)):
                for P_, kf in zip(P, self.kalman_filters):
                    assert kf.P_post.shape == P_.shape
                    kf.init_posterior(P=P_)
            else:
                for kf in self.kalman_filters:
                    ndim = kf.P_post.shape[0]
                    assert P.shape[0] >= ndim
                    kf.init_posterior(P=np.copy(P[:ndim, :ndim]))

    def update_mixing_probability(self):
        """Update the mixing probability."""

        c_bar = self.mode_proba @ self.transition_mat
        for i in range(self.num_f):
            for j in range(self.num_f):
                self.mixing_proba[i, j] = self.transition_mat[i, j] \
                                        * self.mode_proba[i] / c_bar[j]

    def update_mode_probability(self):
        """Update the mode probability."""

        c_bar = self.mode_proba @ self.transition_mat
        c = np.sum(c_bar * self.likelihood)
        self.mode_proba = c_bar * self.likelihood / c

    def mixing(self):
        """Mix the prior distributions of the kalman filters."""

        for j in range(self.num_f):
            x_prior = np.zeros(self.ndim_aug)
            P_prior = np.zeros((self.ndim_aug, self.ndim_aug))

            for i in range(self.num_f):
                kf = self.kalman_filters[i]
                ndim = kf.x_prior.shape[0]
                x_prior[:ndim] += kf.x_prior * self.mixing_proba[j, i]

            for i in range(self.num_f):
                kf = self.kalman_filters[i]
                ndim = kf.x_prior.shape[0]
                dx = np.expand_dims(kf.x_prior - x_prior[:ndim], axis=1)
                P_prior[:ndim, :ndim] += self.mixing_proba[j, i] * (kf.P_prior + dx @ dx.T)

            ndim = self.kalman_filters[j].x_prior.shape[0]
            self.kalman_filters[j].x_prior = x_prior[:ndim]
            self.kalman_filters[j].P_prior = P_prior[:ndim, :ndim]

    def estimate_each(self, t, z, u_prev):
        """Estimate each kalman filter's state."""

        for kf in self.kalman_filters:
            kf.estimate(t, z, u_prev)

    def update_likelihood(self, z):
        """Update the likelihoods."""

        for i in range(self.num_f):
            mean = self.kalman_filters[i].z_prior
            cov = self.kalman_filters[i].Pz_prior
            self.likelihood[i] = multivariate_normal(mean, cov).pdf(z)

    def update_posterior(self):
        """Update the mean and covarance of the posterior distribution."""

        self.x_post.fill(0.)
        self.P_post.fill(0.)

        for i in range(self.num_f):
            kf = self.kalman_filters[i]
            ndim = kf.x_post.shape[0]
            self.x_post[:ndim] += kf.x_post * self.mode_proba[i]

        for i in range(self.num_f):
            kf = self.kalman_filters[i]
            ndim = kf.x_post.shape[0]
            dx = np.expand_dims(kf.x_post - self.x_post[:ndim], axis=1)
            self.P_post[:ndim, :ndim] += self.mode_proba[i] * (kf.P_post + dx @ dx.T)

    def estimate(self, t, z, u_prev):
        """Estimate the state."""

        self.update_mixing_probability()

        self.mixing()

        self.estimate_each(t, z, u_prev)

        self.update_likelihood(z)

        self.update_mode_probability()

        self.update_posterior()

        return self.x_post, self.P_post
