import numpy as np


class KalmanFilter():
    """Kalman filter class."""

    def __init__(self, kalman_behavior):

        # kalman filter's behavior
        self.behavior = kalman_behavior

        ndim_x = self.behavior.model.NDIM['x']

        # mean and covariance of prior distribution
        self.x_prior = np.zeros(ndim_x)
        self.P_prior = np.eye(ndim_x)

        # mean and covariance of posterior distribution
        self.x_post = np.zeros(ndim_x)
        self.P_post = np.eye(ndim_x)

        ndim_z = self.behavior.model.NDIM['z']

        # mean and covariance of z distribution
        self.z_prior = np.zeros(ndim_z)
        self.Pz_prior = np.eye(ndim_z)

    def init_posterior(self, x=None, P=None):
        """Initialize the mean and covariance of the postrerion distribution."""

        if x is not None:
            self.x_post = x

        if P is not None:
            self.P_post = P

    def update_prior(self, t, u):
        """Update x[t+1|t] and P[t+1|t]."""

        self.x_prior, self.P_prior = \
            self.behavior.compute_prior(t, self.x_post, self.P_post, u)

    def update_z_prior(self, t):
        """Update z[t+1|t] and Pz[t+1|t]."""

        self.z_prior, self.Pz_prior = \
            self.behavior.compute_z_prior(
                                    t, self.x_prior, self.P_prior)

    def update_posterior(self, z):
        """Update x[t|t] and P[t|t]."""

        self.x_post, self.P_post = \
            self.behavior.compute_posterior(
                    self.x_prior, self.P_prior, z, self.z_prior, self.Pz_prior)

    def estimate(self, t, z, u_prev=0):
        """Estimate the state."""

        self.update_prior(t-1, u_prev)

        self.update_z_prior(t)

        self.update_posterior(z)

        return self.x_post, self.P_post
