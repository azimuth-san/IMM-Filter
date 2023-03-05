from abc import abstractmethod, ABC
import numpy as np


class KalmanBehavior(ABC):
    """Kalman filter's behavior."""

    def __init__(self, model, covar_w, covar_v):

        self.model = model
        self.covar_w = covar_w
        self.covar_v = covar_v
        self.gain = None

    @abstractmethod
    def compute_prior(self, t, x_post, P_post, u):
        """Compute x[t+1|t] and P[t+1|t]. """
        pass

    @abstractmethod
    def compute_z_prior(self, t, x_prior, P_prior):
        """Compute z[t+1|t] and Pz[t+1|t]."""
        pass

    def _update_gain(self, Pxz, Pzz):
        """Update the kalman gain. """
        self.gain = Pxz @ np.linalg.inv(Pzz)

    def compute_posterior(self, x_prior, P_prior, z, z_prior, Pz_prior):
        """Compute x[t|t] and P[t|t]."""
        x_post = x_prior + self.gain @ (z - z_prior)
        P_post = P_prior - self.gain @ Pz_prior @ self.gain.T

        return x_post, P_post


class LinearKalmanBehavior(KalmanBehavior):
    """Linear Kalman filter's behavior.

    The LKF estimates the state variable of a plant.
    The state space model of the plant is below.
    x[t+1] = F[t]*x[t] + G[t]*u[t] + L[t]*w[t]
    z[t] = H[t]*x[t] + M[t]*v[t]

    x: state
    z: output
    u: control input
    w: system noise
    v: observation noise
    """

    def compute_prior(self, t, x_post, P_post, u):
        """Compute x[t+1|t] and P[t+1|t]. """

        F = self.model.Ft(t)
        L = self.model.Lt(t)
        Q = self.covar_w

        # update the prior.
        # state_equation : f(t, x[t], u[t], w[t])
        x_prior = self.model.state_equation(t, x_post, u)
        P_prior = F @ P_post @ F.T + L @ Q @ L.T

        return x_prior, P_prior

    def compute_z_prior(self, t, x_prior, P_prior):
        """Compute z[t+1|t] and Pz[t+1|t]."""

        H = self.model.Ht(t)
        M = self.model.Mt(t)
        R = self.covar_v

        Pxz = P_prior @ H.T
        z_prior = self.model.observation_equation(t, x_prior)
        Pz_prior = H @ P_prior @ H.T + M @ R @ M.T
        self._update_gain(Pxz, Pz_prior)

        return z_prior, Pz_prior


class ExtendedKalmanBehavior(KalmanBehavior):
    """Extended Kalman filter's behavior.

    The EKF estimates the state variable of a plant.
    The state space model of the plant is below.
    x[t+1] = f(t, x[t], u[t], w[t])
    z[t] = h(t, x[t], v[t])

    f: state equation
    h: observation equation
    x: state
    z: output
    u: control input
    w: system noise
    v: observation noise
    """

    def compute_prior(self, t, x_post, P_post, u):
        """Compute x[t+1|t] and P[t+1|t]."""

        F = self.model.Jfx(t, x_post)  # df/dx
        L = self.model.Jfw(t, x_post)  # df/dw
        Q = self.covar_w

        # predict the state and state covariance.
        # state_equation : f(t, x[t], u[t], w[t])
        x_prior = self.model.state_equation(t, x_post, u)
        P_prior = F @ P_post @ F.T + L @ Q @ L.T

        return x_prior, P_prior

    def compute_z_prior(self, t, x_prior, P_prior):
        """Compute z[t+1|t] and Pz[t+1|t]."""

        H = self.model.Jhx(t, x_prior)  # dh/dx
        M = self.model.Jhv(t, x_prior)  # dh/dv
        R = self.covar_v

        # update the kalman gain.
        z_prior = self.model.observation_equation(t, x_prior)
        Pxz = P_prior @ H.T
        Pz_prior = H @ Pxz + M @ R @ M.T
        self._update_gain(Pxz, Pz_prior)

        return z_prior, Pz_prior


class UnscentedendKalmanBehavior(KalmanBehavior):
    """Unscented Kalman filter's behavior.

    The UKF estimates the state variable of a plant.
    The state space model of the plant is below.

    x[t+1] = f(t, x[t], u[t]) + L[t] * w[t]
    z[t] = h(t, x[t]) + M[t] * v[t]

    f: state equation
    h: observation equation
    x: state
    z: output
    u: control input
    w: system noise (additive)
    v: observation noise (additive)
    """

    def __init__(self, model, covar_w, covar_v,
                 kappa=0, decompose_method='cholesky'):

        super().__init__(model, covar_w, covar_v)

        # weights of sigma points
        n = self.model.NDIM['x']
        self.weights = np.zeros(2 * n + 1)
        self.weights[0] = kappa / (n + kappa)
        self.weights[1:] = 1 / (2 * (n + kappa))
        self.kappa = kappa

        self.decompose_method = decompose_method.lower()

    def _compute_sigma_points(self, x_center, P):
        """Compute sigma points."""

        if self.decompose_method == 'cholesky':
            # P_sqr @ P_sqr.T is equal to P
            P_sqr = np.linalg.cholesky(P)
        elif self.decompose_method == 'svd':
            U, S, Vh = np.linalg.svd(P)
            P_sqr = U @ np.diag(np.sqrt(S))

        n = P_sqr.shape[0]
        num_points = 2 * n + 1

        x_sigmas = np.zeros((num_points, x_center.shape[0]))
        x_sigmas[0] = x_center
        # add the new axis for broadcast.
        # x_center.shape: (d,) -> (1, d) -> (n, d)
        x_sigmas[1:n+1] = (x_center + np.sqrt(n + self.kappa) * P_sqr.T)
        x_sigmas[n+1:] = (x_center - np.sqrt(n + self.kappa) * P_sqr.T)

        return x_sigmas

    def compute_prior(self, t, x_post, P_post, u):
        """Compute x[t+1|t] and P[t+1|t]."""

        x_sigmas = self._compute_sigma_points(x_post, P_post)
        x_sigmas_next = np.zeros_like(x_sigmas)
        for i in range(self.weights.shape[0]):
            x_sigmas_next[i] = self.model.state_equation(t, x_sigmas[i], u)

        # update the prior.
        x_prior = np.sum(self.weights[:, np.newaxis] * x_sigmas_next, axis=0)

        L = self.model.Lt(t)
        x_error = x_prior - x_sigmas_next
        P_prior = (self.weights * x_error.T) @ x_error + L @ self.covar_w @ L.T

        return x_prior, P_prior

    def compute_z_prior(self, t, x_prior, P_prior):
        """Compute z[t+1|t] and Pz[t+1|t]."""

        x_sigmas = self._compute_sigma_points(x_prior, P_prior)
        z_sigmas = np.zeros((self.weights.shape[0], self.model.ndim('z')))
        for i in range(self.weights.shape[0]):
            z_sigmas[i] = self.model.observation_equation(t, x_sigmas[i])
        z_prior = np.sum(self.weights[:, np.newaxis] * z_sigmas, axis=0)

        M = self.model.Mt(t)
        # add the new axis for broadcast.
        # y.shape: (d,) -> (d, 1) -> (d, num_sigma_points)
        z_error = z_sigmas - z_prior
        Pz_prior = (self.weights * z_error.T) @ z_error + M @ self.covar_v @ M.T

        x_error = x_sigmas - x_prior
        Pxz = (self.weights * x_error.T) @ z_error
        self._update_gain(Pxz, Pz_prior)

        return z_prior, Pz_prior
