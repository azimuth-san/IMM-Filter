import numpy as np
from .linear_state_space_model import LinearStateSpaceModel


class ConstantVelocity(LinearStateSpaceModel):
    """Linear constant velocity model.

    State space model of the plant.
    x[t+1] = F[t]*x[t] + G[t]*u[t] + L[t]*w[t]
    z[t] = H[t]*x[t] + M[t]*v[t]

    x: state, [x1, vx1, x2, vx2]
    z: output, [x1, x2]
    u: control input
    w: system noise
    v: observation noise
    """

    # # demensions
    NDIM = {
        'x': 4,  # state
        'z': 2,  # output
        'u': 0,  # control input
        'w': 2,  # process noise
        'v': 2,  # observation noise
    }

    def __init__(self, dt=0.1):

        self.F = np.array([[1, dt, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, dt],
                           [0, 0, 0, 1]])

        self.L = np.array([[0.5*(dt**2), 0],
                           [dt, 0],
                           [0, 0.5*(dt**2)],
                           [0, dt]])

        self.H = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0]])

        self.M = np.eye(2)

    def state_equation(self, t, x, u=0, w=np.zeros(2)):
        """Sate equation.

        x[t+1] = F[t]*x[t] + G[t]*u[t] + L[t]*w[t],
        x: state, [x1, vx1, x2, vx2]
        """
        x_next = self.F @ x + self.L @ w
        return x_next

    def observation_equation(self, t, x, v=np.zeros(2)):
        """Observation equation.

        z[t] = H[t]*x[t] + M[t]*v[t],
        x: state, [x1, vx1, x2, vx2]
        z: output, [x1, x2]
        """
        z = self.H @ x + self.M @ v
        return z

    def Ft(self, t):
        """x[t+1] = F[t]*x[t] + G[t]*u[t] + L[t]*w[t].

        return F[t].
        """
        return self.F

    def Lt(self, t):
        """x[t+1] = F[t]*x[t] + G[t]*u[t] + L[t]*w[t].

        return L[t].
        """
        return self.L

    def Ht(self, t):
        """z[t] = H[t]*x[t] + M[t]*v[t].

        return H[t].
        """
        return self.H

    def Mt(self, t):
        """z[t] = H[t]*x[t] + M[t]*v[t].

        return M[t] .
        """
        return self.M
