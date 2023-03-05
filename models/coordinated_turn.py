import numpy as np
from .nonlinear_state_space_model import NonlinearStateSpaceModel


class CoordinatedTurn(NonlinearStateSpaceModel):
    """Coordinated turn model.

    State space model of the plant to be estimated.
    x[t+1] = f(t, x[t], u[t], w[t])
    z[t] = h(t, x[t], v[t])

    f: state equation
    h: observation equation
    x: state, [x1, vx1, x2, vx2, omega]
    z: output, [x1, x2]
    u: control input
    w: system noise
    v: observation noise
    """

    # dimensions
    NDIM = {'x': 5,  # state
            'z': 2,  # output
            'u': 0,  # control input
            'w': 3,  # system noise
            'v': 2   # observation noise
            }

    def __init__(self, dt=0.1):
        self.dt = dt  # sampling time
        self.eps = 1e-14

    def compute_F(self, omega, dt):

        if np.abs(omega) < self.eps:
            return np.array([[1., dt, 0., 0., 0.],
                             [0., 1., 0., 0., 0.],
                             [0., 0., 1., dt, 0.],
                             [0., 0., 0., 1., 0.],
                             [0., 0., 0., 0., 1.]])

        Ft = np.zeros((5, 5))
        Ft[0, 0] = 1
        Ft[0, 1] = np.sin(omega*dt) / omega
        Ft[0, 3] = -(1 - np.cos(omega*dt)) / omega

        Ft[1, 1] = np.cos(omega*dt)
        Ft[1, 3] = -np.sin(omega*dt)

        Ft[2, 1] = (1 - np.cos(omega*dt)) / omega
        Ft[2, 2] = 1
        Ft[2, 3] = np.sin(omega*dt) / omega

        Ft[3, 1] = np.sin(omega*dt)
        Ft[3, 3] = np.cos(omega*dt)

        Ft[4, 4] = 1

        return Ft

    def state_equation(self, t, x, u=0, w=np.zeros(NDIM['w'])):
        """Calculate the state equation.

        x[t+1] = f(t, x[t], u[t], w[t])
        x: state, [x1, vx1, x2, vx2, omega]
        """

        dt = self.dt
        omega = x[4]
        Ft = self.compute_F(omega, dt)

        Lt = np.array([[0.5*(dt**2), 0., 0.],
                       [dt, 0., 0.],
                       [0., 0.5*(dt**2), 0.],
                       [0., dt, 0.],
                       [0., 0., dt]])

        return Ft @ x + Lt @ w

    def observation_equation(self, t, x, v=np.zeros(NDIM['v'])):
        """Calculate the observation equation.

        y[t] = h(t, x[t], v[t])
        x: state, [x1, vx1, x2, vx2, omega]
        z: output, [x1, x2]
        """
        Ht = np.array([[1., 0., 0., 0., 0.],
                       [0., 0., 1., 0., 0.]])
        return Ht @ x + v

    def Jfx(self, t, x):
        """The Jacobian of the system model.

        x[t+1] = f(x[t], u[t], w[t], t),
        return (df/dx)(x).
        """
        dt = self.dt
        vx, vy, omega = x[1], x[3], x[4]

        if np.abs(omega) < self.eps:
            return np.array([[1., dt, 0., 0., -0.5*(dt**2)*vy],
                             [0., 1., 0., 0., -dt*vy],
                             [0., 0., 1., dt, 0.5*(dt**2)*vx],
                             [0., 0., 0., 1., dt*vx],
                             [0., 0., 0., 0., 1.]])

        J = np.zeros((5, 5))
        J[0, 0] = 1
        J[0, 1] = np.sin(omega*dt) / omega
        J[0, 3] = -(1 - np.cos(omega*dt)) / omega
        J[0, 4] = np.cos(omega*dt)*dt*vx/omega - np.sin(omega*dt)*vx/(omega**2) \
            - np.sin(omega*dt)*dt*vy/omega - (-1 + np.cos(omega*dt))*vy/(omega**2)

        J[1, 1] = np.cos(omega*dt)
        J[1, 3] = -np.sin(omega*dt)
        J[1, 4] = -np.sin(omega*dt)*dt*vx - np.cos(omega*dt)*dt*vy

        J[2, 1] = (1 - np.cos(omega*dt)) / omega
        J[2, 2] = 1
        J[2, 3] = np.sin(omega*dt) / omega
        J[2, 4] = np.sin(omega*dt)*dt*vx/omega - (1 - np.cos(omega*dt))*vx/(omega**2) \
            + np.cos(omega*dt)*dt*vy/omega - np.sin(omega*dt)*vy/(omega**2)

        J[3, 1] = np.sin(omega*dt)
        J[3, 3] = np.cos(omega*dt)
        J[3, 4] = np.cos(omega*dt)*dt*vx - np.sin(omega*dt)*dt*vy

        J[4, 4] = 1

        return J

    def Jfw(self, t, x):
        """The Jacobian of the state equation.

        x[t+1] = f(x[t], u[t], w[t], t),
        return (df/dw)(x).
        """
        return self.Lt(t)

    def Jhx(self, t, x):
        """The Jacobian of the observation equation.

        z[t] = h(x[t], v[t], t),
        return (dh/dx)(x).
        """
        return np.array([[1., 0., 0., 0., 0.],
                         [0., 0., 1., 0., 0.]])

    def Jhv(self, t, x):
        """The Jacobian of the observation equation.

        z[t] = h(x[t], v[t], t),
        return (dh/dv)(x).
        """
        return self.Mt(t)

    def Lt(self, t):
        """x[t+1] = f(x[t], u[t], t) + L[t] * w[t]

        In case of system noise is additive, return L[t].
        """
        dt = self.dt
        Lt = np.array([[0.5*(dt**2), 0., 0.],
                       [dt, 0., 0.],
                       [0., 0.5*(dt**2), 0.],
                       [0., dt, 0.],
                       [0., 0., dt]])
        return Lt

    def Mt(self, t):
        """z[t] = h(x[t], t) + M[t] * v[t]

        In case of observatoin noise is additive, return M[t].
        """
        return np.eye(2)
