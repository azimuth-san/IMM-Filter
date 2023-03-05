from .state_space_model import StateSpaceModel


class NonlinearStateSpaceModel(StateSpaceModel):
    """Non-Linear time varying system.

    State space model of the plant to be estimated.
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

    def Jfx(self, t, x):
        """The Jacobian of the system model.

        x[t+1] = f(x[t], u[t], w[t], t),
        return (df/dx)(x).
        """
        raise NotImplementedError

    def Jfw(self, t, x):
        """The Jacobian of the state equation.

        x[t+1] = f(x[t], u[t], w[t], t),
        return (df/dw)(x).
        """
        raise NotImplementedError

    def Jhx(self, t, x):
        """The Jacobian of the observation equation.

        z[t] = h(x[t], v[t], t),
        return (dh/dx)(x).
        """
        raise NotImplementedError

    def Jhv(self, t, x):
        """The Jacobian of the observation equation.

        z[t] = h(x[t], v[t], t),
        return (dh/dv)(x).
        """
        raise NotImplementedError

    def Lt(self, t):
        """x[t+1] = f(x[t], u[t], t) + L[t] * w[t]

        In case of system noise is additive, return L[t].
        """
        raise NotImplementedError

    def Mt(self, t):
        """z[t] = h(x[t], t) + M[t] * v[t]

        In case of observatoin noise is additive, return M[t].
        """
        raise NotImplementedError
