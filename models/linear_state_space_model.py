from .state_space_model import StateSpaceModel


class LinearStateSpaceModel(StateSpaceModel):
    """Linear time varying system.

    State space model of the plant.
    x[t+1] = F[t]*x[t] + G[t]*u[t] + L[t]*w[t]
    z[t] = H[t]*x[t] + M[t]*v[t]

    x: state
    z: output
    u: control input
    w: system noise
    v: observation noise
    """

    def Ft(self, t):
        """x[t+1] = F[t]*x[t] + G[t]*u[t] + L[t]*w[t].

        return F[t].
        """
        raise NotImplementedError

    def Gt(self, t):
        """x[t+1] = F[t]*x[t] + G[t]*u[t] + L[t]*w[t].

        return G[t].
        """
        raise NotImplementedError

    def Lt(self, t):
        """x[t+1] = F[t]*x[t] + G[t]*u[t] + L[t]*w[t].

        return L[t].
        """
        raise NotImplementedError

    def Ht(self, t):
        """y[t] = H[t]*x[t] + M[t]*v[t].

        return H[t].
        """
        raise NotImplementedError

    def Mt(self, t):
        """y[t] = H[t]*x[t] + M[t]*v[t].

        return M[t].
        """
        raise NotImplementedError
