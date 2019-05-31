import numpy as np


class DockingUtils(object):
    def __init__(self):
        """
        class containing docking utils docking engine, inheriting after oddt.docking.Docking

        Parameters
        ----------
        engine
        """

    def score_coords(self, x, engine, scoring_func):
        """

        Parameters
        ----------
        x
        engine

        Returns
        -------

        """
        c1 = engine.lig.mutate(x)
        return scoring_func(c1)

    def score_coords_jac(self, x, engine, scoring_func, step=1e-2):
        """

        Parameters
        ----------
        x
        engine
        step

        Returns
        -------

        """
        c1 = engine.lig.mutate(x)
        e1 = scoring_func(c1)
        grad = []
        for i in range(len(x)):
            x_g = x.copy()
            x_g[i] += step  # if i < 3 else 1e-2
            cg = engine.lig.mutate(x_g)
            grad.append(scoring_func(cg))
        return (np.array(grad) - e1) / step

    def rand_mutate_small(self, x, box_size):
        """

        Parameters
        ----------
        x
        box_size

        Returns
        -------

        """
        x = x.copy()
        m = np.random.random_integers(0, len(x) - 5)
        if m == 0:  # do random translation
            x[:3] += np.random.uniform(-1, 1, size=3) / box_size
        elif m == 1:  # do random rotation step
            x[3:6] += self._random_angle(size=3) / 6
        else:  # do random dihedral change
            ix = 6 + m - 2
            x[ix] += self._random_angle() / 6
            x[ix] = np.clip(x[ix], -3.1415, 3.1415)
        x = self.keep_bound(x)
        return x

    def _random_angle(self, size=1):
        if size > 1:
            return np.random.uniform(-np.pi, np.pi, size=size)
        return np.random.uniform(-np.pi, np.pi)

    def keep_bound(self, x):
        """

        Parameters
        ----------
        x

        Returns
        -------

        """
        x[:3] = np.clip(x[:3], -1, 1)
        x[3:] = np.clip(x[3:], -np.pi, np.pi)
        return x

    def rand_mutate_big(self, x):
        """

        Parameters
        ----------
        x

        Returns
        -------

        """
        x = x.copy()
        m = np.random.randint(0, len(x) - 4)
        if m == 0:  # do random translation
            x[:3] = np.random.uniform(-0.3, 0.3, size=3)
        elif m == 1:  # do random rotation step
            x[3:6] = self._random_angle(size=3)
        else:  # do random dihedral change
            ix = 6 + m - 2
            x[ix] = self._random_angle()
        return x
