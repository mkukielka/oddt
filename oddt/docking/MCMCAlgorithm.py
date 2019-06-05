from enum import Enum
import numpy as np
from scipy.optimize import minimize

from oddt import random_seed
from oddt.docking.dockingUtils import DockingUtils


class OptimizationMethod(Enum):
    SIMPLEX = 1
    NO_OPTIM = 1
    NELDER_MEAD = 2
    LBFGSB = 3


# noinspection PyUnboundLocalVariable
class MCMCAlgorithm(object):
    def __init__(self, engine, scoring_func=None, optim=OptimizationMethod.NELDER_MEAD, optim_iter=10, mc_steps=50,
                 mut_steps=100, seed=None, ):

        self.engine = engine
        self.scoring_func = scoring_func
        if not self.scoring_func:
            self.scoring_func = engine.score
        self.optim = optim
        self.optim_iter = optim_iter
        self.mc_steps = mc_steps
        self.mut_steps = mut_steps
        if seed:
            random_seed(seed)

        self.num_rotors = len(self.engine.rotors)

        self.lig_dict = self.engine.lig_dict
        self.docking_utils = DockingUtils()

    def perform(self):

        x1 = self.generate_rotor_vector()
        c1 = self.engine.lig.mutate(x1)
        e1 = self.scoring_func(c1)
        out = {'score': e1, 'conformation': x1.copy().tolist()}

        for _ in range(self.mc_steps):
            c2, x2 = self.generate_conformation(x1)
            e2 = self.scoring_func(c2)
            e3, x3 = self._optimize(e2, x2)

            delta = e3 - e1

            if delta < 0 or np.exp(-delta) > np.random.uniform():  # Metropolis criterion
                x1 = x3
                if delta < 0:
                    e1 = e3
                    out = {'score': e1, 'conformation': x1.copy().tolist()}

        return out

    def _optimize(self, e2, x2):

        bounds = ((-1., 1.), (-1., 1.), (-1., 1.), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi))
        for i in range(len(self.engine.rotors)):
            bounds += ((-np.pi, np.pi),)
        bounds = np.array(bounds)

        if self.optim == OptimizationMethod.SIMPLEX:
            return e2, x2
        elif self.optim == OptimizationMethod.NELDER_MEAD:
            return self._minimize_nelder_mead(x2)
        elif self.optim == OptimizationMethod.LBFGSB:
            return self._minimize_lbfgsb(bounds, x2)
        return e2, x2

    def _minimize_nelder_mead(self, x2):

        m = minimize(self.docking_utils.score_coords, x2, args=(self.engine, self.scoring_func),
                     method='Nelder-Mead')
        e3, x3 = self._extract_from_scipy_minimize(m)
        return e3, x3

    def _extract_from_scipy_minimize(self, m):

        x3 = m.x
        x3 = self.docking_utils.keep_bound(x3)
        e3 = m.fun
        return e3, x3

    def _minimize_lbfgsb(self, bounds, x2):

        m = minimize(self.docking_utils.score_coords, x2, method='L-BFGS-B',
                     jac=self.docking_utils.score_coords_jac,
                     args=(self.engine, self.scoring_func), bounds=bounds, options={'maxiter': self.optim_iter})
        e3, x3 = self._extract_from_scipy_minimize(m)
        return e3, x3

    def generate_conformation(self, x1):

        for _ in range(self.mut_steps):
            x2 = self.docking_utils.rand_mutate_big(x1.copy())
            c2 = self.engine.lig.mutate(x2)
        return c2, x2

    # generate random coordinate
    def generate_rotor_vector(self):

        trans_vec = np.random.uniform(-1, 1, size=3)
        rot_vec = np.random.uniform(-np.pi, np.pi, size=3)
        rotors_vec = np.random.uniform(-np.pi, np.pi, size=self.num_rotors)
        return np.hstack((trans_vec, rot_vec, rotors_vec))
