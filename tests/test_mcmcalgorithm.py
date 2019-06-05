import os

from numpy.testing import assert_almost_equal

import oddt
from oddt.docking.MCMCAlgorithm import MCMCAlgorithm, OptimizationMethod
from oddt.docking.internal import vina_docking

test_data_dir = os.path.dirname(os.path.abspath(__file__))
receptor = next(oddt.toolkit.readfile('pdb', os.path.join(
    test_data_dir, 'data/dude/xiap/receptor_rdkit.pdb')))
receptor.protein = True
receptor.addh(only_polar=True)

mols = list(oddt.toolkit.readfile('sdf', os.path.join(
        test_data_dir, 'data/dude/xiap/actives_docked.sdf')))
mols = list(filter(lambda x: x.title == '312335', mols))
_ = list(map(lambda x: x.addh(only_polar=True), mols))
lig = mols[0]


def test_score_nelder_mead():
    engine = vina_docking(receptor, lig)
    mcmc = MCMCAlgorithm(engine, optim=OptimizationMethod.NELDER_MEAD, optim_iter=7, mc_steps=7, mut_steps=100, seed=316815)
    out = mcmc.perform()
    assert_almost_equal(out['score'], -3.630795265315847, decimal=0)


def test_score_simplex():
    engine = vina_docking(receptor, lig)
    mcmc = MCMCAlgorithm(engine, optim=OptimizationMethod.SIMPLEX, optim_iter=7, mc_steps=50, mut_steps=100, seed=316815)
    out = mcmc.perform()
    assert_almost_equal(out['score'], -0.8167642695350316, decimal=0)


def test_score_lbfgsb():
    engine = vina_docking(receptor, lig)
    mcmc = MCMCAlgorithm(engine, optim=OptimizationMethod.LBFGSB, optim_iter=7, mc_steps=7, mut_steps=100, seed=316815)
    out = mcmc.perform()
    assert_almost_equal(out['score'], -1.746366155192594, decimal=0)
