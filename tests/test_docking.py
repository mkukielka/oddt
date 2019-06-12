import os
from numpy.testing import assert_array_less
import oddt
from oddt.docking.dock import Dock


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


def test_genetic_algorithm():
    """Checks, whether genetic algorithm is minimizing energy of protein-ligand complex."""
    ga_params = {'n_population': 20, 'n_generations': 20, 'top_individuals': 2,
                 'top_parents': 5, 'crossover_prob': 0.9, 'seed': 123}
    docker = Dock(receptor, mols[:5], docking_type='GeneticAlgorithm', additional_params=ga_params)
    init_scores = [engine.best_score for engine in docker.custom_engines]
    docker.perform()
    scores = [score for _, score in docker.output]

    assert_array_less(scores, init_scores)
