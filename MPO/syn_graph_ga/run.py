from __future__ import print_function

import random
from typing import List

import joblib
import numpy as np
from joblib import delayed
from rdkit import Chem, rdBase
from rdkit.Chem.rdchem import Mol
rdBase.DisableLog('rdApp.error')

from MPO.optimizer import BaseOptimizer
from . import crossover as co, mutate as mu

MINIMUM = 1e-10

def make_mating_pool(population_mol: List[Mol], population_scores, offspring_size: int):
    """
    Given a population of RDKit Mol and their scores, sample a list of the same size
    with replacement using the population_scores as weights
    Args:
        population_mol: list of RDKit Mol
        population_scores: list of un-normalised scores given by ScoringFunction
        offspring_size: number of molecules to return
    Returns: a list of RDKit Mol (probably not unique)
    """
    # scores -> probs 
    population_scores = [s + MINIMUM for s in population_scores]
    sum_scores = sum(population_scores)
    population_probs = [p / sum_scores for p in population_scores]
    mating_pool = np.random.choice(population_mol, p=population_probs, size=offspring_size, replace=True)
    return mating_pool


def reproduce(mating_pool, mutation_rate):
    """
    Args:
        mating_pool: list of RDKit Mol
        mutation_rate: rate of mutation
    Returns:
    """
    parent_a = random.choice(mating_pool)
    parent_b = random.choice(mating_pool)
    new_child = co.crossover(parent_a, parent_b)
    if new_child is not None:
        new_child = mu.mutate(new_child, mutation_rate)
    return new_child
        
class Syn_GraphGA_Optimizer(BaseOptimizer):
    def __init__(self, syntwins_loader, args=None):
        super().__init__(args)
        self.model_name = "syn_graphga"
        self.syntwins_loader = syntwins_loader

    def get_syntwins(self, mol_list):
        twins_set = set()
        for mol in mol_list:
            self.tree = SynTree(self.syntwins_loader, _nBits=256, k=1, max_depth=2)
            twins = self.tree.get_twins(Chem.MolToSmiles(mol)).keys()
            twins_set.update(list((twins)))
        return list(twins_set)
        
    def _optimize(self, oracle, config):
        
        self.oracle.assign_evaluator(oracle)

        # pool = joblib.Parallel(n_jobs=self.n_jobs)
        pool = joblib.Parallel(n_jobs=self.n_jobs, prefer='threads')
        
        if self.smi_file is not None:
            # Exploitation run
            starting_population = self.all_smiles[:config["population_size"]]
        else:
            # Exploration run
            starting_population = self.all_smiles[:1] + list(np.random.choice(self.all_smiles, config["population_size"]))

        print ('length of starting population:', len(starting_population))
        # select initial population
        # population_smiles = heapq.nlargest(config["population_size"], starting_population, key=oracle)
        # population_smiles = starting_population
        # population_smiles = list(pool(delayed(self.get_syntwins)(smi) for smi in population_smiles))
        population_mols = [Chem.MolFromSmiles(smi) for smi in starting_population]
        population_smiles = self.get_syntwins(population_mols)
        population_mol = [Chem.MolFromSmiles(s) for s in set(population_smiles)]
        print ('length of starting mols:', len(population_mol))
        population_scores = self.oracle([Chem.MolToSmiles(mol) for mol in population_mol])

        patience = 0

        while True:

            if len(self.oracle) > 100:
                self.sort_buffer()
                old_score = np.mean([item[1][0] for item in list(self.mol_buffer.items())[:100]])
            else:
                old_score = 0

            # new_population
            mating_pool = make_mating_pool(population_mol, population_scores, config["population_size"])
            offspring_mol = pool(delayed(reproduce)(mating_pool, config["mutation_rate"]) for _ in range(config["offspring_size"]))
            
            offsrping_twins = self.get_syntwins(self.sanitize(offspring_mol))
            offspring_mol = [Chem.MolFromSmiles(smi) for smi in set(offsrping_twins)]
            
            # offspring_lists = pool(delayed(self.get_syntwins)(smiles) for smiles in [Chem.MolToSmiles(mol) for mol in self.sanitize(population_mol)])
            # offspring_smi = list(set([offspring for offsprings in offspring_lists for offspring in offsprings]))
            # offspring_mol = [Chem.MolFromSmiles(smi) for smi in offspring_smi]

            # add new_population
            population_mol += offspring_mol
            population_mol = self.sanitize(population_mol)

            # stats
            old_scores = population_scores
            population_scores = self.oracle([Chem.MolToSmiles(mol) for mol in population_mol])
            population_tuples = list(zip(population_scores, population_mol))
            population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:config["population_size"]]
            population_mol = [t[1] for t in population_tuples]
            population_scores = [t[0] for t in population_tuples]

            # # early stopping
            # if population_scores == old_scores:
            #     patience += 1
            #     if patience >= self.args.patience:
            #         self.log_intermediate(finish=True)
            #         break
            # else:
            #     patience = 0

            ### early stopping
            if len(self.oracle) > 100:
                self.sort_buffer()
                new_score = np.mean([item[1][0] for item in list(self.mol_buffer.items())[:100]])
                # import ipdb; ipdb.set_trace()
                if (new_score - old_score) < 1e-3:
                    patience += 1
                    if patience >= self.args.patience:
                        self.log_intermediate(finish=True)
                        print('convergence criteria met, abort ...... ')
                        break
                else:
                    patience = 0

                old_score = new_score
                
            if self.finish:
                break

