from collections import defaultdict
from functools import partial
from joblib import Parallel,  delayed

import numpy as np
import itertools
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')                

from time import time
from tqdm import tqdm

from syntwins.chem_utils import *
from syntwins.data_utils import * 
        
class MoleculeNode:
    def __init__(self, smiles, bb_precursor=None, bblock_dict=None, is_available=False):
        self.mol = Chem.MolFromSmiles(smiles)
        self.smi = Chem.MolToSmiles(self.mol)
        self.bb_precursor = bb_precursor
        self.bblock_dict = bblock_dict
        self.children = None
        self.bblocks_updated = False
        self.is_available = is_available
        self.synthesis_paths = {}

    def set_available(self, bblocks):
        if self.smi in bblocks:
            self.is_available = True
        return
    
    def reaction_filtering(self, filter_k=1):
        filtered_children = []
        for reaction, reactants in self.children:
            filtered_reactants = []
            work_templates = reaction.work_rxn.reactant_template
            if len(work_templates) != len(reactants):
                continue
            for i, reactant in enumerate(reactants):
                filtered_bbs, filtered_sims = [], []
                pattern = Chem.MolFromSmarts(work_templates[i])
                sorted_bblock_idx = np.argsort(-np.array(reactant.bblock_dict['similarity']))
                for k in sorted_bblock_idx:
                    bblock = reactant.bblock_dict['bblock'][k]
                    k_sim = reactant.bblock_dict['similarity'][k]
                    k_smi, k_mol = bblock.smi, bblock.mol
                    if k_mol.HasSubstructMatch(pattern):
                        filtered_bbs.append(bblock)
                        filtered_sims.append(k_sim)
                        if len(filtered_bbs) == filter_k:
                            break
                reactant.bblock_dict = {'bblock': filtered_bbs, 'similarity': filtered_sims}
                filtered_reactants.append(reactant)
            filtered_children.append((reaction, filtered_reactants))
        self.filtered_children = filtered_children
        return
    
    def get_analogs(self, filter_k=1, debug = False):
        results = {}
        for i, (reaction, reactants) in enumerate(self.filtered_children):
            work_reaction = reaction.retro_rxn.work_rxn
            reactant_smis = [reactant.smi for reactant in reactants]
            knn_bb_smi, knn_bb_mols = [], []
            bb_precursors = {}
            for reactant in reactants:
                
                template_reactant_smi, template_reactant_mol = [], []
                bblocks = reactant.bblock_dict['bblock']
                ranks = [i for i, similarity in enumerate(sorted(reactant.bblock_dict['similarity'], key=lambda x: -x))]
                bblocks = [bblocks[i] for i in ranks[:filter_k]]
                for bblock in bblocks:
                    if bblock.bb_precursor:
                        bb_precursors[bblock.smi] = bblock.bb_precursor
                    template_reactant_smi.append(bblock.smi)    
                    template_reactant_mol.append(bblock.mol)
                knn_bb_smi.append(template_reactant_smi)
                knn_bb_mols.append(template_reactant_mol)
            if debug:
                print (knn_bb_smi)
            
            knn_smi_sets = list(itertools.product(*knn_bb_smi))
            knn_mol_sets = list(itertools.product(*knn_bb_mols))
            if not knn_smi_sets:
                continue
                
            for smi_set, mol_set in zip(knn_smi_sets, knn_mol_sets):
                try:
                    target_analogs = reaction.run_forward(mol_set)
                    for analog in target_analogs:
                        similarity = get_sim(analog, self.smi, _nBits=4096)
                        if analog not in results or similarity > results[analog]['similarity']:
                            anolog_bb_precursors = {smi: bb_precursors[smi] for smi in smi_set if smi in bb_precursors}
                            results[analog] = {'reaction': work_reaction.reference, 
                                               'reference': '.'.join(reactant_smis),
                                               'bblocks': '.'.join(smi_set), 
                                               'bb_precursors': anolog_bb_precursors,
                                               'similarity': similarity}
                except Exception as e:
                    print (e)
                    pass
        
        high_rank_analogs = [analog for analog, result in sorted(results.items(), key=lambda x: -x[1]['similarity'])]
        if filter_k > 0:
            high_rank_analogs = high_rank_analogs[:filter_k]
        top_results = {analog: results[analog] for analog in high_rank_analogs}
        self.analogs = top_results
        return top_results
    
    def update_bblock_dict_with_analogs(self):
        knn_bb_smi = [bblock.smi for bblock in self.bblock_dict['bblock']]
        for analog, result in self.analogs.items():
            bb_precursor = '%s;%s' % (result['bblocks'], result['reaction'])
            self.bblock_dict['bblock'].append(MoleculeNode(analog, bb_precursor=bb_precursor))
            self.bblock_dict['similarity'].append(result['similarity'])
        self.bblocks_updated = True
        return
    
class ReactionNode:
    def __init__(self, retro_rxn):
        self.retro_rxn = retro_rxn
        self.work_rxn = retro_rxn.work_rxn
        self.work_ref = self.work_rxn.reference
        self.retro_smirks = retro_rxn.smirks
        p_temp, r_temp = self.retro_smirks.split('>>')
        
    def run_forward(self, mols):
        reaction = self.work_rxn.rxn
        try:
            products = reaction.RunReactants(mols)
        except Exception as e:
            return []
        sanitized_products = []
        for product in products:
            sanitized_product = sanitize(product[0])
            if sanitized_product:
                sanitized_products.append(sanitized_product)
        return list(set([Chem.MolToSmiles(mol) for mol in sanitized_products]))

class NullReactionNode:
    def __init__(self):
        self.retro_smirks = '>>'
        p_temp, r_temp = self.retro_smirks.split('>>')
        
    def run_forward(self, mols):
        return mols
        
class SynTree:
    def __init__(self, loader, _nBits=1024, max_depth=3, k=5, verbose=False):
        self.bblocks = loader.bblocks_smi
        self.kdtrees = loader.kd_trees
        self.kd_bblocks = loader.kd_bblocks
        self.retro_rxns = loader.retro_rxns
        self.unique_precursors = defaultdict(list)
        work_rxn_dict = defaultdict(list)
        for retro_rxn in self.retro_rxns:
            reference = retro_rxn.work_rxn.reference
            work_rxn_dict[reference].append(retro_rxn)
        self._nBits = _nBits
        self.max_depth = max_depth
        self.k = k
        self.verbose = verbose
        self.retrosynthesis_lookup = {}
        self.pool = Parallel(n_jobs=64, prefer='threads')
        
    def reset_cache(self):
        # self.retrosynthesis_lookup = {}
        self.unique_precursor_bblock = {}
        self.all_precursor_nodes = []
        self.expanded_nodes = {}
        self.analogs = None
        
    def retrosynthesis(self, node, fast=False):
        # if node.is_available:
        #     precursors = []
        if node.smi in self.retrosynthesis_lookup:
            precursors = self.retrosynthesis_lookup[node.smi]
        else:
            
            reaction_precursors = defaultdict(list)
            precursors = []
            for retro_rxn in self.retro_rxns:
                rxn_precursors = []
                rxn_ref = retro_rxn.work_rxn.reference
                rxn_prod = retro_rxn.smirks.split('>>')[0]
                rxn_rep = '%s_%s' % (rxn_ref, rxn_prod)
                reaction_node = ReactionNode(retro_rxn)
                reactant_mols = retro_rxn.rxn.RunReactants([node.mol])
                if len(reactant_mols) > 0:
                    for mols in reactant_mols:
                        mols = sanitize_mols(mols)
                        if not all(mols):
                            continue
                        reactant_smi = '.'.join([Chem.MolToSmiles(mol) for mol in mols])
                        # if reactant_smi in precursors:
                        #     continue
                        if min([len(smi) for smi in reactant_smi.split('.')]) < 3 or reactant_smi in precursors:
                            continue
                        if node.smi not in reaction_node.run_forward(mols):  # if it cannot recover original product
                            continue
                        else:
                            precursors.append(reactant_smi)
                            reactant_nodes = [MoleculeNode(smi) for smi in reactant_smi.split('.')]
                            [reactant_node.set_available(self.bblocks) for reactant_node in reactant_nodes]
                            rxn_precursors.append((reaction_node, reactant_nodes))
                if len(rxn_precursors) > 0:
                    reaction_precursors[rxn_rep].append(rxn_precursors)
                            
            all_precursors = []
            available_precursors = []
            n_available_rxn = 0
            for rxn_rep, rxn_childrens in reaction_precursors.items():
                if fast:
                    max_available = 0
                    fetched_children = []
                    fetch_ij = (0, 0)
                    for i, rxn_children in enumerate(rxn_childrens):
                        for j, (rxn, children) in enumerate(rxn_children):
                            n_available = sum([child.is_available for child in children])
                            if n_available > 0:
                                n_available_rxn += 1
                                fetch_ij = (i, j)
                                max_available = n_available
                                fetched_children.append((rxn, children))
                    
                    if max_available == 0:
                        fetched_children = rxn_childrens[0]
                        np.random.shuffle(fetched_children)
                    else:
                        available_precursors += fetched_children[:min([len(fetched_children), self.k])]
                    all_precursors += fetched_children[:min([len(fetched_children), self.k])]
                    
                else:
                    all_precursors += list(itertools.chain(*rxn_childrens))
            
            if fast and n_available_rxn >= self.k:
                precursors = available_precursors
            else:
                precursors = all_precursors

        self.retrosynthesis_lookup[node.smi] = precursors
        node.children = precursors
        if len(precursors) > 0 and self.verbose:
            print ('Molecule %s get %d precursors...' % (node.smi, len(precursors)))
        return precursors
        
    def multistep_retrosynthesis(self, node, depth=0, fast=False): ## DFS
        precursors = self.retrosynthesis(node, fast)
        # node.children = precursors
        depth += 1
        if depth >= self.max_depth:
            return
        for reaction, children in precursors:
            for child in children:
                self.multistep_retrosynthesis(child, depth, fast)
        return
        
    def collect_precursors(self, node):
        if not node.children:
            return
        for reaction, children in node.children:
            rxn_ref = reaction.work_ref
            for i, child in enumerate(children):
                if (i, rxn_ref) not in self.unique_precursors[child.smi]:
                    self.unique_precursors[child.smi].append((i, rxn_ref))
                self.collect_precursors(child)
        return    

    def get_knn_bbs(self, precursor, bblocks, kdtree):
        precursor_fps = get_fps(precursor, _nBits=self._nBits).reshape(1,-1)
        dists, inds = kdtree.query(precursor_fps, k=min([self.k, len(bblocks)]))
        knn_smis = [bblocks[j] for j in inds[0]]
        knn_mols = [MoleculeNode(smi) for smi in knn_smis]
        similarities = [get_sim(smi, precursor, _nBits=4096) for smi in knn_smis]
        return knn_mols, similarities
        
    def get_all_knn_bbs(self):
        if len(self.unique_precursors) == 0:
            return
        for precursor, rxns in self.unique_precursors.items():
            bblocks = []
            similarities = []
            for i, work_ref in rxns:
                ref_kdtree = self.kdtrees[work_ref][i]
                ref_bblocks = self.kd_bblocks[work_ref][i]
                knn_mols, knn_sims = self.get_knn_bbs(precursor, ref_bblocks, ref_kdtree)
                bblocks += knn_mols
                similarities += knn_sims
            bblock_dict = {'bblock': bblocks, 'similarity': similarities}
            self.unique_precursor_bblock[precursor] = bblock_dict
        return
        
    def update_precursors_bbs(self, node, depth=0):
        depth += 1
        for reaction, children in node.children:
            for child in children:
                child.bblock_dict = self.unique_precursor_bblock[child.smi]
                if child.children:
                    self.all_precursor_nodes.append([child, depth])
                    self.update_precursors_bbs(child, depth)
        return

    def generate_all_analogs(self, target_node):
        # generate from deepest to shallowest
        all_precursor_nodes = sorted(self.all_precursor_nodes, key=lambda x: -x[1]) 
        completed_analogs = {}
        for node_depth in all_precursor_nodes:
            node, depth = node_depth
            node.reaction_filtering(self.k)
            if node.smi not in completed_analogs:
                analogs = node.get_analogs(self.k)
                completed_analogs[node.smi] = analogs
            else:
                node.analogs = completed_analogs[node.smi]
            node.update_bblock_dict_with_analogs()

        target_node.reaction_filtering(10)
        analogs = target_node.get_analogs(-1)
        return analogs
    
    def get_twins(self, target_smi, fast=True, return_as_list=False):
        self.reset_cache()
        target_node = MoleculeNode(target_smi)
        
        t0 = time()
        if self.verbose:
            print ('1. Multi-step retrosynthesis...')
        self.multistep_retrosynthesis(target_node, depth=0, fast=fast)
        
        t1 = time()
        if self.verbose:
            print ('2. Getting k-nearest building blocks for precursors...')
        self.collect_precursors(target_node)
        self.get_all_knn_bbs()
        self.update_precursors_bbs(target_node, 0)
        target_knn_mols, target_knn_sims = self.get_knn_bbs(target_node.smi, self.bblocks, self.kdtrees['all'])
        taget_analogs = {
            mol.smi: {'reaction': None, 'reference': None, 'bblocks': None, 'bb_precursors': None, 'similarity': sim}
            for mol, sim in zip(target_knn_mols, target_knn_sims)
        }
        
        t2 = time()
        if self.verbose:
            print ('3. Synthesizing analogs...')
        analogs = self.generate_all_analogs(target_node)
        analogs.update(taget_analogs)
        self.analogs = {k: v for k, v in sorted(analogs.items(), key=lambda x: -x[1]['similarity'])}
        
        t3 = time()
        self.inference_time = {'retrosynthesis': t1-t0, 
                               'knn': t2-t1, 
                               'synthesis': t3-t2, 
                               'total': t3-t0
                              }
            
        if return_as_list:
            return [(k, '%s, %s' % (v['bblocks'], v['bb_precursors'])) for k, v in self.analogs.items()]
        else:
            return self.analogs

    def show_analogs(self, show_n=5):
        for i, (analog, result) in enumerate(self.analogs.items()):
            if i == show_n:
                break
            print ('Rank %d twin: %s' % (i+1, analog))
            print ('Similarity=%.2f' % (result['similarity']))
            print ('Reference precursors:', result['reference'])
            print ('Twin precursors: %s;%s' % (result['bblocks'], result['reaction']))
            print ('Precursors of twin precursors:', result['bb_precursors'])
            display(Chem.MolFromSmiles(analog))
        return 
    