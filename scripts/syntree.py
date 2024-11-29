from collections import defaultdict

import numpy as np
import itertools
from rdkit import Chem
from time import time
from tqdm import tqdm

from .chem_utils import *
from .data_utils import *
        
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
                filtered_bbs, filtered_dists, filtered_sims = [], [], []
                pattern = Chem.MolFromSmarts(work_templates[i])
                sorted_bblock_idx = np.argsort(-np.array(reactant.bblock_dict['similarity']))
                for k in sorted_bblock_idx:
                    bblock = reactant.bblock_dict['bblock'][k]
                    k_dist, k_sim = reactant.bblock_dict['distance'][k], reactant.bblock_dict['similarity'][k]
                    k_smi, k_mol = bblock.smi, bblock.mol
                    if k_mol.HasSubstructMatch(pattern):
                        filtered_bbs.append(bblock)
                        filtered_dists.append(k_dist)
                        filtered_sims.append(k_sim)
                        if len(filtered_bbs) == filter_k:
                            break
                reactant.bblock_dict = {'bblock': filtered_bbs, 'distance': filtered_dists, 'similarity': filtered_sims}
                filtered_reactants.append(reactant)
            filtered_children.append((reaction, filtered_reactants))
        self.filtered_children = filtered_children
        return
    
    def get_analogs(self, filter_k=1, filtered_by='similarity', debug = False):
        results = {}
        for i, (reaction, reactants) in enumerate(self.filtered_children):
            work_reaction = reaction.retro_rxn.work_rxn
            reactant_smis = [reactant.smi for reactant in reactants]
            knn_bb_smi, knn_bb_mols = [], []
            bb_precursors = {}
            for reactant in reactants:
                
                template_reactant_smi, template_reactant_mol = [], []
                bblocks = reactant.bblock_dict['bblock']
                if filtered_by == 'similarity':
                    ranks = [i for i, similarity in enumerate(sorted(reactant.bblock_dict['similarity'], key=lambda x: -x))]
                    bblocks = [bblocks[i] for i in ranks[:filter_k]]
                elif filtered_by == 'distance':
                    ranks = [i for i, distance in enumerate(sorted(reactant.bblock_dict['distance']))]
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
            
            knn_distance = [[dist for dist in reactant.bblock_dict['distance']] for reactant in reactants]
            knn_smi_sets = list(itertools.product(*knn_bb_smi))
            knn_mol_sets = list(itertools.product(*knn_bb_mols))
            knn_dist_sets = list(itertools.product(*knn_distance))
            if not knn_dist_sets:
                continue
                
            for smi_set, mol_set, dist_set in zip(knn_smi_sets, knn_mol_sets, knn_dist_sets):
                try:
                    target_analogs = reaction.run_forward(mol_set)
                    weighted_dist = atomic_weighted_mean('.'.join(reactant_smis), dist_set)
                    for analog in target_analogs:
                        similarity = get_sim(analog, self.smi)
                        if similarity == 1 and analog != self.smi:
                            similarity = 0.99
                        if analog not in results or similarity > results[analog]['similarity']:
                            anolog_bb_precursors = {smi: bb_precursors[smi] for smi in smi_set if smi in bb_precursors}
                            results[analog] = {'reaction': work_reaction.reference, 
                                               'reference': '.'.join(reactant_smis),
                                               'bblocks': '.'.join(smi_set), 
                                               'bb_precursors': anolog_bb_precursors,
                                               'distance': weighted_dist,
                                               'similarity': similarity}
                except Exception as e:
                    print (e)
                    pass
        
        high_rank_analogs = [analog for analog, result in sorted(results.items(), key=lambda x: -x[1]['similarity'])][:filter_k]
        top_results = {analog: results[analog] for analog in high_rank_analogs}
        self.analogs = top_results
        return top_results
    
    def update_bblock_dict_with_analogs(self):
        knn_bb_smi = [bblock.smi for bblock in self.bblock_dict['bblock']]
        for analog, result in self.analogs.items():
            bb_precursor = '%s;%s' % (result['bblocks'], result['reaction'])
            self.bblock_dict['bblock'].append(MoleculeNode(analog, bb_precursor=bb_precursor))
            self.bblock_dict['distance'].append(result['distance'])
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
#             print ('Error!')
#             print (self.work_ref)
#             print (self.retro_smirks)
#             print (len(mols))
#             print (e)
            return []
        sanitized_products = []
        for product in products:
            sanitized_product = sanitize(product[0])
            if sanitized_product:
                sanitized_products.append(sanitized_product)
        return list(set([Chem.MolToSmiles(mol) for mol in sanitized_products]))

class SynTree:
    def __init__(self, target_smi, retro_rxns, bblocks, kdtrees, kd_bblocks, max_depth=2, verbose=False):
        self.target_node = MoleculeNode(target_smi)
        self.bblocks = bblocks
        self.kdtrees = kdtrees
        self.kd_bblocks = kd_bblocks
        self.unique_precursors = defaultdict(list)
        self.retrosynthesis_lookup = {}
        self.unique_precursor_bblock = {}
        self.all_precursor_nodes = []
        self.expanded_nodes = {}
        work_rxn_dict = defaultdict(list)
        for retro_rxn in retro_rxns:
            reference = retro_rxn.work_rxn.reference
            work_rxn_dict[reference].append(retro_rxn)
        self.retro_rxns = retro_rxns
        self.max_depth = max_depth
        self.verbose = verbose
        
    def retrosynthesis(self, node, fast=False):
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
                    mols = [sanitize(fix_aromatic(mol)) for mol in mols]
                    if not all(mols):
                        continue
                    reactant_smi = '.'.join([Chem.MolToSmiles(mol) for mol in mols])
                    if min([len(smi) for smi in reactant_smi.split('.')]) < 5 or reactant_smi in precursors:
                        continue
                    if node.smi not in reaction_node.run_forward(mols):
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
                    available_precursors += fetched_children[:min([len(fetched_children), 3])]
                all_precursors += fetched_children[:min([len(fetched_children), 3])]
                
                
            else:
                all_precursors += list(itertools.chain(*rxn_childrens))
            
        if fast and n_available_rxn >= 10:
            return available_precursors
        else:
            return all_precursors
                    
    def multistep_retrosynthesis(self, node, depth=0, fast=False):
        if node.smi in self.retrosynthesis_lookup:
            precursors = self.retrosynthesis_lookup[node.smi]
        else:
            precursors = self.retrosynthesis(node, fast)
            self.retrosynthesis_lookup[node.smi] = precursors
            if self.verbose:
                print ('Molecule %s get %d precursors...' % (node.smi, len(precursors)))
            
        node.children = precursors
        depth += 1
        if depth >= self.max_depth:
            return
        for reaction, children in precursors:
            for child in children:
                if not child.is_available:
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
    
    def get_knn_bbs(self, k=1, _nBits=256):
        if len(self.unique_precursors) == 0:
            return
        for precursor, rxns in self.unique_precursors.items():
            bblocks = []
            distances = []
            similarities = []
            for i, work_ref in rxns:
                kdtree = self.kdtrees[work_ref][i]
                kd_bblock = self.kd_bblocks[work_ref][i]
                precursor_fps = get_fps(precursor, _nBits=_nBits).reshape(1,-1)
                dists, inds = kdtree.query(precursor_fps, k=min([k, len(kd_bblock)]))
                bblocks_smi = [kd_bblock[j] for j in inds[0]]
                bblocks += [MoleculeNode(smi) for smi in bblocks_smi]
                distances += [d for d in dists[0]]
                similarities += [get_sim(smi, precursor) for smi in bblocks_smi]
            bblock_dict = {'bblock': bblocks, 'distance': distances, 'similarity': similarities}
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
        
    def generate_all_analogs(self, k=1, sort_by='similarity'):
        # generate from deepest to shallowest
        all_precursor_nodes = sorted(self.all_precursor_nodes, key=lambda x: -x[1]) 
        completed_analogs = {}
        for node_depth in all_precursor_nodes:
            node, depth = node_depth
            node.reaction_filtering(k)
            if node.smi not in completed_analogs:
                analogs = node.get_analogs(k, sort_by)
                completed_analogs[node.smi] = analogs
            else:
                node.analogs = completed_analogs[node.smi]
            node.update_bblock_dict_with_analogs()

        self.target_node.reaction_filtering(k)
        analogs = self.target_node.get_analogs(k, sort_by)
        return analogs
    
    
    def get_twins(self, k=1, fast=False, sort_by='similarity'):
        t0 = time()
        if self.verbose:
            print ('1. Multi-step retrosynthesis...')
        self.multistep_retrosynthesis(self.target_node, depth=0, fast=fast)
        
        t1 = time()
        if self.verbose:
            print ('2. Getting k-nearest building blocks for precursors...')
        self.collect_precursors(self.target_node)
        self.get_knn_bbs(k)
        self.update_precursors_bbs(self.target_node, 0)
        t2 = time()
        if self.verbose:
            print ('3. Synthesizing analogs...')
        analogs = self.generate_all_analogs(k, sort_by)
        analogs = sort_analogs(analogs, sort_by=sort_by)
        t3 = time()
        self.inference_time = {'retrosynthesis': t1-t0, 
                               'knn': t2-t1, 
                               'synthesis': t3-t2, 
                               'total': t3-t0
                              }
        return analogs

    