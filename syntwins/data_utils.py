import os
import pickle
import gzip
from tqdm import tqdm
import numpy as np
import pandas as pd
from functools import partial
from sklearn.neighbors import BallTree

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*') 
from rdkit.Chem import AllChem, Draw, rdChemReactions

from syntwins.chem_utils import *
    
class Reaction:
    def __init__(self, template=None, rxnname=None, smiles=None, work_rxn=None, reference=None):
        self.smirks = template.strip()
        self.rxnname = rxnname
        self.smiles = smiles
        self.work_rxn = work_rxn
        self.reference = reference

        self.rxn = self.__init_reaction(self.smirks)
        
        reactants, agents, products = self.smirks.split(">")
        self.num_reactant = len(reactants.split('.'))
        self.num_agent = len(agents.split('.'))
        self.num_product = len(products.split('.'))

        if self.num_reactant == 1:
            self.reactant_template = list((reactants,))
        else:
            self.reactant_template = list(reactants.split("."))
        self.product_template = products
        self.agent_template = agents

    def __init_reaction(self, smirks: str) -> Chem.rdChemReactions.ChemicalReaction:
        rxn = AllChem.ReactionFromSmarts(smirks)
        rdChemReactions.ChemicalReaction.Initialize(rxn)
        return rxn

    def _filter_reactants(
        self, bblocks_smi, bblocks_mol, verbose = False):
        matched_reactants = []
        matched_reactants_ids = []
        self.n_potential_reactions = 1
        for smarts in self.reactant_template:
            pattern = Chem.MolFromSmarts(smarts)
            matches = []
            matches_id = []
            for i, (smi, mol) in enumerate(zip(bblocks_smi, bblocks_mol)):
                if mol.HasSubstructMatch(pattern):
                    matches.append(smi)
                    matches_id.append(i)
            matched_reactants.append(matches)
            matched_reactants_ids.append(matches_id)
            self.n_potential_reactions *= len(matches)
            
        return tuple(matched_reactants), tuple(matched_reactants_ids)

    def set_available_reactants(self, bblocks_smi, bblocks_mol, verbose: bool = False):
        available_reactants, available_reactants_ids = self._filter_reactants(bblocks_smi, bblocks_mol, verbose=verbose)
        self.available_reactants = available_reactants
        self.available_reactants_ids = available_reactants_ids
        return self

    
class SynTwin_DataLoader:
    def __init__(self, data_dir='data', _nBits=1024, verbose=True):
        self.verbose = verbose
        
        self.bblocks_path = '%s/building_blocks.smi' % data_dir
        self.template_path = '%s/templates.csv' % data_dir
        self.retro_template_path = '%s/retro_templates.csv' % data_dir
        self.reactions_path = '%s/preprocessed/syntwins_reactions.pickle' % (data_dir)
        self._nBits = _nBits
        self.verbose = verbose
        
        self.load_building_blocks()
        self.load_work_reactions()
        self.load_retro_reactions()
        self.load_bblocks_balltrees()
        
    def load_building_blocks(self):
        self.bblocks_smi = []
        with open(self.bblocks_path, "rt") as f:
            for i, line in enumerate(f.readlines()):
                smi = line.strip()
                self.bblocks_smi.append(smi)
                
        self.bblocks_mol = []
        for smi in tqdm(self.bblocks_smi, desc='Loading building blocks...'):
            self.bblocks_mol.append(Chem.MolFromSmiles(smi))
        return
        
    def load_work_reactions(self):
        self.template_refs = {}
        df = pd.read_csv(self.template_path)
        for template_ref, template in zip(df['rxn_ref'], df['rxn_smirks']):
            self.template_refs[template_ref] = template
        
        if not os.path.exists(self.reactions_path):
            self.work_rxns = {}
            for i, (rxn_ref, template) in tqdm(enumerate(self.template_refs.items()), total=len(self.template_refs), desc='Loading reactions...'):
                rxn = Reaction(template=template, reference=rxn_ref)
                rxn.set_available_reactants(self.bblocks_smi, self.bblocks_mol)
                self.work_rxns[rxn_ref] = rxn
                with gzip.open(self.reactions_path, 'wb') as f:
                    pickle.dump(self.work_rxns, f)
                
        else:
            with gzip.open(self.reactions_path,'rb') as f:
                self.work_rxns = pickle.load(f)
            print ('Loaded %d reactions from %s.' % (len(self.work_rxns), self.reactions_path))
        
        return

    def load_retro_reactions(self):
        self.retro_rxns = []
        self.retro_df = pd.read_csv(self.retro_template_path)
        for temp_ref, retro_template in zip(self.retro_df['rxn_ref'], self.retro_df['retro_template']):
            n_product = len(retro_template.split('>>')[0].split('.'))
            if n_product == 1 and temp_ref in self.template_refs:
                work_rxn = Reaction(template=self.work_rxns[temp_ref].smirks, reference=temp_ref)
                retro_reaction = Reaction(template=retro_template, work_rxn=work_rxn)
                self.retro_rxns.append(retro_reaction)
        return
        
    def load_bblocks_balltrees(self):
        self.kd_trees, self.kd_bblocks = {}, {}
        embeddings = [get_fps(smi, _radius=2, _nBits=self._nBits, useChirality=False) for smi in self.bblocks_smi]
        all_fps = np.array(embeddings)
        self.kd_trees['all'] = BallTree(all_fps, metric="euclidean")
        for temp_ref, rxn in tqdm(self.work_rxns.items(), desc='Building reaction balltrees...'):
            trees = []
            if rxn.n_potential_reactions == 0:
                self.kd_trees[temp_ref] = []
                self.kd_bblocks[temp_ref] = []
                continue
            for rxn_ids in rxn.available_reactants_ids:
                filtered_fps = all_fps[rxn_ids]
                trees.append(BallTree(filtered_fps, metric="euclidean"))
            self.kd_trees[temp_ref] = trees
            self.kd_bblocks[temp_ref] = rxn.available_reactants
        return 