import re
import os
import functools
import gzip
import random
import itertools
import json
import copy
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from typing import Any, Optional, Set, Tuple, Union

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*') 
from rdkit.Chem import AllChem, Draw, rdChemReactions
from tqdm import tqdm
import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)

from .chem_utils import *

def sample_bb(building_blocks):
    idx = np.random.choice(len(building_blocks))
    return building_blocks[idx]

def map_bbs(sampled_smis):
    sampled_mapped_smis = []
    sampled_mols = []
    map_num = 0
    for i, smi in enumerate(sampled_smis):
        mol = Chem.MolFromSmiles(smi)
        for atom in mol.GetAtoms():
            map_num += 1
            atom.SetProp('reactant_idx', str(i))
            atom.SetProp('hidden_mapno', str(map_num))
        sampled_mols.append(mol)
        mol_copy = copy.deepcopy(mol)
        [atom.SetAtomMapNum(int(atom.GetProp('hidden_mapno'))) for atom in mol_copy.GetAtoms()]
        sampled_mapped_smis.append(Chem.MolToSmiles(mol_copy))
    return sampled_mapped_smis, sampled_mols

def get_product_mappings(product_mol, reactant_mols, map_number_groups):
    product_mapping_list = []
    for i, atom in enumerate(product_mol.GetAtoms()):
        if atom.HasProp('old_mapno') and atom.HasProp('react_atom_idx'):
            template_num = int(atom.GetProp('old_mapno'))
            reactant_atom_idx = int(atom.GetProp('react_atom_idx'))
            which_reactant = [i for i, map_numbers in enumerate(map_number_groups) if template_num in map_numbers][0]
            AtomMapNum = int(reactant_mols[which_reactant].GetAtomWithIdx(reactant_atom_idx).GetProp('hidden_mapno'))
        elif atom.HasProp('hidden_mapno'):
            AtomMapNum = int(atom.GetProp('hidden_mapno'))
        else:
            AtomMapNum = 0
        atom.SetAtomMapNum(AtomMapNum)
    return product_mol
    
def recover_products_map(product_mols, reactant_mols, map_number_groups):
    mapped_products = []
    for product_mol in product_mols:
        product_mol = get_product_mappings(product_mol, reactant_mols, map_number_groups)
        product_mol = sanitize(product_mol, canonical=False)
        if not product_mol:
            continue
        mapped_products.append(product_mol)
    return mapped_products

def generate_mapped_rxns(rxn, sampled_smis):
    map_number_groups = extract_map_numbers(rxn.smirks)
    sampled_smis, sampled_mols = map_bbs(sampled_smis)
    product_mols = [mols[0] for mols in rxn.rxn.RunReactants(sampled_mols)]
    product_mols = recover_products_map(product_mols, sampled_mols, map_number_groups)
    mapped_rxns = ['%s>>%s' % ('.'.join(sampled_smis), Chem.MolToSmiles(mol)) for mol in product_mols]
    return mapped_rxns

# def get_product_mappings(product_mol, reactant_mols, map_number_groups):
#     product_mapping_list = []
#     for i, atom in enumerate(product_mol.GetAtoms()):
#         if atom.HasProp('old_mapno') and atom.HasProp('react_atom_idx'):
#             template_num = int(atom.GetProp('old_mapno'))
#             reactant_atom_idx = int(atom.GetProp('react_atom_idx'))
#             which_reactant = [i for i, map_numbers in enumerate(map_number_groups) if template_num in map_numbers][0]
#             AtomMapNum = reactant_mols[which_reactant].GetAtomWithIdx(reactant_atom_idx).GetAtomMapNum()
#             product_mapping_list.append(AtomMapNum)
#     return product_mapping_list

# def remap_product(mol, product_mapping_list):
#     for i, atom in enumerate(mol.GetAtoms()):
#         if atom.GetAtomMapNum() == 0:
#             atom.SetAtomMapNum(product_mapping_list.pop(0))
#         if len(product_mapping_list) == 0:
#             break
#     return mol

# def recover_products_map(product_mols, reactant_mols, map_number_groups):
#     mapped_products = []
#     for product_mol in product_mols:
#         product_mapping_list = get_product_mappings(product_mol, reactant_mols, map_number_groups)
#         product_mol = sanitize(product_mol, canonical=False)
#         if not product_mol:
#             continue
#         product_mol = remap_product(product_mol, product_mapping_list)
#         mapped_products.append(product_mol)
#     return mapped_products

# def generate_mapped_rxns(rxn, sampled_smis):
#     sampled_smis, sampled_mols = map_bbs(sampled_smis)
#     product_mols = [mols[0] for mols in rxn.rxn.RunReactants(sampled_mols)]
#     map_number_groups = extract_map_numbers(rxn.smirks)
#     product_mols = recover_products_map(product_mols, sampled_mols, map_number_groups)
#     products_smi = [Chem.MolToSmiles(mol) for mol in product_mols if mol]
#     mapped_rxns = ['%s>>%s' % ('.'.join(sampled_smis), product_smi) for product_smi in products_smi]
#     return mapped_rxns

    
class SynTwin_DataLoader:
    def __init__(self, bblocks_path, template_path, retro_template_path, n_samples=100, verbose=True):
        self.n_samples = n_samples
        self.verbose = verbose
        self.load_data(bblocks_path, template_path, retro_template_path)
        
        
    def load_data(self, bblocks_path, template_path, retro_template_path):
        self.bblocks_smi = []
        with open(bblocks_path, "rt") as f:
            for i, line in enumerate(f.readlines()):
                smi = line.strip()
                self.bblocks_smi.append(smi)
                
        self.bblocks_mol = []
        for smi in tqdm(self.bblocks_smi, desc='Loading building blocks...'):
            self.bblocks_mol.append(Chem.MolFromSmiles(smi))
        print ('Loaded %d building blocks' % len(self.bblocks_smi))
        
        template_refs = {}
        with open(template_path, "rt") as f:
            for i, line in enumerate(f.readlines()):
                template = line.strip()
                template_refs['R%d' % i] = template        
        self.work_rxns = {}
        for i, (rxn_ref, template) in tqdm(enumerate(template_refs.items()), desc='Loading reactions...'):
            rxn = Reaction(template=template, reference=rxn_ref)
            rxn.set_available_reactants(self.bblocks_smi, self.bblocks_mol)
            self.work_rxns[rxn_ref] = rxn
        print ('Loaded %d reaction templates' % len(self.work_rxns))
        if not os.path.exists(retro_template_path):
            sampled_reactions = self.sample_reactions(self.n_samples)
            self.sampled_reactions = sampled_reactions
            retro_templates = self.get_retro_templates(sampled_reactions, retro_template_path)
        retro_templates = self.load_retro_data(retro_template_path)
        return

    def load_retro_data(self, retro_template_path):
        self.retro_rxns = []
        self.retro_df = pd.read_csv(retro_template_path)
        for temp_ref, retro_template in zip(self.retro_df['Original_reference'], self.retro_df['Retro_template']):
            n_product = len(retro_template.split('>>')[0].split('.'))
            if n_product == 1:
                work_rxn = Reaction(template=self.work_rxns[temp_ref].smirks, reference=temp_ref)
                retro_reaction = Reaction(template=retro_template, work_rxn=work_rxn)
                self.retro_rxns.append(retro_reaction)
        print ('Loaded %d retro-reactions' % len(self.retro_rxns))
        return self.retro_df

    def sample_reactions(self, n_samples=5000):
        template_sampled_reactions = {}
        n_sampled, n_potential, n_templates = 0, 0, 0
        for (temp_ref, rxn) in tqdm(self.work_rxns.items(), total=len(self.work_rxns)):
            if rxn.n_potential_reactions == 0:
                continue
#             all_potential_smis = list(itertools.product(*rxn.available_reactants))
#             random.shuffle(all_potential_smis)
#             all_sampled_smis = all_potential_smis[:n_samples]
            
            all_sampled_smis = []
            for _ in range(n_samples):
                all_sampled_smis.append([sample_bb(bbs) for bbs in rxn.available_reactants])
            
            unique_sampled_reactions = []
            for sampled_smis in all_sampled_smis:
                mapped_rxns = generate_mapped_rxns(rxn, sampled_smis)
                if not mapped_rxns:
                    continue
                for mapped_rxn in mapped_rxns:
                    if mapped_rxn not in unique_sampled_reactions:
                        unique_sampled_reactions.append(mapped_rxn)
            if len(unique_sampled_reactions) == 0:
                print ('Failed to generate reactions:', temp_ref)
                print (rxn.smirks)
            template_sampled_reactions[temp_ref] = unique_sampled_reactions
            n_sampled += len(unique_sampled_reactions)
            n_potential += rxn.n_potential_reactions
            n_templates += 1
        print ('Generated %d unique reactions from %d potential reactions with %d templates' % (n_sampled, n_potential, n_templates))
        return template_sampled_reactions

    def get_retro_templates(self, generated_reactions, retro_template_path=None):
        from .template_extractor import extract_retro_template
        references = []
        templates = []
        counts = []
        for temp_ref, mapped_rxns in tqdm(generated_reactions.items(), desc='Getting retro tempaltes...', position=0):
            retro_templates = defaultdict(int)
            for mapped_rxn in mapped_rxns:
                try:
                    retro_template = extract_retro_template(mapped_rxn)
                except:
                    continue
                if retro_template:
                    retro_templates[retro_template] += 1
            retro_templates = {k: v for k, v in sorted(retro_templates.items(), key=lambda x: -x[1])}
            references += [temp_ref]*len(retro_templates)
            templates += list(retro_templates.keys())
            counts += list(retro_templates.values())
        
        template_df = pd.DataFrame({'Original_reference': references, 'Retro_template': templates, 'Count': counts})
        if retro_template_path:
            template_df.to_csv(retro_template_path, index=None)
        return template_df
    
"""
Here we define the following classes for working with synthetic tree data:
* `Reaction`
* `ReactionSet`

"""

# the definition of reaction classes below
class Reaction:
    """
    This class models a chemical reaction based on a SMARTS transformation.

    Args:
        template (str): SMARTS string representing a chemical reaction.
        rxnname (str): The name of the reaction for downstream analysis.
        smiles: (str): A reaction SMILES string that macthes the SMARTS pattern.
        reference (str): Reference information for the reaction.
    """
    def __init__(self, template=None, rxnname=None, smiles=None, work_rxn=None, reference=None):

        if template is not None:
            # define a few attributes based on the input
            self.smirks = template.strip()
            self.rxnname = rxnname
            self.smiles = smiles
            self.work_rxn = work_rxn
            self.reference = reference
            
            # compute a few additional attributes
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
        else:
            self.smirks = None

    def __init_reaction(self, smirks: str) -> Chem.rdChemReactions.ChemicalReaction:
        """Initializes a reaction by converting the SMARTS-pattern to an `rdkit` object."""
        rxn = AllChem.ReactionFromSmarts(smirks)
        rdChemReactions.ChemicalReaction.Initialize(rxn)
        return rxn

    def load(
        self,
        smirks,
        num_reactant,
        num_agent,
        num_product,
        reactant_template,
        product_template,
        agent_template,
        available_reactants,
        rxnname,
        smiles,
        reference,
    ):
        """
        This function loads a set of elements and reconstructs a `Reaction` object.
        """
        self.smirks = smirks
        self.num_reactant = num_reactant
        self.num_agent = num_agent
        self.num_product = num_product
        self.reactant_template = list(reactant_template)
        self.product_template = product_template
        self.agent_template = agent_template
        self.available_reactants = list(available_reactants)  # TODO: use Tuple[list,list] here
        self.rxnname = rxnname
        self.smiles = smiles
        self.reference = reference
        self.rxn = self.__init_reaction(self.smirks)
        return self

    @functools.lru_cache(maxsize=20)
    def get_mol(self, smi: Union[str, Chem.Mol]) -> Chem.Mol:
        """
        A internal function that returns an `RDKit.Chem.Mol` object.

        Args:
            smi (str or RDKit.Chem.Mol): The query molecule, as either a SMILES
                string or an `RDKit.Chem.Mol` object.

        Returns:
            RDKit.Chem.Mol
        """
        if isinstance(smi, str):
            return Chem.MolFromSmiles(smi)
        elif isinstance(smi, Chem.Mol):
            return smi
        else:
            raise TypeError(f"{type(smi)} not supported, only `str` or `rdkit.Chem.Mol`")

    def visualize(self, name="./reaction1_highlight.o.png"):
        """
        A function that plots the chemical translation into a PNG figure.
        One can use "from IPython.display import Image ; Image(name)" to see it
        in a Python notebook.

        Args:
            name (str): The path to the figure.

        Returns:
            name (str): The path to the figure.
        """
        rxn = AllChem.ReactionFromSmarts(self.smirks)
        d2d = Draw.MolDraw2DCairo(800, 300)
        d2d.DrawReaction(rxn, highlightByReactant=True)
        png = d2d.GetDrawingText()
        open(name, "wb+").write(png)
        del rxn
        return name

    def is_reactant(self, smi: Union[str, Chem.Mol]) -> bool:
        """Checks if `smi` is a reactant of this reaction."""
        smi = self.get_mol(smi)
        return self.rxn.IsMoleculeReactant(smi)

    def is_agent(self, smi: Union[str, Chem.Mol]) -> bool:
        """Checks if `smi` is an agent of this reaction."""
        smi = self.get_mol(smi)
        return self.rxn.IsMoleculeAgent(smi)

    def is_product(self, smi):
        """Checks if `smi` is a product of this reaction."""
        smi = self.get_mol(smi)
        return self.rxn.IsMoleculeProduct(smi)

    def is_reactant_first(self, smi: Union[str, Chem.Mol]) -> bool:
        """Check if `smi` is the first reactant in this reaction"""
        mol = self.get_mol(smi)
        pattern = Chem.MolFromSmarts(self.reactant_template[0])
        return mol.HasSubstructMatch(pattern)

    def is_reactant_second(self, smi: Union[str, Chem.Mol]) -> bool:
        """Check if `smi` the second reactant in this reaction"""
        mol = self.get_mol(smi)
        pattern = Chem.MolFromSmarts(self.reactant_template[1])
        return mol.HasSubstructMatch(pattern)

    def run_reaction(
        self, reactants: Tuple[Union[str, Chem.Mol, None]], keep_main: bool = True
    ) -> Union[str, None]:
        """Run this reactions with reactants and return corresponding product.

        Args:
            reactants (tuple): Contains SMILES strings for the reactants.
            keep_main (bool): Return main product only or all possibel products. Defaults to True.

        Returns:
            uniqps: SMILES string representing the product or `None` if not reaction possible
        """
        # Input validation.
        if not isinstance(reactants, tuple):
            raise TypeError(f"Unsupported type '{type(reactants)}' for `reactants`.")
        if not len(reactants) in (1, 2):
            raise ValueError(f"Can only run reactions with 1 or 2 reactants, not {len(reactants)}.")

        rxn = self.rxn  # TODO: investigate if this is necessary (if not, delete "delete rxn below")

        # Convert all reactants to `Chem.Mol`
        r: Tuple = tuple(self.get_mol(smiles) for smiles in reactants if smiles is not None)

        if self.num_reactant == 1:
            if len(r) == 2:  # Provided two reactants for unimolecular reaction -> no rxn possible
                return None
            if not self.is_reactant(r[0]):
                return None
        elif self.num_reactant == 2:
            # Match reactant order with reaction template
            if self.is_reactant_first(r[0]) and self.is_reactant_second(r[1]):
                pass
            elif self.is_reactant_first(r[1]) and self.is_reactant_second(r[0]):
                r = tuple(reversed(r))
            else:  # No reaction possible
                return None
        else:
            raise ValueError("This reaction is neither uni- nor bi-molecular.")

        # Run reaction with rdkit magic
        ps = rxn.RunReactants(r)

        # Filter for unique products (less magic)
        # Note: Use chain() to flatten the tuple of tuples
        uniqps = list({Chem.MolToSmiles(p) for p in itertools.chain(*ps)})

        # Sanity check
        if not len(uniqps) >= 1:
            # TODO: Raise (custom) exception?
            raise ValueError("Reaction did not yield any products.")

        del rxn

        if keep_main:
            uniqps = uniqps[:1]
        # >>> TODO: Always return list[str] (currently depends on "keep_main")
        uniqps = uniqps[0]
        # <<< ^ delete this line if resolved.
        return uniqps

    def _filter_reactants(
        self, bblocks_smi, bblocks_mol, verbose = False):
        """
        Filters reactants which do not match the reaction.

        Args:
            smiles: Possible reactants for this reaction.

        Returns:
            :lists of SMILES which match either the first
                reactant, or, if applicable, the second reactant.

        Raises:
            ValueError: If `self` is not a uni- or bi-molecular reaction.
        """
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
        """
        Finds applicable reactants from a list of building blocks.
        Sets `self.available_reactants`.

        Args:
            building_blocks: Building blocks as SMILES strings.
        """
        available_reactants, available_reactants_ids = self._filter_reactants(bblocks_smi, bblocks_mol, verbose=verbose)
        self.available_reactants = available_reactants
        self.available_reactants_ids = available_reactants_ids
        return self

    @property
    def get_available_reactants(self) -> Set[str]:
        return {x for reactants in self.available_reactants for x in reactants}

    def asdict(self) -> dict():
        """Returns serializable fields as new dictionary mapping.
        *Excludes* Not-easily-serializable `self.rxn: rdkit.Chem.ChemicalReaction`."""
        import copy

        out = copy.deepcopy(self.__dict__)  # TODO:
        _ = out.pop("rxn")
        return out
