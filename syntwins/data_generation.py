import re
import copy
import numpy as np
from rdkit import Chem
from syntwins.chem_utils import *

def demap(smi):
    mol = Chem.MolFromSmiles(smi)
    [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
    return Chem.MolToSmiles(mol)

def sample_bb(building_blocks):
    idx = np.random.choice(len(building_blocks))
    return building_blocks[idx]

def extract_map_numbers(reaction_string):
    reactants = reaction_string.split('>>')[0]
    reactant_list = reactants.split('.')
    map_number_groups = []
    for reactant in reactant_list:
        map_numbers = re.findall(r':(\d+)]', reactant)
        map_number_groups.append(tuple(map(int, map_numbers)))
    return map_number_groups

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


def reactions2seq(reactions):
    seq_list = []
    for i, reaction in enumerate(reactions):
        reactant, ref, product = reaction.split('>')
        reactant = demap(reactant)
        product = demap(product)
        if i != 0:
            reactant = '.'.join([smi for smi in reactant.split('.') if smi != prev_product])
        if reactant != '':
            seq_list.append(reactant)
        seq_list.append(ref)
        prev_product = product
    return product, ';'.join(seq_list)

def sample_rxn(smi, rxns, taget_ref=None):
    filtered_rxns = []
    mol = Chem.MolFromSmiles(smi)
    if taget_ref:
        rxn = rxns[taget_ref]
        for j, smarts in enumerate(rxn.reactant_template):
            pattern = Chem.MolFromSmarts(smarts)
            if mol.HasSubstructMatch(pattern):
                filtered_rxns.append((taget_ref, rxn, j))
    else:
        for i, (ref, rxn) in enumerate(rxns.items()):
            for j, smarts in enumerate(rxn.reactant_template):
                pattern = Chem.MolFromSmarts(smarts)
                if mol.HasSubstructMatch(pattern):
                    filtered_rxns.append((ref, rxn, j))
        
    if len(filtered_rxns) == 0:
        return 0, []
    ref, rxn, rxn_pos = filtered_rxns[np.random.choice(len(filtered_rxns))]
    if rxn_pos == 0:
        if len(rxn.available_reactants) == 1:
            sampled_smis = [smi]
        else:
            if len(rxn.available_reactants[1]) == 0:
                return 0, []
            other_smi = sample_bb(rxn.available_reactants[1])
            sampled_smis = [smi, other_smi]
    elif rxn_pos == 1:
        if len(rxn.available_reactants[0]) == 0:
                return 0, []
        other_smi = sample_bb(rxn.available_reactants[0])
        sampled_smis = [other_smi, smi]
        
    mapped_rxns = generate_mapped_rxns(rxn, sampled_smis)
    return ref, mapped_rxns

def synthesis(bblocks, rxns, start_ref=None, max_step=5):
    if start_ref:
        bb_position = np.random.choice(len(rxns[start_ref].available_reactants))
        smi = sample_bb(rxns[start_ref].available_reactants[bb_position])
    else:
        smi = sample_bb(bblocks)
    reactions = []
    n_step = 0
    patience = 0
    while n_step < max_step and patience < 5:
        if n_step >= 3 and np.random.random_sample() >= 0.5:
            break
        if n_step == 0 and start_ref:
            ref, sampled_rxns = sample_rxn(smi, rxns, start_ref)
        else:
            ref, sampled_rxns = sample_rxn(smi, rxns)
        if len(sampled_rxns) == 0:
            patience += 1
            if n_step == 0:
                continue
            else:
                break
        else:
            patience = 0
        reactant, product = sampled_rxns[0].split('>>')
        if len(Chem.MolFromSmiles(product).GetAtoms()) >= 80:
            break
            
        n_step += 1
        smi = product
        reactions.append('%s>%s>%s' % (reactant, ref, product))
        
    if patience == 5:
        return None
    else:
        return reactions
    