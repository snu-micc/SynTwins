import re
import numpy as np
from rdkit import Chem, DataStructs

def sanitize(mol, canonical=True):
    return Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=canonical))

def sanitize_mols(mols):
    mols = [Chem.MolFromSmiles(Chem.MolToSmiles(mol)) for mol in mols]
    return [mol for mol in mols if mol]

def fix_aromatic(mol):
    for atom in mol.GetAtoms():
        if not atom.IsInRing() and atom.GetIsAromatic():
            atom.SetIsAromatic(False)
        
    for bond in mol.GetBonds():
        if not bond.IsInRing():
            bond.SetIsAromatic(False) 
            if str(bond.GetBondType()) == 'AROMATIC':
                bond.SetBondType(Chem.rdchem.BondType.SINGLE)
    return mol

def demap_reactants(reactants, products):
    rmol, pmol = Chem.MolFromSmiles(reactants), Chem.MolFromSmiles(products)
    pmaps = [atom.GetAtomMapNum() for atom in pmol.GetAtoms()]
    [atom.SetAtomMapNum(0) for atom in rmol.GetAtoms() if atom.GetAtomMapNum() not in pmaps]
    reactants = Chem.MolToSmiles(rmol, canonical=False)
    return reactants, products

def isvalid_template(template):
    r_smarts, a_smarts, p_smarts = template.strip().split('>')
    r_mol = Chem.MolFromSmarts(r_smarts)
    p_mol = Chem.MolFromSmarts(p_smarts)
    r_mapno = set([atom.GetAtomMapNum() for atom in r_mol.GetAtoms() if atom.GetAtomMapNum() != 0])
    p_mapno = set([atom.GetAtomMapNum() for atom in p_mol.GetAtoms()])
    return all([mapno in r_mapno for mapno in p_mapno])

def extract_map_numbers(reaction_string):
    reactants = reaction_string.split('>>')[0]
    reactant_list = reactants.split('.')
    map_number_groups = []
    for reactant in reactant_list:
        map_numbers = re.findall(r':(\d+)]', reactant)
        map_number_groups.append(tuple(map(int, map_numbers)))
    return map_number_groups

def get_fps(smi, _radius=2, _nBits=1024, useChirality=False, return_array=True) -> np.ndarray:  # dtype=int64
    if smi is None:
        return np.zeros(_nBits)
    else:
        mol = Chem.MolFromSmiles(smi)
        features_vec = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, _radius, _nBits, useChirality=useChirality)
        if return_array:
            return np.array(
                features_vec
            )  # TODO: much slower compared to `DataStructs.ConvertToNumpyArray` (20x?) so deprecates
        else:
            return features_vec

def get_sim(smi1, smi2, _nBits=1024, useChirality=False, alpha=0.5, beta=0.5):
    mol1, mol2  = Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2)
    fps1, fps2 = get_fps(smi1, _nBits=_nBits, useChirality=useChirality, return_array=False), get_fps(smi2, _nBits=_nBits, useChirality=useChirality, return_array=False)
    similarity = DataStructs.TverskySimilarity(fps1,fps2, alpha, beta)
    return similarity

def atomic_weighted_mean(smiles, bb_dist):
    dists = []
    for smi, dist in zip(smiles.split('.'), bb_dist):
        mol = Chem.MolFromSmiles(smi)
        dists += [dist]*mol.GetNumAtoms()
    return np.mean(dists)

def get_unique_smi(list_of_smiles, known_smiles=[]):
    unique_smi = []
    for smi in list_of_smiles:
        if isinstance(smi, list):
            smi_list = smi
        elif isinstance(smi, str):
            smi_list = smi.split('.')  
        for s in smi_list:
            mol = Chem.MolFromSmiles(s)
            if s not in unique_smi+known_smiles and mol:
                unique_smi.append(s)
    return unique_smi

def sort_analogs(analogs, sort_by='similarity'):
    if sort_by == 'similarity':
        return {k: v for k, v in sorted(analogs.items(), key=lambda x: -x[1][sort_by])}
    elif sort_by == 'distance':
        return {k: v for k, v in sorted(analogs.items(), key=lambda x: x[1][sort_by])}
    else:
        return analogs
    
    