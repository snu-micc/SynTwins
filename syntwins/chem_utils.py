import re
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

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

def sanitize(mol, canonical=True):
    mol = fix_aromatic(mol)
    smi = Chem.MolToSmiles(mol, canonical=canonical)
    return Chem.MolFromSmiles(smi)

def sanitize_mols(mols):
    return [sanitize(mol) for mol in mols if mol]

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
    
def get_fps(mol, _radius=2, _nBits=-1, useChirality=False, return_array=True):
    if mol is None:
        return np.zeros(_nBits)
    elif isinstance(mol, str):  # smiles input
        mol = Chem.MolFromSmiles(mol)
    if _nBits > 0:
        fps = AllChem.GetMorganFingerprintAsBitVect(mol, _radius, _nBits, useChirality=useChirality)
    else:
        fps = AllChem.GetMorganFingerprint(mol, _radius, useChirality=useChirality)
        
    if return_array:
        array = np.zeros((_nBits,))
        DataStructs.ConvertToNumpyArray(fps, array)
        return array
    else:
        return fps

def get_sim(mol1, mol2, _nBits=-1, useChirality=True, alpha=0.5, beta=0.5):
    fps1 = get_fps(mol1, _nBits=_nBits, useChirality=useChirality, return_array=False)
    fps2 = get_fps(mol2, _nBits=_nBits, useChirality=useChirality, return_array=False)
    similarity = DataStructs.TanimotoSimilarity(fps1,fps2)
#     similarity = DataStructs.TverskySimilarity(fps1, fps2, alpha, beta)
    return similarity
