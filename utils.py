import os
import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from tqdm import tqdm
from rdkit.Chem.rdmolops import GetAdjacencyMatrix


def one_hot_encoding(x, permitted_list):
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding


def generateMolFromSmiles(smiles):
    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)
    AllChem.EmbedMolecule(m, randomSeed=0xF00D)
    AllChem.MMFFOptimizeMolecule(m)
    return m


def get_atom_features(atom, use_chirality=True, hydrogens_implicit=True):
    permitted_list_of_atoms = [
        "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe",
        "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd", "Co",
        "Se", "Ti", "Zn", "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn", "Zr",
        "Cr", "Pt", "Hg", "Pb", "Unknown"
    ]
    if not hydrogens_implicit:
        permitted_list_of_atoms = ["H"] + permitted_list_of_atoms

    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()),
                                              ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    is_in_a_ring_enc = [int(atom.IsInRing())]
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    atomic_mass_scaled = [float((atom.GetMass() - 10.812) / 116.092)]
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6)]
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64) / 0.76)]

    atom_feature_vector = (
            atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc +
            is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled
    )

    if use_chirality:
        chirality_type_enc = one_hot_encoding(
            str(atom.GetChiralTag()),
            ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"]
        )
        atom_feature_vector += chirality_type_enc

    if hydrogens_implicit:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector)


def get_bond_features(bond, use_stereochemistry=True):
    permitted_list_of_bond_types = [
        Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC
    ]
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc

    if use_stereochemistry:
        stereo_type_enc = one_hot_encoding(
            str(bond.GetStereo()),
            ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"]
        )
        bond_feature_vector += stereo_type_enc

    return np.array(bond_feature_vector)


def process_molecule(smiles_list):
    mol_list = [generateMolFromSmiles(smiles) for smiles in smiles_list]

    n_nodes_total = sum([mol.GetNumAtoms() for mol in mol_list])
    n_edges_total = sum([2 * mol.GetNumBonds() for mol in mol_list])

    unrelated_smiles = "O=O"
    unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
    n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
    n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0, 1)))

    X = np.zeros((n_nodes_total, n_node_features))
    EF = np.zeros((n_edges_total, n_edge_features))
    offset = 0

    for mol in mol_list:
        n_nodes = mol.GetNumAtoms()
        for atom in mol.GetAtoms():
            X[offset + atom.GetIdx(), :] = get_atom_features(atom)
        for k, bond in enumerate(mol.GetBonds()):
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            EF[offset + k] = get_bond_features(mol.GetBondBetweenAtoms(i, j))
        offset += n_nodes

    X = torch.tensor(X, dtype=torch.float)
    adjacency_matrices = [GetAdjacencyMatrix(mol) for mol in mol_list]
    rows_list, cols_list = zip(*[np.nonzero(adj_mat) for adj_mat in adjacency_matrices])

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)

    torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
    torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
    E = torch.stack([torch_rows, torch_cols], dim=0)

    EF = torch.tensor(EF, dtype=torch.float)
    data = Data(x=X)
    print(data)

    return data

def pad_graph(data, max_nodes, max_edges):
    num_nodes_to_add = max_nodes - data.num_nodes
    num_edges_to_add = max_edges - data.num_edges

    if num_nodes_to_add > 0:
        data.x = torch.cat([data.x, torch.zeros((num_nodes_to_add, data.x.shape[1]))])
    if num_edges_to_add > 0:
        data.edge_index = torch.cat([data.edge_index, torch.zeros((2, num_edges_to_add), dtype=torch.long)], dim=1)
        data.edge_attr = torch.cat([data.edge_attr, torch.zeros((num_edges_to_add, data.edge_attr.shape[1]))])

    return data


def process_reactions(df, idx, col):
    smiles = df.iloc[idx, df.columns.get_loc(col)]
    precursors, products = smiles.split(">>")
    precursors = precursors.split(".")
    products = products.split(".")

    prec_data = process_molecule(precursors)
    prod_data = process_molecule(products)

    max_nodes = max(prec_data.num_nodes, prod_data.num_nodes)
    max_edges = max(prec_data.num_edges, prod_data.num_edges)

    prec_data.num_node_features = prec_data.x.shape[1]
    prec_data.num_edge_features = prec_data.edge_attr.shape[1]
    prod_data.num_node_features = prod_data.x.shape[1]
    prod_data.num_edge_features = prod_data.edge_attr.shape[1]

    prec_data = pad_graph(prec_data, max_nodes, max_edges)
    prod_data = pad_graph(prod_data, max_nodes, max_edges)

    return prec_data, prod_data


def process_data(filename):
    processed_file_path = os.path.join("./processed", filename + ".pt")
    if os.path.exists(processed_file_path):
        return torch.load(processed_file_path)

    data_list = []
    raw_file_path = os.path.join("./raw", filename + ".csv")
    df = pd.read_csv(raw_file_path)

    for index, _ in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            input_data = process_reactions(df, index, "reactions")
            output_data = process_reactions(df, index, "ground_truth")
            data_list.append([input_data, output_data])
        except Exception as e:
            print(f"Error processing row {index}: {e}")

    if data_list:
        torch.save(data_list, processed_file_path)

    return data_list
