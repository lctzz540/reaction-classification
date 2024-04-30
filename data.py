from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

import torch
from torch_geometric.data import Data
from utils import get_atom_features, get_bond_features


class MolecularGraphDataset(Dataset):
    def __init__(self, csv_file, split="train"):
        self.df = pd.read_csv(csv_file, sep="\t")
        self.df = self.df[self.df["split"] == split]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        smiles = self.df.iloc[idx, self.df.columns.get_loc("rxn")]
        y_val = self.df.iloc[idx, self.df.columns.get_loc("rxn_class")]

        class_components = [int(c) for c in y_val.split(".")]
        max_class = max(class_components)

        y_tensor = torch.zeros(max_class + 1)
        for c in class_components:
            y_tensor[c] = 1

        precursors, products = smiles.split(">>")
        precursors = precursors.split(".")
        products = products.split(".")

        prec_data = self.process_molecule(precursors)
        prod_data = self.process_molecule(products)

        max_nodes = max(prec_data.num_nodes, prod_data.num_nodes)
        max_edges = max(prec_data.num_edges, prod_data.num_edges)

        prec_data = self.pad_graph(prec_data, max_nodes, max_edges)
        prod_data = self.pad_graph(prod_data, max_nodes, max_edges)

        return prec_data, prod_data, y_tensor

    def process_molecule(self, smiles_list):
        mol_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

        n_nodes_total = sum([mol.GetNumAtoms() for mol in mol_list])
        n_edges_total = sum([2 * mol.GetNumBonds() for mol in mol_list])

        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(get_atom_features(
            unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(
            get_bond_features(unrelated_mol.GetBondBetweenAtoms(0, 1))
        )

        X = np.zeros((n_nodes_total, n_node_features))
        EF = np.zeros((n_edges_total, n_edge_features))
        offset = 0

        for mol in mol_list:
            n_nodes = mol.GetNumAtoms()

            for atom in mol.GetAtoms():
                X[offset + atom.GetIdx(), :] = get_atom_features(atom)

            for k, bond in enumerate(mol.GetBonds()):
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                EF[offset +
                    k] = get_bond_features(mol.GetBondBetweenAtoms(i, j))

            offset += n_nodes

        X = torch.tensor(X, dtype=torch.float)

        adjacency_matrices = [GetAdjacencyMatrix(mol) for mol in mol_list]
        rows_list, cols_list = zip(
            *[np.nonzero(adj_mat) for adj_mat in adjacency_matrices]
        )

        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)

        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim=0)

        EF = torch.tensor(EF, dtype=torch.float)

        data = Data(x=X, edge_index=E, edge_attr=EF)
        data.num_node_features = n_node_features
        data.num_edge_features = n_edge_features

        return data

    def pad_graph(self, data, max_nodes, max_edges):
        num_nodes_to_add = max_nodes - data.num_nodes
        num_edges_to_add = max_edges - data.num_edges

        if num_nodes_to_add > 0:
            data.x = torch.cat(
                [data.x, torch.zeros(
                    (num_nodes_to_add, data.num_node_features))]
            )
        if num_edges_to_add > 0:
            data.edge_index = torch.cat(
                [data.edge_index, torch.zeros(
                    (2, num_edges_to_add), dtype=torch.long)],
                dim=1,
            )
            data.edge_attr = torch.cat(
                [
                    data.edge_attr,
                    torch.zeros((num_edges_to_add, data.num_edge_features)),
                ]
            )

        return data
