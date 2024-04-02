from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

import torch
from torch_geometric.data import Data
from utils import get_atom_features, get_bond_features


class MolecularGraphDataset(Dataset):
    def __init__(self, csv_file, split="train", num_classes=None):
        self.df = pd.read_csv(csv_file, sep="\t")
        self.df = self.df[self.df["split"] == split]
        self.num_classes = num_classes

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

        prec_data = self.process_molecule(precursors)
        prod_data = self.process_molecule(products)

        max_nodes = max(prec_data.num_nodes, prod_data.num_nodes)
        max_edges = max(prec_data.num_edges, prod_data.num_edges)

        prec_data = self.pad_graph(prec_data, max_nodes, max_edges)
        prod_data = self.pad_graph(prod_data, max_nodes, max_edges)

        return prec_data, prod_data, y_tensor

    def process_molecule(self, smiles):
        mol = Chem.MolFromSmiles(smiles)

        n_nodes = mol.GetNumAtoms()
        n_edges = 2 * mol.GetNumBonds()
        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(get_atom_features(
            unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(
            get_bond_features(unrelated_mol.GetBondBetweenAtoms(0, 1))
        )

        X = np.zeros((n_nodes, n_node_features))

        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)

        X = torch.tensor(X, dtype=torch.float)

        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim=0)

        EF = np.zeros((n_edges, n_edge_features))

        for k, (i, j) in enumerate(zip(rows, cols)):
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j)))

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
