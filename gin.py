import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, global_mean_pool


class GIN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super().__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(79, hidden),
                # Linear(dataset.num_features, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden),
            ),
            train_eps=True,
        )
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                    ),
                    train_eps=True,
                )
            )
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, 50)
        # self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data, ret_repr=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x_repr = global_mean_pool(x, batch)
        if ret_repr:
            return x_repr
        x = F.relu(self.lin1(x_repr))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class ContraGIN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super().__init__()
        self.gin = GIN(dataset, num_layers, hidden)

    def forward(self, precursor_data, product_data, ret_repr=False):
        precursor_outputs = []
        product_outputs = []

        for prec_data in precursor_data:
            precursor_outputs.append(self.gin(prec_data, ret_repr))

        for prod_data in product_data:
            product_outputs.append(self.gin(prod_data, ret_repr))

        return precursor_outputs, product_outputs
