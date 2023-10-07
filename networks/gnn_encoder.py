import torch
from torch import nn
import torch.nn.functional as F


class GNNLayer(nn.Module):
    def __init__(self, **model_params):
        super(GNNLayer, self).__init__()
        hidden_dim = model_params['embedding_dim']
        self.aggregation = model_params['aggregation']

        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.A = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.B = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.norm_h = nn.BatchNorm1d(hidden_dim, affine=True, track_running_stats=False)
        self.norm_e = nn.BatchNorm1d(hidden_dim, affine=True, track_running_stats=False)


    def forward(self, h, e):
        batch_size, num_nodes, hidden_dim = h.shape
        h_in = h
        e_in = e

        # Linear transformations for node update
        Uh = self.U(h)  # B x V x H
        Vh = self.V(h).unsqueeze(1).expand(-1, num_nodes, -1, -1)  # B x V x V x H

        # Linear transformations for edge update and gating
        Ah = self.A(h)  # B x V x H
        Bh = self.B(h)  # B x V x H
        Ce = self.C(e)  # B x V x V x H

        # Update edge features and compute edge gates
        e = Ah.unsqueeze(1) + Bh.unsqueeze(2) + Ce  # B x V x V x H
        gates = torch.sigmoid(e)  # B x V x V x H

        # Update node features
        h = Uh + self.aggregate(Vh, gates)  # B x V x H

        # Normalize node features
        h = self.norm_h(h.view(batch_size * num_nodes, hidden_dim)).view(batch_size, num_nodes, hidden_dim)

        # Normalize edge features
        e = self.norm_e(e.view(batch_size * num_nodes * num_nodes, hidden_dim)).view(batch_size, num_nodes, num_nodes, hidden_dim)

        # Apply non-linearity
        h = F.relu(h)
        e = F.relu(e)

        # Make residual connection
        h = h_in + h
        e = e_in + e

        return h, e

    def aggregate(self, Vh, gates):
        # Perform feature-wise gating mechanism
        Vh = gates * Vh  # B x V x V x H

        if self.aggregation == "mean":
            return torch.sum(Vh, dim=2) / Vh.shape[1]
        elif self.aggregation == "max":
            return torch.max(Vh, dim=2)[0]
        else:
            return torch.sum(Vh, dim=2)


class GNN_Encoder(nn.Module):
    def __init__(self, **model_params):
        super(GNN_Encoder, self).__init__()
        self.init_node_embed = nn.Linear(model_params['node_dim'], model_params['embedding_dim'], bias=True)
        self.init_edge_embed = nn.Linear(model_params['edge_dim'], model_params['embedding_dim'], bias=True)
        self.layers = nn.ModuleList([
            GNNLayer(**model_params)
            for _ in range(model_params['n_encode_layers'])
        ])

    def forward(self, nodes, edges):
        # Embed node and edge features
        x = self.init_node_embed(nodes)
        e = self.init_edge_embed(edges)
        for layer in self.layers:
            x, e = layer(x, e)
        return x, e
