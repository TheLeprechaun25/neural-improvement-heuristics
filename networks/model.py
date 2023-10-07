import torch
from torch import nn
from networks.gnn_encoder import GNN_Encoder
from networks.mlp_decoder import MLP_Decoder


class Model(nn.Module):
    def __init__(self, **model_params):
        super(Model, self).__init__()
        self.encoder = GNN_Encoder(**model_params)
        self.decoder = MLP_Decoder()

    def forward(self, batch, mask, problem, solutions=None):
        nodes, edges = batch
        batch_size, n, _ = edges.shape

        if problem == 'prp':
            edges = torch.stack([torch.triu(edges, 1), torch.triu(edges.permute(0, 2, 1), 1)])
            edges = edges.permute(1, 2, 3, 0)
        elif problem == 'tsp':
            if not hasattr(self, 'distance_indices'):
                distance_indices = torch.zeros((n, n), dtype=torch.bool)
                for i in range(n - 1):
                    distance_indices[i, i + 1] = True
                    distance_indices[i + 1, i] = True
                distance_indices[-1, 0] = True
                distance_indices[0, -1] = True
                self.distance_indices = distance_indices
            edges = torch.stack((self.distance_indices.repeat(edges.shape[0], 1, 1),
                                ~self.distance_indices.repeat(edges.shape[0], 1, 1), edges), -1)
        elif problem == 'gpp':
            assert solutions is not None
            edge_features = []
            for b in range(batch_size):
                e = torch.zeros(n, n, 2)
                sol = solutions[b, :, :]
                idx_mat = torch.zeros(n, n, dtype=torch.bool)
                for i in range(n // 2):
                    for j in range(n // 2):
                        idx_mat[sol[0, i], sol[1, j]] = True
                        idx_mat[sol[1, j], sol[0, i]] = True

                e[:, :, 0] = edges[b, :, :] * idx_mat
                e[:, :, 1] = edges[b, :, :] * ~idx_mat
                edge_features.append(e)

            edges = torch.stack(edge_features)

        # ENCODER
        node_embedding, edge_embedding = self.encoder(nodes, edges)
        edge_embedding = edge_embedding.view(edge_embedding.shape[0], -1, edge_embedding.shape[-1])

        # DECODER
        log_p = self.decoder(edge_embedding, mask)

        return log_p
