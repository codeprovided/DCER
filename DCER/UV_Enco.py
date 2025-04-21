import torch
import torch.nn as nn
import torch.nn.functional as F


class UV_Encoder(nn.Module):

    def __init__(self, features, embed_dim, aggregator, cuda, uv=True):
        super(UV_Encoder, self).__init__()

        self.features = features  # u2e/v2e
        self.uv = uv
        # self.history_uv_lists = history_uv_lists
        # self.history_ra_lists = history_ra_lists
        # self.history_re_lists = history_re_lists
        self.aggregator = aggregator
        self.embed_dim = embed_dim
        self.device = cuda
        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)

    def forward(self, nodes, history_uv_lists, history_ra_lists):
        tmp_history_uv = []
        tmp_history_ra = []
        for node in nodes:
            tmp_history_uv.append(history_uv_lists[int(node)])
            tmp_history_ra.append(history_ra_lists[int(node)])

        neigh_feats = self.aggregator.forward(nodes, tmp_history_uv, tmp_history_ra)  # user-item network

        self_feats = self.features.weight[nodes]
        # self-connection could be considered.
        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(self.linear1(combined))

        return combined