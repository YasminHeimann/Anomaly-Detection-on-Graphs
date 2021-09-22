import torch.nn as nn
import torch.nn.functional as F

from dgl.nn import GraphConv, AGNNConv


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = AGNNConv(h_feats, h_feats)
        self.conv3 = AGNNConv(h_feats, num_classes)

    def forward(self, g):
        in_feat = g.ndata['feat']

        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.conv3(g, h)
        return h

