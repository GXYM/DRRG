from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, features, A):
        x = torch.bmm(A, features)
        return x

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(self.in_features, self.out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(self.out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.agg = MeanAggregator()

    def forward(self, h, adj):
        h = h.to(self.W.device)
        adj = adj.to(self.W.device)
        # print("===="*5)
        # print('h.device',h.device)
        # print('adj.device',adj.device)
        # print('self.W.device',self.W.device)
        # print("===="*5)
    
        if self.concat:
            GX = self.agg(h,adj).to(self.W.device)
            # print('GX.device',GX.device)
            # input()
            Wh = torch.einsum('ijk,ko->ijo', (GX, self.W))
        else:
            Wh = torch.einsum('ijk,ko->ijo', (h, self.W))
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        e = torch.matmul(Wh,self.a)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        self.agg = MeanAggregator()
        self.predictor = nn.Sequential(
            nn.Linear(nhid ,32),
            nn.PReLU(32),
            nn.Linear(32, nclass))
    def forward(self, X, adj,one_hop_idcs, train=True):
        B, N, D = X.shape
        k1 = one_hop_idcs.size(-1)


        X = F.dropout(X, self.dropout, training=self.training)

        X = torch.cat([att( X ,adj) for att in self.attentions], dim=2)

        X = F.dropout(X, self.dropout, training=self.training)

        X = F.elu(self.out_att(X, adj))
        
        dout = X.size(-1)
        edge_feat = torch.zeros(B, k1, dout).cuda()
        for b in range(B):
            edge_feat[b, :, :] = X[b, one_hop_idcs[b]]
        edge_feat = edge_feat.view(-1, dout)
        X = self.predictor(edge_feat)
        return X
