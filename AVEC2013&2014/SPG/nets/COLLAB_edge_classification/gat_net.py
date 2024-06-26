import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""
from layers.gat_layer import GATLayer, CustomGATLayer, CustomGATLayerEdgeReprFeat, CustomGATLayerIsotropic
from layers.mlp_readout_layer import MLPReadout

class GATNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        in_dim_edge = net_params['in_dim_edge']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        self.device = net_params['device']
        
        self.layer_type = {
            "dgl": GATLayer,
            "edgereprfeat": CustomGATLayerEdgeReprFeat,
            "edgefeat": CustomGATLayer,
            "isotropic": CustomGATLayerIsotropic,
        }.get(net_params['layer_type'], GATLayer)
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim * num_heads)
        
        if self.layer_type != GATLayer:
            self.edge_feat = net_params['edge_feat']
            self.embedding_e = nn.Linear(in_dim_edge, hidden_dim * num_heads)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([self.layer_type(hidden_dim * num_heads, hidden_dim, num_heads,
                                                     dropout, self.batch_norm, self.residual) for _ in range(n_layers-1)])
        self.layers.append(self.layer_type(hidden_dim * num_heads, out_dim, 1, dropout, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(2*out_dim, 1)
        
    def forward(self, g, h, e):
        h = self.embedding_h(h.float())
        h = self.in_feat_dropout(h)
        
        if self.layer_type == GATLayer:
            for conv in self.layers:
                h = conv(g, h)
        else:
            if not self.edge_feat:
                e = torch.ones_like(e).to(self.device)
            e = self.embedding_e(e.float())
            
            for conv in self.layers:
                h, e = conv(g, h, e)
        
        g.ndata['h'] = h
        
        return h
    
    def edge_predictor(self, h_i, h_j):
        x = torch.cat([h_i, h_j], dim=1)
        x = self.MLP_layer(x)
        
        return torch.sigmoid(x)
    
    def loss(self, pos_out, neg_out):
        pos_loss = -torch.log(pos_out + 1e-15).mean()  # positive samples
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()  # negative samples
        loss = pos_loss + neg_loss
        
        return loss
# net_params= {
#         "L": 3,
#         "hidden_dim": 19,
#         "out_dim": 76,
#         "residual": True,
#         "readout": "mean",
#         "n_heads": 4,
#         "in_feat_dropout": 0.0,
#         "dropout": 0.0,
#         "batch_norm": True,
#         "self_loop": False,
#         "layer_type": "dgl",
#         'in_dim':28,
#         'in_dim_edge':28,
#         "device": 'cpu'
#     }
# model = GATNet(net_params)
# print(model)