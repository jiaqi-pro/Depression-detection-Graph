
import torch
import dgl
from gat_net import GATNet
from icecream import ic
def build_graph(x):
    m = x.shape[0]
    u = []
    v = []
    for i in range(m):
        count = 0
        for j in range(m):
            u.append(i)
            v.append(j)
            count += 1

    g = dgl.graph((u,v))
    return g




net_params = {
    'num_atom_type': 28, # input_dim
    'hidden_dim': 7,
    'out_dim':42,
    'L':4,
    'readout':'mean',
    'residual': True,
    'edge_feat': True,
    'device': 'cpu',
    'pos_enc':False,
    'batch_norm':False,
    'layer_type':'edgereprfeat',
    'in_feat_dropout':0.0,
    'dropout':0.0,
    'n_heads' : 6
}


#
model = GATNet(net_params)
print(model)
input_data = torch.rand([256,28]) # [256,28] keep the input_data same as the net_params['num_atom_type']
G = build_graph(input_data)
e = torch.ones(G.num_edges(),1).long()
result = model(G,input_data,e)
ic(input_data.shape,result.shape)