import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    MLP Layer used after graph vector representation
"""

# class MLPReadout(nn.Module):

#     def __init__(self, input_dim, output_dim, L=1): #L=nb_hidden_layers
#         super().__init__()
#         list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
#         list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
#         self.FC_layers = nn.ModuleList(list_FC_layers)
#         # self.sigmod = nn.functional.sigmoid
#         self.L = L
        
#     def forward(self, x):
#         y = x
#         for l in range(self.L):
#             y = self.FC_layers[l](y)
#             y = F.relu(y)
#         y = self.FC_layers[self.L](y)
#         # y = self.sigmod(y)
#         return y
    
    
class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=1): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim , input_dim//2 , bias=True )  ]
        list_FC_layers.append(nn.Linear( input_dim//2 , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        # self.sigmod = nn.functional.sigmoid

        
    def forward(self, x):
        y = x
        y = self.FC_layers[0](y)
        y = F.relu(y)
        y = self.FC_layers[-1](y)
        # y = self.sigmod(y)
        return y