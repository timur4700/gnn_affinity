import torch
from torch.nn import Module, Linear, Sequential, Dropout, SiLU  
from models.nn.utils import make_normalization,make_act_function, DropoutEdge
from torch_geometric.nn import GraphNorm, GCNConv, GINConv, GINEConv, GATConv, GATv2Conv

from models.nn import utils



class ModuleMPNN(Module):
    def __init__(self,
                 node_hidden_dim: int,
                 edge_attr=False,
                 edge_dim=None,
                 node_feature_dropout_p: float=0.1,
                 edge_dropout_p: float=0.0,
                 normaliztion: str='batchnorm',
                 residual=True,
                 undirected=True
                 ):
        super().__init__()


        self.gnn_conv_module=None
        self.node_hidden_dim = node_hidden_dim

        self.edge_attr = edge_attr
        self.edge_dim = edge_dim

        self.norm = make_normalization(normaliztion)(self.node_hidden_dim)
        self.dropout_feature = Dropout(node_feature_dropout_p)
        self.dropout_edge = DropoutEdge(edge_dropout_p, undirected=undirected)
        self.residual = residual


    def forward(self, x, edge_index, edge_attr=None, batch_idx=None):

        residual = x 

        edge_index, edge_attr_mask = self.dropout_edge(edge_index)
        edge_attr = edge_attr[edge_attr_mask] if self.edge_attr else None

        x = self.gnn_conv_module(x, edge_index, edge_attr) if self.edge_attr else self.gnn_conv_module(x, edge_index)
        x = self.norm(x, batch_idx) if isinstance(self.norm, GraphNorm) else self.norm(x)
        x = self.dropout_feature(x)
 
        if self.residual:
            x = residual + x
            return x
        
        return x



class GCNLayer(ModuleMPNN):
    def __init__(self,
                 node_input_dim: int,
                 node_hidden_dim: int,
                 improved=False,
                 cached=False,
                 add_self_loops=None,
                 normalize_in=True,
                 bias=True,
                 **kwrags):
        super().__init__(node_hidden_dim=node_hidden_dim, **kwrags)

        self.edge_attr = False
        self.gnn_conv_module = GCNConv(in_channels=node_input_dim,
                                       out_channels=node_hidden_dim,
                                       improved=improved,
                                       cached=cached,
                                       add_self_loops=add_self_loops,
                                       normalize=normalize_in,
                                       bias=bias)



class GINlayer(ModuleMPNN):
    def __init__(self, 
                 node_input_dim: int,
                 node_hidden_dim: int,
                 node_output_dim: int,
                 act_function='relu',
                 eps: int|float=0,
                 train_eps=True, **kwargs):
        
        super().__init__(node_hidden_dim=node_hidden_dim, **kwargs)

        self.fc_module = Sequential(Linear(node_input_dim, node_input_dim), 
                                    make_act_function(act_function)(),
                                    Linear(node_hidden_dim, node_output_dim))

        if self.edge_attr:

            assert self.edge_dim, 'Error, edge ffeature input dimension is needed'
            self.gnn_conv_module = GINEConv(nn=self.fc_module,
                                     eps=eps,
                                     edge_dim=self.edge_dim)
            
        else:
            self.gnn_conv_module = GINConv(nn=self.fc_module,
                                    eps=eps,
                                    train_eps=train_eps)



class GATLayer(ModuleMPNN):
    def __init__(self,
                 node_input_dim: int,
                 node_hidden_dim: int,
                 gat_version: int=1,
                 num_heads: int=1,
                 concat=True,
                 negative_slope: float=0.2,
                 add_self_loops=True,
                 bias=True,
                 **kwargs):
        super().__init__(node_hidden_dim=node_hidden_dim, **kwargs)

        if self.edge_attr:
            assert self.edge_dim, 'Error, edge feature input dimension is needed'

        if gat_version == 1:
            self.gnn_conv_module = GATConv(in_channels=node_input_dim,
                                        out_channels=node_hidden_dim,
                                        heads=num_heads,
                                        concat=concat,
                                        edge_dim=self.edge_dim if self.edge_attr else None,
                                        add_self_loops=add_self_loops,
                                        negative_slope=negative_slope,
                                        bias=bias)
            
        elif gat_version == 2:
             self.gnn_conv_module = GATv2Conv(in_channels=node_input_dim,
                                        out_channels=node_hidden_dim,
                                        heads=num_heads,
                                        concat=concat,
                                        edge_dim=self.edge_dim if self.edge_attr else None,
                                        add_self_loops=add_self_loops,
                                        negative_slope=negative_slope,
                                        bias=bias)    

        else:
            raise ValueError('Only GAT versions 1 and 2 are available')

class EGNNConv(Module):
        def __init__(self, 
                     input_dim, 
                     hidden_dim, 
                     output_dim, 
                     cutoff=5, 
                     n_rbf=32, 
                     edge_attr=False, 
                     cosine_cutoff=True, 
                     velocity=False, 
                     weight=False):
            
            super().__init__()

            if edge_attr:
                self.edge_function = Sequential(Linear(input_dim*4, hidden_dim), 
                                                SiLU(), 
                                                Linear(hidden_dim, hidden_dim))

            else:
                self.edge_function = Sequential(Linear(input_dim*3, hidden_dim), 
                                                SiLU(), 
                                                Linear(hidden_dim, hidden_dim))

            self.edge_attr = edge_attr

            self.coordinate_function = Sequential(Linear(hidden_dim, hidden_dim), 
                                                  SiLU(), 
                                                  Linear(hidden_dim, 1))
            
            self.node_function = Sequential(Linear(hidden_dim + input_dim, hidden_dim), 
                                            SiLU(), 
                                            Linear(hidden_dim, output_dim))

            self.cutoff = cutoff
            self.cosine_cutoff = cosine_cutoff
            self.velocity = velocity
            self.weight = weight

            self.rbf = utils.RBF(n_rbf, cutoff=cutoff)
            self.dist_embedd = utils.mlp(n_rbf, hidden_dim, hidden_dim)


        def forward(self, 
                    x: torch.Tensor, 
                    edge_index: torch.Tensor, 
                    coords: torch.Tensor, 
                    edge_attr: torch.Tensor=None):

            device = x.device

            i, j = edge_index #[E]
            h_i, h_j = x[i], x[j] #[E, n]
            x_i, x_j = coords[i], coords[j] #[E, 3]
            dist = torch.sqrt(utils.distance(x_i, x_j).clamp(min=1e-12)) #[E, 1]
            dist = dist.clamp(min=1e-6)
            if self.cosine_cutoff:
                cosine_weight = utils.cosine_cutoff(dist, cutoff=self.cutoff)

            dist = self.rbf(dist  * cosine_weight)
            dist = self.dist_embedd(dist)

            if self.edge_attr:
                m_ij = self.edge_function(torch.cat([h_i, h_j, dist, edge_attr], dim=-1)) # [E, n*3 + 1]

            else:
                m_ij = self.edge_function(torch.cat([h_i, h_j, dist], dim=-1)) # [E, n*2 + 1]



            r = x_i - x_j # [E, 3]
            r = r #/ torch.sqrt(gnn_func.distance(x_i, x_j))
            weight = self.coordinate_function(m_ij) # [E, 1]


            i_expand = i.unsqueeze(-1).broadcast_to(m_ij.shape) #[E, n*2 + 1]
            i_expand_coord = i.unsqueeze(-1).broadcast_to(m_ij.shape[0], 3)
            coord_update = r * weight # [E, 3]


            summed_m_ij = torch.zeros_like(coords, device=device).scatter_add_(0, i_expand_coord, coord_update) #[E, 3]
            degree = torch.bincount(i, minlength=x.size(0)).float().unsqueeze(-1)
            C = 1 / (degree + 1e-8)
            coords_new = coords + C * summed_m_ij


            m_i = torch.zeros(x.shape[0], m_ij.shape[1], device=device, dtype=m_ij.dtype).scatter_add_(0, i_expand, m_ij)
            x_new = self.node_function(torch.cat([x, m_i], dim=-1))
            x = x + x_new

            return x, coords_new


class EGNNConvBlock(Module):

        def __init__(self, 
                     n_rbf, 
                     input_dim, 
                     hidden_dim, 
                     output_dim, 
                     normalization='batchnorm', 
                     cutoff=5, 
                     edge_attr=False, 
                     velocity=False, 
                     weight=False):

            
            super().__init__()

            self.egnn_conv = EGNNConv(input_dim, 
                                      hidden_dim, 
                                      output_dim, 
                                      cutoff=cutoff, 
                                      edge_attr=edge_attr, 
                                      n_rbf=n_rbf, 
                                      velocity=velocity, 
                                      weight=weight)


            self.norm = utils.make_normalization(normalization)(hidden_dim)
            self.norm_name = normalization


        def forward(self, x, edge_index, pos, edge_attr=None, batch_idx=None):

            x_layer, pos = self.egnn_conv(x, edge_index, pos, edge_attr)

            if self.norm_name == 'graphnorm':
                x_layer = self.norm(x_layer, batch_idx)

            elif self.norm_name == 'layernorm':
              x_layer = self.norm(x_layer, batch_idx)

            else:
                x_layer = self.norm(x_layer)

            return x_layer, pos