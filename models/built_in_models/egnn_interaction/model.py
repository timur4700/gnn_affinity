import torch
from torch.nn import (Module, 
                      Embedding, 
                      ModuleList, 
                      Linear, 
                      Sequential, 
                      ReLU,
                      Dropout)

#from torch_geometric.utils import subgraph

from models.nn import utils as utils_general, mpnn
from models.built_in_models.egnn_interaction import utils


class EgnnInteraction(Module):
    def __init__(self,
                 n_rbf=32,
                 hidden_dim=64,
                 output_dim=1,
                 cutoff=10,
                 dropout=0.1,
                 n_gine=2,
                 n_egnn=2,
                 normalization='batchnorm'):

        super().__init__()



        self.atom_embedd = utils.AtomEmbeddingLayer(101, hidden_dim)
        self.rbf = utils_general.RBF(n_rbf, cutoff=cutoff)
        self.cutoff = cutoff
        self.bond_embedd = Embedding(3, hidden_dim)


        self.gine_ligand = ModuleList([mpnn.GINlayer(node_input_dim=hidden_dim,
                                                     node_hidden_dim=hidden_dim,
                                                     node_output_dim=hidden_dim,
                                                     act_function='relu', 
                                                     eps=0,
                                                     train_eps=True,
                                                     normaliztion=normalization,
                                                     residual=True,
                                                     node_feature_dropout_p=dropout)
                                                     for _ in range(n_gine)])




        self.gine_protein = ModuleList([mpnn.GINlayer(node_input_dim=hidden_dim,
                                                     node_hidden_dim=hidden_dim,
                                                     node_output_dim=hidden_dim,
                                                     act_function='relu', 
                                                     eps=0,
                                                     train_eps=True,
                                                     normaliztion=normalization,
                                                     residual=True,
                                                     node_feature_dropout_p=dropout) 

                                                     for _ in range(n_gine)])





        self.egnn_interaction =  ModuleList([mpnn.EGNNConvBlock(input_dim=hidden_dim,
                                                                hidden_dim=hidden_dim,
                                                                output_dim=hidden_dim,
                                                                cutoff=cutoff,
                                                                n_rbf=n_rbf,
                                                                normalization=normalization,
                                                                weight=True, edge_attr=False)

                                                                for _ in range(n_egnn)])



        self.dist_embedd_ligand = utils_general.mlp(n_rbf, hidden_dim, hidden_dim)
        self.dist_embedd_protein = utils_general.mlp(n_rbf, hidden_dim, hidden_dim)
        self.dist_embedd_inter = utils_general.mlp(n_rbf, hidden_dim, hidden_dim)


        self.affinity_head = Sequential(Linear(hidden_dim*3, hidden_dim*3),
                                      ReLU(),
                                      Linear(hidden_dim*3, hidden_dim*2),
                                      ReLU(),
                                      Linear(hidden_dim*2, hidden_dim*2),
                                      ReLU(),
                                      Linear(hidden_dim*2, output_dim))



        self.dropout = Dropout(dropout)
        self.dropout_prob = dropout

        self.pool = utils_general.make_pooling('sum')

        self.hidden_dim = hidden_dim
        self.n_layers = n_egnn


    def forward(self, batch):

        x = batch.x
        mol_mask = batch.mol_id.squeeze(-1)
        edge_index = batch.edge_index
        edge_attr = batch.edge_type.squeeze(-1)
        batch_idx = batch.batch
        pos = batch.pos

        node_idx = torch.arange(0, x.size(0), dtype=torch.long, device=x.device)


        mask_edge_nonbonded = edge_attr == 0 # / Nonbonded
        mask_edge_bonded = edge_attr == 1 # / bonded
        mask_edge_inter = edge_attr == 2 # / intermolecular
        edge_inter = edge_index[:,mask_edge_inter]


        mask_ligand = mol_mask == 0
        mask_protein = mol_mask == 1

        label_ligand = node_idx[mask_ligand]
        label_protein = node_idx[mask_protein]
        label_inter = edge_inter.unique()

        edge_index_ligand, edge_attr_ligand = subgraph(label_ligand, 
                                                       edge_index, 
                                                       edge_attr, 
                                                       relabel_nodes=True,
                                                       num_nodes=x.size(0))

        
        edge_index_protein, edge_attr_protein = subgraph(label_protein, 
                                                         edge_index, 
                                                         edge_attr, 
                                                         relabel_nodes=True,
                                                         num_nodes=x.size(0))

        x = self.atom_embedd(x)
        x_ligand = x[mask_ligand]
        x_protein = x[mask_protein]

        pos_ligand = pos[mask_ligand]
        pos_protein = pos[mask_protein]
        pos_inter = pos[label_inter]

        edge_index_ligand = edge_index_ligand[:,edge_attr_ligand == 1]
        edge_index_protein = edge_index_protein[:,edge_attr_protein == 1]




        i_l, j_l = edge_index_ligand
        dist_ligand = utils_general.distance(pos_ligand[i_l], pos_protein[j_l])**0.5
        dist_ligand = self.rbf(dist_ligand)
        dist_ligand = self.dist_embedd_ligand(dist_ligand)

        residual_ligand = x_ligand


        for layer in self.gine_ligand:
            x_ligand = layer(x_ligand, edge_index_ligand, dist_ligand)


        x_ligand = residual_ligand + x_ligand


        i_p, j_p = edge_index_protein
        dist_protein = utils_general.distance(pos_protein[i_p], pos_protein[j_p])**0.5
        dist_protein = self.rbf(dist_protein)
        dist_protein = self.dist_embedd_protein(dist_protein)

        residual_protein = x_protein


        for layer in self.gine_protein:
            x_protein = layer(x_protein, edge_index_protein, dist_protein)

        x_protein = residual_protein + x_protein

        x = x.index_copy(0, label_ligand, x_ligand)
        x = x.index_copy(0, label_protein, x_protein)

        x_inter = x[label_inter]

        edge_inter_sub, edge_attr_sub = subgraph(label_inter, 
                                                 edge_index=edge_index, 
                                                 edge_attr=edge_attr, 
                                                 relabel_nodes=True, 
                                                 num_nodes=x.size(0))


        residual_interaction = x_inter

        for layer in self.egnn_interaction:
            x_inter, _ = layer(x_inter, 
                               edge_inter_sub, 
                               pos_inter, 
                               edge_attr_sub, 
                               batch_idx=batch_idx[label_inter])

        x_inter = residual_interaction + x_inter

        x = x.index_copy(0, label_inter, x_inter)

        ie, je = edge_inter
        hi, hj = x[ie], x[je]

        dist_inter = utils_general.distance(pos[ie], pos[je])**0.5
        weight = utils_general.cosine_cutoff(dist_inter, self.cutoff)
        dist_inter = self.rbf(dist_inter)
        dist_inter = self.dist_embedd_inter(dist_inter)

        batch_edge = batch_idx[ie]

        h_ijd = torch.cat([hi, hj, dist_inter], dim=-1) #, dist[mask_edge_inter]]
        out = self.affinity_head(h_ijd)
        out = out * weight

        out = self.pool(out, batch_edge)

        return out