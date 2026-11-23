import torch
from torch.nn import (
    Module, 
    Embedding, 
    ModuleList, 
    Linear, 
    Sequential, 
    ReLU,
    Dropout
)

from torch_geometric.utils import subgraph

from models.nn import utils as utils_general, mpnn
from models.built_in_models.egnn_interaction import utils
from models.nn.mpnn import MPNNModule

from typing import Literal



class ComponentModule(Module):

    """Base module for processing molecular or interaction graph components.

    Initializes a configurable message-passing neural network and, for
    non-EGNN architectures, radial basis function encoding for geometric
    edge features.
    """

    def __init__(
            self,
            mpnn_type: Literal['gin', 'gcn', 'gat', 'egnn'],
            input_dim: int,
            hidden_dim: int,
            n_layers: int,
            n_rbf: int,
            cutoff: float,
            **kwargs
    ):

        """Initialize the component processing module.

        Args:
            mpnn_type: Type of message-passing network. Supported options
                are 'gin', 'gcn', 'gat', and 'egnn'.
            input_dim: Dimensionality of input node features.
            hidden_dim: Dimensionality of hidden node representations.
            n_layers: Number of message-passing layers.
            n_rbf: Number of radial basis functions for distance encoding.
            cutoff: Maximum distance used for radial basis function centers.
            **kwargs: Additional configuration arguments passed to MPNNModule.
        """
        
        super().__init__()

        self.mpnn_type = mpnn_type

        self.mpnn = MPNNModule(
            mpnn_type=mpnn_type,
            node_input_dim=input_dim,
            node_hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_rbf=n_rbf,
            cutoff=cutoff,
            **kwargs
        )

        if mpnn_type != 'egnn':
            self.rbf = utils_general.RBF(n_rbf, cutoff=cutoff)
            self.dist_embedd = utils_general.mlp(
                n_rbf, 
                hidden_dim, 
                hidden_dim
        )


class MolModule(ComponentModule):

    """Message-passing module for intramolecular graph processing.

    Extracts a selected molecular component from a combined graph,
    constructs its subgraph, and applies message passing using
    molecular node features and distance-based edge representations.
    """

    def __init__(
            self, 
            mpnn_type, 
            input_dim, 
            hidden_dim, 
            n_layers, 
            n_rbf, 
            cutoff,
            **kwargs
    ):

        """Initialize the molecular processing module.

        Args:
            mpnn_type: Type of message-passing network.
            input_dim: Dimensionality of input node features.
            hidden_dim: Dimensionality of hidden node representations.
            n_layers: Number of message-passing layers.
            n_rbf: Number of radial basis functions for distance encoding.
            cutoff: Maximum distance used for geometric feature encoding.
            **kwargs: Additional configuration arguments passed to
                ComponentModule.
        """

        super().__init__(mpnn_type, input_dim, hidden_dim, n_layers, n_rbf, cutoff, **kwargs)

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            mol_label: torch.Tensor,
            edge_attr=None,
            batch_idx=None,
            pos=None
    ):

        x_mol = x[mol_label]
        pos_mol = pos[mol_label]

        edge_index_mol, _ = subgraph(
                    mol_label, 
                    edge_index, 
                    relabel_nodes=True,
                    num_nodes=x.size(0)
                )

        if self.mpnn_type != 'egnn':
            i, j = edge_index_mol
            dist = utils_general.distance(pos[i], pos[j])**0.5
            edge_attr_mol = self.rbf(dist)
            edge_attr_mol = self.dist_embedd(edge_attr_mol)


        x_mol = self.mpnn(
            x_mol,
            edge_index_mol,
            edge_attr_mol,
            batch_idx[mol_label],
            pos_mol
        )

        x.index_copy_(0, mol_label, x_mol)

        return x



class InterModule(ComponentModule):

    """Message-passing module for intermolecular graph interactions.

    Applies graph convolution to interaction edges connecting different
    molecular components, using geometric distance embeddings for
    non-EGNN architectures.
    """

    def __init__(
            self, 
            mpnn_type, 
            input_dim, 
            hidden_dim, 
            n_layers, 
            n_rbf, 
            cutoff,
            #direction: Literal['lig<prot', 'lig>prot', 'lig<>prot'],
            **kwargs
    ):

        """Initialize the intermolecular interaction module.

        Args:
            mpnn_type: Type of message-passing network.
            input_dim: Dimensionality of input node features.
            hidden_dim: Dimensionality of hidden node representations.
            n_layers: Number of message-passing layers.
            n_rbf: Number of radial basis functions for distance encoding.
            cutoff: Maximum distance used for geometric feature encoding.
            **kwargs: Additional configuration arguments passed to
                ComponentModule.
        """
        
        super().__init__(mpnn_type, input_dim, hidden_dim, n_layers, n_rbf, cutoff, **kwargs)

    def forward(
            self,
            x: torch.Tensor,
            edge_index_interaction: torch.Tensor,
            edge_attr=None,
            batch_idx=None,
            pos=None
    ):

        if self.mpnn_type != 'egnn':
            i, j = edge_index_interaction
            dist = utils_general.distance(pos[i], pos[j])**0.5
            edge_attr = self.rbf(dist)
            edge_attr = self.dist_embedd(dist)


        x = self.mpnn(
            x,
            edge_index_interaction,
            edge_attr,
            batch_idx,
            pos
        )

        return x

        

class InteractionModel(Module):

    """Graph neural network for protein-ligand binding affinity prediction.

    Processes ligand and protein molecular graphs independently using
    intramolecular message passing, followed by intermolecular message
    passing across protein-ligand interaction edges.

    Binding affinity is predicted using either interaction-edge-based
    pooling or separate ligand and protein graph pooling.
    """

    def __init__(
        self,
        n_rbf=32,
        hidden_dim=64,
        output_dim=1,
        ligand_mpnn: Literal['gin', 'gcn', 'gat', 'egnn']='gin',
        ligand_mpnn_n_layers: int=2,
        protein_mpnn: Literal['gin', 'gcn', 'gat', 'egnn']='gin',
        protein_mpnn_n_layers: int=2,
        interaction_mpnn: Literal['gin', 'gcn', 'gat', 'egnn']='egnn',
        interaction_mpnn_n_layers: int=2,
        ligand_cutoff: float=3.0,
        protein_cutoff: float=3.0,
        interaction_cutoff: float=5.0,
        sum_over_interaction: bool=True,
        **kwargs
    ):

        """Initialize the protein-ligand interaction model.

        Args:
            n_rbf: Number of radial basis functions for distance encoding.
            hidden_dim: Dimensionality of node and hidden representations.
            output_dim: Number of predicted output values per complex.
            ligand_mpnn: Message-passing architecture for ligand processing.
            ligand_mpnn_n_layers: Number of ligand message-passing layers.
            protein_mpnn: Message-passing architecture for protein processing.
            protein_mpnn_n_layers: Number of protein message-passing layers.
            interaction_mpnn: Message-passing architecture for intermolecular
                interaction processing.
            interaction_mpnn_n_layers: Number of interaction message-passing
                layers.
            ligand_cutoff: Distance cutoff for ligand geometric encoding.
            protein_cutoff: Distance cutoff for protein geometric encoding.
            interaction_cutoff: Distance cutoff for intermolecular geometric
                encoding.
            sum_over_interaction: If True, predict affinity from interaction
                edges using sum and mean pooling. Otherwise, pool ligand and
                protein representations separately.
            **kwargs: Additional configuration arguments passed to the
                component modules.

        """

        super().__init__()

        self.atom_embedd = utils.AtomEmbeddingLayer(11, hidden_dim) # 101

        self.ligand_module = MolModule(
            mpnn_type=ligand_mpnn,
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            n_layers=ligand_mpnn_n_layers,
            n_rbf=n_rbf,
            cutoff=ligand_cutoff,
            **kwargs
        )

        self.protein_module = MolModule(
            mpnn_type=protein_mpnn,
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            n_layers=protein_mpnn_n_layers,
            n_rbf=n_rbf,
            cutoff=protein_cutoff,
            **kwargs
        )       

        self.interaction_module = InterModule(
            mpnn_type=interaction_mpnn,
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            n_layers=interaction_mpnn_n_layers,
            n_rbf=n_rbf,
            cutoff=interaction_cutoff,
            **kwargs
        )

        num_aff_head_input = 3 if sum_over_interaction else 2


        self.affinity_head = Sequential(
            Linear(hidden_dim*num_aff_head_input, hidden_dim*3),
            ReLU(),
            Linear(hidden_dim*3, hidden_dim*2),
            ReLU(),
            Linear(hidden_dim*2, hidden_dim*2),
            ReLU(),
            Linear(hidden_dim*2, output_dim)
        )


        self.pool_sum = utils_general.make_pooling('sum')
        self.pool_mean = utils_general.make_pooling('mean')
        self.sum_over_interaction = sum_over_interaction
        self.hidden_dim = hidden_dim

        if sum_over_interaction:
            self.distance_embedd = utils_general.mlp(n_rbf, hidden_dim, hidden_dim)
            self.rbf = utils_general.RBF(n_rbf, interaction_cutoff)

            self.inter_cutoff = interaction_cutoff

    def forward(self, batch, perturb=None):

        x = batch.x
        mol_mask = batch.mol_id.squeeze(-1)
        edge_index = batch.edge_index
        edge_attr = batch.edge_type.squeeze(-1)
        batch_idx = batch.batch
        pos = batch.pos

        node_idx = torch.arange(
            0, 
            x.size(0), 
            dtype=torch.long, 
            device=x.device
        )

        mask_edge_inter = edge_attr == 2 # / intermolecular
        edge_inter = edge_index[:,mask_edge_inter]

        mask_ligand = mol_mask == 0
        mask_protein = mol_mask == 1

        label_ligand = node_idx[mask_ligand]
        label_protein = node_idx[mask_protein]

        x = self.atom_embedd(x)

        if perturb is not None:
            x = x + perturb


        x = self.ligand_module(
            x,
            edge_index,
            label_ligand,
            edge_attr,
            batch_idx,
            pos
        )

        x = self.protein_module(
            x,
            edge_index,
            label_protein,
            edge_attr,
            batch_idx,
            pos
        )

        x = self.interaction_module(
            x,
            edge_index,
            edge_attr,
            batch_idx,
            pos
        )


        if self.sum_over_interaction:

            i, j = edge_inter
            batch_idx_inter_edge = batch_idx[i]

            d = utils_general.distance(pos[i], pos[j])

            cosine_weight = utils_general.cosine_cutoff(
                d, 
                self.inter_cutoff
            )

            d = self.rbf(d)
            d = self.distance_embedd(d)



            h_i, h_j = x[i], x[j]
            h_ij = torch.cat(
                [
                    h_i,
                    h_j,
                    d
                ],
                dim=-1
            )

            s = self.affinity_head(h_ij)
            s = s * cosine_weight

            out = self.pool_sum(s, batch_idx_inter_edge)

            return out

        x_ligand = x[mask_ligand]
        x_protein = x[mask_protein]

        batch_idx_ligand = batch_idx[mask_ligand]
        batch_idx_protein = batch_idx[mask_protein]

        x_ligand = self.pool(x_ligand, batch_idx_ligand)
        x_protein = self.pool(x_protein, batch_idx_protein)

        x = torch.cat(
            [
                x_ligand,
                x_protein
            ],
            dim=-1
        )

        out = self.affinity_head(x)

        return out