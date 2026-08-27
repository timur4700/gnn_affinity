import torch
from torch.nn import Module, Embedding


class AtomEmbeddingLayer(Module):

    def __init__(self, n_atoms, hidden_dim):
        super().__init__()


        n_arom_class = 2
        n_hydrogens = 5
        self.atom_embedd = Embedding(n_atoms, int(hidden_dim/2))
        self.arom_embedd = Embedding(n_arom_class, int(hidden_dim/4))
        self.hydrogen_embedd = Embedding(n_hydrogens, int(hidden_dim/4))

    def forward(self, x):

        x_atom = self.atom_embedd(x[:,0].long())
        x_atom = x_atom.squeeze(1)

        x_arom = self.arom_embedd(x[:,1].long())
        x_arom = x_arom.squeeze(1)

        x_hydro = self.hydrogen_embedd(x[:,2].long())
        x_hydro = x_hydro.squeeze(1)

        x = torch.cat([x_atom, x_arom, x_hydro], dim=-1)

        return x