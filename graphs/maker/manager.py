from typing import Any
from graphs.maker.builders import GraphBuilder

from pathlib import Path

from rdkit import Chem
from chem import protein_chem, utils


from configs.mol import MolConfig
from schemas.mol import LigandData, ProteinData

from dataclasses import asdict

from datasets.paths.builders import PathBuilder

class GraphManager():

    """
    Manages molecular complex loading and graph preparation.

    Coordinates file path resolution, molecular structure loading,
    protein pocket extraction, target data retrieval, and graph
    construction for protein-ligand complexes.

    Parameters
    ----------
    path_builder : PathBuilder
        Builder responsible for resolving paths to molecular structure files.
    graph_builder : GraphBuilder
        Builder responsible for converting ligand and protein data into
        graph representations.
    target_data : dict[str, Any]
        Mapping between complex identifiers and their associated target
        data, such as binding affinity values.
    mol_config : MolConfig
        Configuration specifying molecular preprocessing options,
        including protein pocket extraction settings.
    sanitize : bool, default=False
        Whether to sanitize molecules during loading and preprocessing.

    Attributes
    ----------
    path_builder : PathBuilder
        Path builder used to resolve molecular structure files.
    graph_builder : GraphBuilder
        Graph builder used to construct molecular graphs.
    mol_config : MolConfig
        Molecular preprocessing configuration.
    index_data : dict[str, Any]
        Target data indexed by complex identifiers.
    sanitize : bool
        Whether molecular sanitization is enabled.
    complex_dir : pathlib.Path
        Directory containing the current molecular complex.
    mol_paths : Path
        Resolved paths to the molecular structure files.
    complex_name : str
        Identifier of the current complex, derived from its directory name.
    ligand_data : LigandData
        Processed ligand molecular data.
    protein_data : ProteinData
        Processed protein molecular data.
    """

    def __init__(self,
                 path_builder: PathBuilder,
                 graph_builder: GraphBuilder,
                 target_data: dict[str, Any],
                 mol_config: MolConfig,
                 sanitize: bool=False):


        # Defining Path 2 Files
        self.path_builder = path_builder

        # Graph Builder
        self.graph_builder = graph_builder

        self.mol_config = mol_config

        self.index_data = target_data
        self.sanitize = sanitize


    def set_data(self, complex_dir: str | Path):

        """
        Set the current molecular complex and resolve its file paths.

        Parameters
        ----------
        complex_dir : str or pathlib.Path
            Directory containing the molecular complex files.

        Notes
        -----
        The complex identifier is derived from the directory name.
        Resolved molecular file paths are stored in ``self.mol_paths``.
        """

        self.complex_dir = complex_dir if isinstance(complex_dir, Path) else Path(complex_dir)
        self.mol_paths = self.path_builder.build(self.complex_dir)

        # Complex ID, stated by complex directory name
        self.complex_name = self.complex_dir.name
        
    def load_protein(self,
            ligand: Chem.Mol=None,
            extract_pocket: bool=True,
            extract_method: str='atom',
            pocket_cutoff: float=10.0,
            sanitize: bool=False
        ) -> Chem.Mol:

        """
        Load a protein structure and optionally extract its binding pocket.

        Parameters
        ----------
        ligand : rdkit.Chem.Mol, optional
            Ligand molecule used as a reference for protein pocket extraction.
        extract_pocket : bool, default=True
            Whether to extract the protein binding pocket around the ligand.
        extract_method : str, default='atom'
            Pocket extraction method. Determines how protein atoms or
            residues are selected around the ligand.
        pocket_cutoff : float, default=10.0
            Distance cutoff, in angstroms, used for pocket extraction.
        sanitize : bool, default=False
            Whether to sanitize the protein during loading.

        Returns
        -------
        rdkit.Chem.Mol
            Loaded protein structure or extracted binding pocket.

        Notes
        -----
        Pocket extraction uses the sanitization setting stored in
        ``self.sanitize`` rather than the local ``sanitize`` argument.
        """

        protein = Chem.MolFromPDBFile(
            self.mol_paths.protein_path, 
            sanitize=sanitize
        )

        if extract_pocket:
            return protein_chem.pocket_extraction(
                ligand,
                protein,
                cutoff_distance=pocket_cutoff,
                method=extract_method,
                sanitize=self.sanitize
            )

        return protein



    def load_ligand(self) -> Chem.Mol:

        """
        Load the ligand structure from the resolved molecular file.

        Returns
        -------
        rdkit.Chem.Mol
            Loaded ligand molecule. For SDF files, the first molecule
            in the file is returned.

        Notes
        -----
        Molecule loading and sanitization are handled by ``utils.load_mol``
        using the sanitization setting stored in ``self.sanitize``.
        """

        # Loading MOL2 if ligand in .mol2 format
        ligand = utils.load_mol(
            self.mol_paths.ligand_path,
            sanitize=self.sanitize
        )

        if self.mol_paths.ligand_path.suffix == '.sdf':
            return ligand[0]

        return ligand

    

    def extract_graph_data(self, complex_id):

        """
        Retrieve target data associated with a molecular complex.

        Parameters
        ----------
        complex_id : str
            Identifier of the molecular complex.

        Returns
        -------
        dict[str, Any]
            Target data associated with the specified complex.

        Raises
        ------
        KeyError
            If the complex identifier is not present in the target data.
        """

        return self.index_data[complex_id]


    def init_graph_preparation(self):


        """
        Load molecular structures and prepare the protein-ligand graph.

        Loads the ligand, extracts the corresponding protein binding pocket
        according to the molecular configuration, retrieves target data,
        and constructs the graph representation using the graph builder.

        Returns
        -------
        Graph
            Prepared protein-ligand graph with the associated target data
            assigned to its ``data`` attribute.

        Raises
        ------
        KeyError
            If the target data for the current complex does not contain
            the required ``'y'`` target value.

        Notes
        -----
        This method requires ``set_data`` to be called beforehand to
        initialize the current complex directory and molecular file paths.

        The prepared ligand and protein data are stored in ``self.ligand_data``
        and ``self.protein_data``, respectively.
        """

        self.ligand_data = LigandData(mol=self.load_ligand())
        self.protein_data = ProteinData(mol=self.load_protein(
            self.ligand_data.mol,
            **self.mol_config.protein.model_dump()
            )
        )

        data = self.extract_graph_data(self.complex_name)

        if 'y' not in data:
            raise KeyError('The target value was not provided')

        graph = self.graph_builder.prepare_graph(
            self.ligand_data,
            self.protein_data
        )

        graph.data = data

        return graph
