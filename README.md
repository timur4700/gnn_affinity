# GNN Affinity

A Python framework for building molecular graph neural network (GNN) models to predict protein–ligand binding affinity from structural data. It provides tools for molecular preprocessing, graph construction, and model development.

# Installation Guide

## 1. Prerequisites

The **GNN Affinity** project uses a Conda-based environment for dependency management. Before proceeding with the installation, ensure that either [Anaconda](https://www.anaconda.com/docs/getting-started/installation) or [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) is installed on your machine.

Refer to the official [Anaconda Installation Guide](https://www.anaconda.com/docs/getting-started/installation) for installation instructions.

## 2. Project Installation

### 2.1. Clone the repository

Clone the project repository into your preferred working directory:

```bash
git clone https://github.com/timur4700/gnn_affinity.git
```

Navigate to the project directory:

```bash
cd gnn_affinity
```

### 2.2. Install the project and dependencies

Run the installation script:

```bash
python install.py
```

The installation script automatically creates a Conda environment named `gnn-aff` and installs the project together with its dependencies.

### 2.3. Select the PyTorch installation type

The installation script supports both CPU-based and CUDA-based installations of PyTorch and PyTorch Geometric.

Choose the appropriate installation option using the `-d` flag.

**CPU installation:**

```bash
python install.py -d cpu
```

**CUDA installation:**

```bash
python install.py -d cuda
```

> **Note:** Select the CUDA option if you intend to use a compatible NVIDIA GPU for model training and inference. Ensure that your system has the required NVIDIA GPU drivers and a compatible CUDA environment.

## 3. Environment Configuration

By default, the installation script creates a Conda environment named `gnn-aff`.

The environment name can be customized by modifying the `name` parameter in the `environment.yaml` configuration file:

```yaml
name: gnn-aff
```

The environment is configured to use **Python 3.11**.

## 4. Installed Dependencies

The installation script installs the `gnn_affinity` project and the following dependencies:

| Package | Version |
|---|---|
| Python | 3.11 |
| PyTorch | 2.12.1 |
| TorchVision | 0.27.1 |
| PyTorch Geometric | — |
| pyg_lib | — |
| torch_scatter | — |
| torch_sparse | — |
| scikit-learn | — |
| pandas | — |
| matplotlib | — |
| MDAnalysis | — |
| NumPy | 2.4.6 |
| h5py | 3.16.0 |
| RDKit | 2025.9.3 |
| PyYAML | — |
| Pydantic | — |

*Packages without an explicitly specified version are installed according to the dependency configuration.*

## 5. Activate the Environment

After installation, activate the Conda environment:

```bash
conda activate gnn-aff
```

The `gnn_affinity` package is installed directly into the environment and is ready to be imported and used.



# Usage Example: PDBbind v.2020 Dataset

> **Note:** Automated dataset preparation is currently supported only for the PDBbind v.2020 dataset. Support for additional datasets and workflows for preparing custom datasets is planned for future development.

## 1. Dataset Download

Download the following two dataset archives from the **PDBbind v.2020** section of the [PDBbind Download Portal](https://www.pdbbind-plus.org.cn/download):

1. **Protein-ligand complexes: The general set minus refined set**
2. **Protein-ligand complexes: The refined set**

> **Note:** Registration on the PDBbind website is required before downloading the datasets.

The downloaded archives can be stored in any directory of your choice. You will specify the directory path during dataset preparation.

## 2. Dataset Preparation

Before constructing molecular graphs, the downloaded dataset must be prepared using the built-in dataset preparation command.

Run the following command:

```bash
gnn_affinity dataset prepare -d pdb pdbbind -i path/to/downloaded/data/
```

Where:

- `-d pdb pdbbind` specifies the dataset format and dataset identifier.
- `-i` specifies the directory containing the downloaded PDBbind archives.

The preparation workflow automatically:

1. Extracts the downloaded dataset archives.
2. Merges the dataset entries into a unified directory named `pdbbind_2020_entries`.
3. Generates a dataset index file named `pdbbind_data.csv`.

The index file contains metadata for each protein-ligand complex, including:

| Column | Description |
|---|---|
| PDB ID | Four-character PDB identifier |
| Resolution (Å) | Resolution of the protein structure |
| Year | Year associated with the entry |
| Dataset Type | PDBbind dataset classification |
| -logKd/Ki | Binding affinity value |
| Ligand Name | Name of the ligand |
| UNIPROT ID | UniProt accession identifier |
| Protein Name | Name of the target protein |

The prepared dataset, including the unified entries directory and index file, is saved in the same directory as the downloaded archives.


## 3. Generating Molecular and Graph Configuration Files

GNN Affinity provides a built-in CLI command for generating molecular and graph configuration files. These files define the preprocessing and graph construction parameters used by the pipeline.

To generate the configuration files, run:

```bash
gnn_affinity graph config -m egnn_interaction -i path/to/downloaded/data/
```

During execution, the CLI will prompt you to select the graph representation type:

```text
Choose Graph Type:
0) general_graph
```

Select `0` (`general_graph`) to proceed.

Upon successful execution, two configuration files will be generated in the specified dataset directory:

- `mol_config.yaml` — defines molecular preprocessing and structure preparation parameters.
- `graph_config.yaml` — defines graph construction parameters, including edge generation and molecular interactions.

Both configuration files can be customized before proceeding with graph construction.

### 3.1. Molecular Configuration

The `mol_config.yaml` file controls ligand and protein preprocessing, binding pocket extraction, structure sanitization, and input molecular structure formats.

**Example configuration (`mol_config.yaml`):**

```yaml
MolConfig:
  Ligand:
    sanitize: false  # Options: [true, false]
                     # Enables or disables ligand sanitization using RDKit.

  Protein:
    extract_pocket: true  # Options: [true, false]
                          # If false, the full protein structure is used,
                          # and extract_method and pocket_cutoff are ignored.

    extract_method: atom  # Options: [atom, cog, com]
                          # Defines the reference for pocket extraction.

    pocket_cutoff: 10.0   # Maximum distance (Å) used for pocket extraction.

    sanitize: false       # Options: [true, false]
                          # Enables or disables protein sanitization.

  Formats:
    ligand: mol2          # Options: [mol2, sdf]
                          # Input format of ligand structures.
                          # mol2 is recommended.

  ProteinSource: pocket   # Options: [protein, pocket]
```

#### Configuration parameters

| Parameter | Description |
|---|---|
| `Ligand.sanitize` | Enables or disables ligand sanitization using RDKit. |
| `Protein.extract_pocket` | Controls whether the protein binding pocket is extracted during preprocessing. |
| `Protein.extract_method` | Defines the distance reference used for pocket identification. |
| `Protein.pocket_cutoff` | Specifies the maximum distance threshold (Å) used for pocket extraction. |
| `Protein.sanitize` | Enables or disables protein sanitization. |
| `Formats.ligand` | Specifies the input format of ligand structures (`mol2` or `sdf`). The `mol2` format is recommended. |
| `ProteinSource` | Selects the protein structure source: the full protein or the predefined binding pocket supplied by PDBbind. |

#### Pocket extraction methods

The `extract_method` parameter determines how the protein binding pocket is identified relative to the ligand:

- `atom`: Uses the distances between protein atoms and ligand atoms.
- `cog`: Uses the ligand center of geometry as the distance reference.
- `com`: Uses the ligand center of mass as the distance reference.

The `pocket_cutoff` parameter specifies the distance threshold (in Å) used by the selected extraction method.

#### Protein structure sources

PDBbind v.2020 provides two types of protein structure files:

- `protein`: The complete protein structure.
- `pocket`: The predefined binding pocket provided by the dataset authors.

The `ProteinSource` parameter determines which structure is used during graph construction.

---

### 3.2. Graph Configuration

The `graph_config.yaml` file defines the graph representation and connectivity rules used to construct molecular graphs.

It controls the topology of ligand and protein graphs, including edge directionality, self-loops, graph dimensionality, and distance-based connectivity. It also defines how ligand-protein interaction edges are generated.

**Example configuration (`graph_config.yaml`):**

```yaml
GraphConfig:
  Graph:
    Ligand:
      undirected: true              # Options: [true, false]
      self_loop: false              # Options: [true, false]
      graph_type: 2d                # Graph dimensionality: 2d or 3d
      intra_cutoff: 5.0             # Intramolecular distance cutoff (Å).
                                    # Applicable to 3D graphs.

    Protein:
      undirected: true              # Options: [true, false]
      self_loop: false              # Options: [true, false]
      graph_type: 2d                # Graph dimensionality: 2d or 3d
      intra_cutoff: 5.0             # Intramolecular distance cutoff (Å).
                                    # Applicable to 3D graphs.

  Interaction:
    add_interaction_edges: true     # Options: [true, false]
    edge_type: true                 # Options: [true, false]
    inter_cutoff: 5.0               # Intermolecular distance cutoff (Å).
```

#### Configuration parameters

**Ligand and protein graph parameters**

The following parameters are defined independently for ligand and protein graphs:

| Parameter | Description |
|---|---|
| `undirected` | Controls whether graph edges are treated as undirected. |
| `self_loop` | Controls whether self-loop edges are added to the graph. |
| `graph_type` | Defines the graph representation dimensionality (`2d` or `3d`). |
| `intra_cutoff` | Specifies the intramolecular distance cutoff (Å) used for connectivity in 3D graphs. |

**Ligand-protein interaction parameters**

The `Interaction` section controls the generation of edges connecting ligand and protein nodes.

| Parameter | Description |
|---|---|
| `add_interaction_edges` | Enables or disables the generation of ligand-protein interaction edges. |
| `edge_type` | Controls whether edge-type information is included in the graph representation. |
| `inter_cutoff` | Specifies the intermolecular distance cutoff (Å) used to establish ligand-protein connectivity. |

The `inter_cutoff` parameter defines the maximum distance between ligand and protein atoms for generating interaction edges.

> **Note:** Intramolecular connectivity (`intra_cutoff`) is currently applicable only to 3D graphs. It is not implemented for 2D graph representations.



