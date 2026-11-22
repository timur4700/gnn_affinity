# GNN Affinity

A Python framework for developing molecular graph neural network (GNN) models to predict protein–ligand binding affinity from structural data. It provides tools for molecular preprocessing, graph construction, and model development.


# Project Features

**GNN Affinity** is a Python framework designed to simplify the development and application of graph neural networks (GNNs) for molecular property prediction, with a primary focus on protein–ligand binding affinity.

The framework provides tools for molecular preprocessing, graph construction, dataset preparation, and model training. Its modular architecture allows users to configure different message-passing neural networks and experiment with alternative representations of molecular structures and protein–ligand interactions.

The project is under active development, with plans to expand the range of supported model architectures, prediction tasks, datasets, and integrations with external cheminformatics and molecular modelling software, such as Schrödinger.

## Available Datasets

The following dataset currently supports automated preparation:

- **PDBbind v.2020** – Protein–ligand complexes with experimentally determined binding affinity data.

Additional datasets and support for custom datasets are planned.

## Built-in Models

### InteractionModel

`InteractionModel` is a modular graph neural network designed to predict protein–ligand binding affinity by learning representations of the ligand, protein, and their intermolecular interactions.

The model processes a protein–ligand complex through three main components:

1. **Ligand representation:** A message-passing neural network learns molecular representations from the ligand's atomic features and intramolecular graph structure.
2. **Protein representation:** A separate message-passing network processes the protein structure, learning representations of its atoms and local structural environment.
3. **Interaction representation:** A third network processes the combined protein–ligand graph, capturing information about intermolecular contacts and their spatial relationships.

Each component can use a configurable message-passing architecture: **[GIN](https://arxiv.org/abs/1810.00826), [GCN](https://arxiv.org/abs/1609.02907), [GAT](https://arxiv.org/abs/1710.10903), or [EGNN](https://arxiv.org/abs/2102.09844)**. The number of message-passing layers and distance cutoffs can also be configured independently.

After message passing, the model predicts binding affinity using one of two aggregation strategies:

- **Interaction-based aggregation:** The model combines the representations of atoms connected by intermolecular edges, predicts a contribution for each interaction, and aggregates these contributions to obtain the final affinity prediction.
- **Separate molecular pooling:** The model independently aggregates ligand and protein atom representations, concatenates the resulting molecular representations, and passes them through a prediction head.

The first strategy is enabled by default through `sum_over_interaction=True`.

The model uses radial basis functions (RBFs) to encode interatomic distances, allowing spatial information to be incorporated into message passing.


## Available Graph Representations

The framework supports two types of molecular graphs:

- **2D graphs:** Represent molecular connectivity based on chemical topology, such as covalent bonds.
- **3D graphs:** Represent molecular structures using spatial proximity. Edges are constructed based on distances between atoms and a user-defined cutoff.

These representations can be used to capture complementary information about molecular connectivity and three-dimensional structure.

## Available Node (Atom) Features

The current implementation supports the following atomic features:

- `atomic_num` – Atomic number of the element.
- `aromatic_status` – Indicates whether an atom belongs to an aromatic system.
- `num_hydrogens` – Number of hydrogen atoms associated with the atom.

## Available Edge (Bond and Interaction) Features

- **Distance encoding:** Interatomic distances are represented using radial basis functions (RBFs).

RBF encoding transforms continuous distances into a higher-dimensional representation that can be processed by the graph neural network.


# Installation Guide

## 1. Prerequisites

GNN Affinity requires **Python 3.11** and uses Conda for environment and dependency management.

### 1.1. Software requirements

Before installing the project, ensure that either [Anaconda](https://www.anaconda.com/docs/getting-started/installation) or [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) is installed.

These distributions provide Conda, which is used to create and manage the project's environment.

Refer to the official [Anaconda Installation Guide](https://www.anaconda.com/docs/getting-started/installation) or [Miniconda Installation Guide](https://docs.conda.io/projects/miniconda/en/latest/) for installation instructions.

> **Note:** Python 3.11 does not need to be installed separately. The installation script creates a Conda environment with the required Python version.

### 1.2. Operating system compatibility

The project has been tested on the following operating systems:

- macOS 15.7.3
- Ubuntu 26.04

Other operating systems and versions have not yet been formally tested.

### 1.3. Hardware requirements

The hardware requirements depend on the selected PyTorch installation:

- **CPU:** Supported for model training and inference.
- **NVIDIA GPU:** CUDA acceleration is supported on compatible NVIDIA hardware with the required drivers.

GPU acceleration is recommended for computationally intensive training tasks and larger datasets.

## 2. Installation

### 2.1. Clone the repository

Clone the repository:

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

The script automatically creates a Conda environment named `gnn-aff` and installs the project and its dependencies.

### 2.3. Select the PyTorch installation type

The installation script supports CPU- and CUDA-based installations of PyTorch and PyTorch Geometric.

Select the appropriate installation using the `-d` argument.

**CPU installation:**

```bash
python install.py -d cpu
```

**CUDA installation:**

```bash
python install.py -d cuda
```

> **Note:** Select the CUDA option if you intend to use a compatible NVIDIA GPU for model training or inference. Ensure that the required NVIDIA drivers are installed.

## 3. Environment configuration

By default, the installation script creates a Conda environment named `gnn-aff`.

You can customize the environment name by modifying the `name` parameter in `environment.yaml`:

```yaml
name: gnn-aff
```

The environment is configured to use Python 3.11.

## 4. Installed dependencies

The installation script installs the `gnn_affinity` package and its dependencies.

| Package | Version |
|---|---|
| Python | 3.11 |
| PyTorch | 2.12.1 |
| TorchVision | 0.27.1 |
| PyTorch Geometric | — |
| `pyg_lib` | — |
| `torch_scatter` | — |
| `torch_sparse` | — |
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

## 5. Activate the environment

After installation, activate the Conda environment:

```bash
conda activate gnn-aff
```

The `gnn_affinity` package is installed in the environment and is ready to be imported and used.

---

# Usage Guide: PDBbind v.2020

> **Note:** Automated dataset preparation currently supports only PDBbind v.2020. Support for additional datasets and workflows for preparing custom datasets is planned for future development.

## 1. Dataset download

Download the following two archives from the **PDBbind v.2020** section of the [PDBbind Download Portal](https://www.pdbbind-plus.org.cn/download):

1. **Protein-ligand complexes: The general set minus refined set**
2. **Protein-ligand complexes: The refined set**

> **Note:** Registration on the PDBbind website is required before downloading the datasets.

The downloaded archives can be stored in any directory. You will specify this directory when preparing the dataset.

## 2. Dataset preparation

Before constructing molecular graphs, prepare the downloaded dataset using the built-in dataset preparation command:

```bash
gnn_affinity dataset prepare -d pdb pdbbind -i path/to/downloaded/data/
```

**Command-line arguments:**

- `-d pdb pdbbind`: Specifies the dataset format and dataset identifier.
- `-i`: Specifies the directory containing the downloaded PDBbind archives.

The preparation workflow automatically:

1. Extracts the downloaded dataset archives.
2. Combines the dataset entries into a unified directory named `pdbbind_2020_entries`.
3. Generates a dataset index file named `pdbbind_data.csv`.

The index file contains metadata for each protein–ligand complex:

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

**Example directory structure:**

```text
downloaded_pdbbind_directory/
├── pdbbind_2020_entries/
├── metadata.json
├── pdbbind_data.csv
├── PDBbind_v2020_other_PL.tar.gz
└── PDBbind_v2020_refined.tar.gz
```

## 3. Molecular and graph configuration

GNN Affinity provides a built-in CLI command for generating molecular and graph configuration files. These files define the preprocessing and graph construction parameters used by the pipeline.

Generate the configuration files using:

```bash
gnn_affinity graph config -m egnn_interaction -i path/to/downloaded/data/
```

**Command-line arguments:**

- `-m`: Specifies the model architecture for which the configuration files will be generated. In this example, `egnn_interaction` is selected.
- `-i`: Specifies the input directory containing the downloaded and prepared dataset.

To view the available model architectures, run:

```bash
gnn_affinity model list
```

During execution, the CLI prompts you to select the graph representation type:

```text
Choose Graph Type:
0) general_graph
```

Select `0` (`general_graph`) to proceed.

Upon successful execution, two configuration files are generated in the specified dataset directory:

- `mol_config.yaml`: Defines molecular preprocessing and structure preparation parameters.
- `graph_config.yaml`: Defines graph construction parameters, including edge generation and molecular interactions.

Both files can be customized before proceeding with graph construction.

### 3.1. Molecular configuration

The `mol_config.yaml` file controls ligand and protein preprocessing, binding pocket extraction, structure sanitization, and ligand input format.

**Example configuration:**

```yaml
MolConfig:
  Ligand:
    sanitize: false  # Options: [true, false]

  Protein:
    extract_pocket: true  # Options: [true, false]
    extract_method: atom  # Options: [atom, cog, com]
    pocket_cutoff: 10.0
    sanitize: false  # Options: [true, false]

  Formats:
    ligand: mol2  # Options: [mol2, sdf]

  ProteinSource: pocket  # Options: [protein, pocket]
```

#### Configuration parameters

| Parameter | Description |
|---|---|
| `Ligand.sanitize` | Enables or disables ligand sanitization using RDKit. |
| `Protein.extract_pocket` | Controls whether the protein binding pocket is extracted during preprocessing. |
| `Protein.extract_method` | Defines the distance reference used for pocket identification. |
| `Protein.pocket_cutoff` | Specifies the maximum distance threshold (Å) used for pocket extraction. |
| `Protein.sanitize` | Enables or disables protein sanitization. |
| `Formats.ligand` | Specifies the ligand input format (`mol2` or `sdf`). The `mol2` format is recommended. |
| `ProteinSource` | Selects the protein structure source: the full protein or the predefined binding pocket supplied by PDBbind. |

#### Pocket extraction methods

The `extract_method` parameter determines how the protein binding pocket is identified relative to the ligand:

- `atom`: Uses distances between protein atoms and ligand atoms.
- `cog`: Uses the ligand center of geometry as the distance reference.
- `com`: Uses the ligand center of mass as the distance reference.

The `pocket_cutoff` parameter specifies the distance threshold, in Å, used by the selected extraction method.

#### Protein structure sources

PDBbind v.2020 provides two types of protein structure files:

- `protein`: The complete protein structure.
- `pocket`: The predefined binding pocket provided by the dataset authors.

The `ProteinSource` parameter determines which structure is used during graph construction.

### 3.2. Graph configuration

The `graph_config.yaml` file defines the graph representation and connectivity rules used to construct molecular graphs.

It controls the topology of ligand and protein graphs, including edge directionality, self-loops, graph dimensionality, and distance-based connectivity. It also defines how ligand–protein interaction edges are generated.

**Example configuration:**

```yaml
GraphConfig:
  Graph:
    Ligand:
      undirected: true
      self_loop: false
      graph_type: 2d
      intra_cutoff: 5.0

    Protein:
      undirected: true
      self_loop: false
      graph_type: 2d
      intra_cutoff: 5.0

  Interaction:
    add_interaction_edges: true
    edge_type: true
    inter_cutoff: 5.0
```

#### Ligand and protein graph parameters

The following parameters are defined independently for ligand and protein graphs:

| Parameter | Description |
|---|---|
| `undirected` | Controls whether graph edges are treated as undirected. |
| `self_loop` | Controls whether self-loop edges are added to the graph. |
| `graph_type` | Defines the graph representation dimensionality (`2d` or `3d`). |
| `intra_cutoff` | Specifies the intramolecular distance cutoff (Å) used for connectivity in 3D graphs. |

#### Ligand–protein interaction parameters

The `Interaction` section controls the generation of edges connecting ligand and protein nodes.

| Parameter | Description |
|---|---|
| `add_interaction_edges` | Enables or disables the generation of ligand–protein interaction edges. |
| `edge_type` | Controls whether edge-type information is included in the graph representation. |
| `inter_cutoff` | Specifies the intermolecular distance cutoff (Å) used to establish ligand–protein connectivity. |

The `inter_cutoff` parameter defines the maximum distance between ligand and protein atoms for generating interaction edges.

> **Note:** Intramolecular connectivity (`intra_cutoff`) is currently applicable only to 3D graphs. It is not implemented for 2D graph representations.

## 4. Graph dataset construction

After generating and configuring the molecular and graph configuration files, you can proceed with molecular graph construction.

Run the following command:

```bash
gnn_affinity graph prepare \
    -i path/to/downloaded/data/ \
    -o path/to/destination/directory/
```

**Command-line arguments:**

- `-i`: Specifies the input directory containing the prepared dataset and configuration files.
- `-o`: Specifies the output directory where the generated graph dataset will be saved.

The output directory can be selected freely. However, using an empty directory is recommended to keep the generated dataset separate from existing files and avoid potential conflicts.

### 4.1. Multiprocessing

By default, graph dataset preparation runs using a single CPU process.

The pipeline also supports multiprocessing, allowing graph construction to be distributed across multiple CPU processes.

To enable multiprocessing, specify the `-mp` argument followed by the desired number of processes:

```bash
gnn_affinity graph prepare \
    -i path/to/downloaded/data/ \
    -o path/to/destination/directory/ \
    -mp 8
```

In this example, graph construction is performed using eight CPU processes.

The number of processes can be adjusted according to the available CPU resources. Avoid assigning more processes than your system can efficiently handle.

### 4.2. Output files

After graph preparation, the output directory contains the following files. Each file is assigned a unique identifier (`[id]`) to distinguish individual dataset preparation runs.

| File | Description |
|---|---|
| `error_[id].log` | Log file containing the PDB IDs of structures that failed during preprocessing or graph construction. |
| `pdbbind_graph_dataset_[id].pkl` | Serialized graph dataset containing PyTorch Geometric `Data` objects representing the prepared protein–ligand complexes. |
| `metadata_[id].json` | JSON file containing metadata associated with the generated graph dataset. |

**Example directory structure:**

```text
output_directory/
└── prepared_data_id/
    ├── error_[id].log
    ├── pdbbind_graph_dataset_[id].pkl
    └── metadata_[id].json
```

---

# 5. Model preparation

## 5.1. Building a model

After preparing the graph dataset, the next step is to initialize a model for training.

Create a model directory using:

```bash
gnn_affinity model build -i path/to/prepared/graphs/
```

The `-i` argument specifies the directory containing the prepared graph dataset.

The command automatically creates a model directory within the specified graph dataset directory. Each model is assigned a unique identifier (`model_id`) to distinguish individual model configurations and training runs.

**Example directory structure:**

```text
output_directory/
└── prepared_data_id/
    └── model_id/
        ├── parameters/
        ├── metadata_[id].json
        ├── model_settings_[id].json
        └── train_settings_[id].json
```

The generated files serve the following purposes:

| File or directory | Description |
|---|---|
| `parameters/` | Directory for storing model parameters, including trained model weights and checkpoints. |
| `metadata_[id].json` | Contains metadata associated with the model and its corresponding graph dataset. |
| `model_settings_[id].json` | Stores the model architecture and configuration parameters. |
| `train_settings_[id].json` | Stores the training configuration and hyperparameters. |

## 5.2. Model configuration

The model configuration defines the architecture, including the message-passing neural networks (MPNNs) used to learn ligand, protein, and ligand–protein interaction representations.

**Example configuration (`model_settings.yaml`):**

```yaml
ModelSettings:
  n_rbf: 32
  hidden_dim: 64
  output_dim: 1

  ligand_mpnn: gin
  ligand_mpnn_n_layers: 2

  protein_mpnn: gin
  protein_mpnn_n_layers: 2

  interaction_mpnn: egnn
  interaction_mpnn_n_layers: 4

  ligand_cutoff: 3.0
  protein_cutoff: 3.0
  interaction_cutoff: 5.0

  sum_over_interaction: true
  edge_attr: true
```

#### Configuration parameters

| Parameter | Description |
|---|---|
| `n_rbf` | Number of radial basis functions used for distance encoding. |
| `hidden_dim` | Dimensionality of the hidden representations. |
| `output_dim` | Number of output dimensions. For affinity prediction, this is typically 1. |
| `ligand_mpnn` | Message-passing architecture used for ligand representation learning. |
| `ligand_mpnn_n_layers` | Number of message-passing layers in the ligand network. |
| `protein_mpnn` | Message-passing architecture used for protein representation learning. |
| `protein_mpnn_n_layers` | Number of message-passing layers in the protein network. |
| `interaction_mpnn` | Message-passing architecture used for ligand–protein interaction learning. |
| `interaction_mpnn_n_layers` | Number of message-passing layers in the interaction network. |
| `ligand_cutoff` | Distance cutoff used for ligand graph processing. |
| `protein_cutoff` | Distance cutoff used for protein graph processing. |
| `interaction_cutoff` | Distance cutoff used for ligand–protein interaction processing. |
| `sum_over_interaction` | Controls whether interaction representations are aggregated using summation. |
| `edge_attr` | Controls whether edge attributes are incorporated into message passing. |

The supported MPNN architectures are:

- `gin`
- `gcn`
- `gat`
- `egnn`

## 5.3. Training configuration

The training configuration defines the optimization strategy, training hyperparameters, computational device, and logging options.

**Example configuration (`train_settings.yaml`):**

```yaml
TrainingSettings:
  optimizer: adam_w
  seed: 42
  n_epochs: 500
  learning_rate: 0.00001
  weight_decay: 0.000001

  early_stop: true
  when_early_stop: 100

  loss_func: mse
  verbose: 1

  device: cuda

  save_train_log: true
  show_test_metrics: true
```

#### Training parameters

| Parameter | Description |
|---|---|
| `optimizer` | Optimization algorithm. Supports Adam (`adam`) and AdamW (`adam_w`). |
| `seed` | Random seed used for reproducibility. |
| `n_epochs` | Maximum number of training epochs. |
| `learning_rate` | Learning rate used by the optimizer. |
| `weight_decay` | Weight decay coefficient used for regularization. |
| `early_stop` | Enables or disables early stopping. |
| `when_early_stop` | Number of epochs without improvement before training is stopped. |
| `loss_func` | Loss function used for optimization. Currently supports mean squared error (`mse`). |
| `verbose` | Controls the verbosity of training output (`0` or `1`). |
| `device` | Specifies the computational device: CPU (`cpu`), NVIDIA GPU (`cuda`), or Apple Silicon GPU (`mps`). |
| `save_train_log` | Controls whether training logs are saved. |
| `show_test_metrics` | Controls whether test-set evaluation metrics are displayed. |

The supported computational devices are:

- `cpu`: CPU execution.
- `cuda`: NVIDIA GPU acceleration.
- `mps`: Apple Silicon GPU acceleration.

> **Note:** Select the computational device according to your available hardware and installed PyTorch version. CUDA requires a compatible NVIDIA GPU, while MPS is intended for Apple Silicon devices.

---

# 6. Model training

## 6.1. Starting model training

Once the model has been initialized and its configuration files prepared, start training using:

```bash
gnn_affinity model train -i path/to/prepared/model/ -m new
```

**Command-line arguments:**

- `-i`: Specifies the path to the prepared model directory.
- `-m`: Specifies the training mode. Two options are available:
  - `new`: Starts a new training run with freshly initialized model parameters.
  - `resume`: Resumes training from a previously saved checkpoint.

### Selecting a training run

When executing the training command, the CLI prompts you to select a model run:

```text
Choose Run for egnn_interaction, ID []:
0) run 0
1) run 1
```

Select the desired run by entering its corresponding ID.

This allows you to select an existing run configuration associated with the chosen model architecture.

## 6.2. Training output and saved files

After training, the generated files are saved in the model's `parameters/` directory.

Each training run is assigned a separate directory (`run_[n]`), allowing multiple runs to be stored independently.

**Example directory structure:**

```text
output_directory/
└── prepared_data_id/
    └── model_id/
        └── parameters/
            └── run_[n]/
                ├── metadata.json
                ├── model_checkpoint.pt
                ├── model_config.yaml
                ├── train_config.yaml
                ├── model_weights.pt
                ├── performance_plot.png
                ├── predictions.csv
                └── train.log
```

### Description of generated files

| File | Description |
|---|---|
| `metadata.json` | Contains metadata associated with the training run. |
| `model_checkpoint.pt` | Saved training checkpoint used to resume training. |
| `model_config.yaml` | Model architecture and configuration parameters used for the current run. |
| `train_config.yaml` | Training configuration and hyperparameters used for the current run. |
| `model_weights.pt` | Saved model parameters corresponding to the best-performing model during training. |
| `performance_plot.png` | Visualization of model training performance. |
| `predictions.csv` | Predictions generated during model evaluation. |
| `train.log` | Training log containing execution information. Generated only when `save_train_log` is enabled in the training configuration. |

## 6.3. Managing training runs

Each time a new training run is initiated using `-m new`, a new `run_[n]` directory is created inside `parameters/`.

This allows multiple independent experiments to be performed using the same model configuration while preserving their respective checkpoints, model weights, predictions, and evaluation results.

When using `-m resume`, training continues from a previously saved checkpoint rather than initializing a new set of model parameters.

