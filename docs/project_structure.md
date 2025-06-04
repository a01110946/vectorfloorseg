# VectorFloorSeg Complete Project Structure

This document outlines the complete project structure for the VectorFloorSeg implementation, organized according to the five development phases.

## Project Root Structure

```
VecFloorSeg/                                    # Phase 1: Root repository directory
├── README.md                                   # Phase 1: Project documentation
├── requirements.txt                            # Phase 1: Python dependencies
├── environment.yml                             # Phase 1: Conda environment file
├── .gitignore                                  # Phase 1: Git ignore rules
├── LICENSE                                     # Phase 1: License file
│
├── vectorfloorseg_env/                         # Phase 1: Virtual environment directory
│
├── pyg_backup/                                 # Phase 1: Backup of original PyG installation
│   └── torch_geometric/                        # Phase 1: Original PyG components
│
├── torch_geometric/                            # Phase 1: Custom PyG modifications
│   ├── nn/                                     # Phase 1: Custom neural network layers
│   └── utils/                                  # Phase 1: Custom utilities
│
├── graphgym/                                   # Phase 1: Custom GraphGym components
│
├── src/                                        # Phase 2: Source code directory
│   ├── __init__.py                            # Phase 2: Package initialization
│   │
│   ├── data/                                   # Phase 3: Data processing modules
│   │   ├── __init__.py                        # Phase 3: Data package init
│   │   ├── svg_processor.py                   # Phase 3: SVG parsing and graph construction
│   │   ├── datasets.py                        # Phase 3: Dataset classes and data loaders
│   │   └── transforms.py                      # Phase 3: Data augmentation transforms
│   │
│   ├── models/                                 # Phase 4: Model implementations
│   │   ├── __init__.py                        # Phase 4: Models package init
│   │   ├── modulated_gat.py                   # Phase 4: Modulated Graph Attention Layer
│   │   ├── two_stream_gnn.py                  # Phase 4: Two-stream GNN architecture
│   │   ├── backbone.py                        # Phase 4: CNN backbone networks
│   │   └── model_factory.py                   # Phase 4: Model creation and management
│   │
│   ├── training/                               # Phase 5: Training components
│   │   ├── __init__.py                        # Phase 5: Training package init
│   │   ├── trainer.py                         # Phase 5: Training manager class
│   │   ├── metrics.py                         # Phase 5: Evaluation metrics
│   │   └── schedulers.py                      # Phase 5: Learning rate schedulers
│   │
│   ├── utils/                                  # Phase 2: Utility functions
│   │   ├── __init__.py                        # Phase 2: Utils package init
│   │   ├── config.py                          # Phase 2: Configuration management
│   │   ├── logging_utils.py                   # Phase 2: Logging utilities
│   │   ├── visualization.py                   # Phase 2: Visualization tools
│   │   └── geometry_utils.py                  # Phase 3: Geometric processing utilities
│   │
│   └── losses/                                 # Phase 4: Custom loss functions
│       ├── __init__.py                        # Phase 4: Losses package init
│       ├── focal_loss.py                      # Phase 4: Focal loss implementation
│       └── boundary_loss.py                   # Phase 4: Boundary detection loss
│
├── data/                                       # Phase 2: Data directory
│   ├── raw/                                    # Phase 2: Original datasets
│   │   ├── R2V/                               # Phase 3: R2V dataset
│   │   │   ├── train/                         # Phase 3: Training split
│   │   │   │   ├── *.svg                      # Phase 3: SVG floorplan files
│   │   │   │   └── *.json                     # Phase 3: Annotation files
│   │   │   └── test/                          # Phase 3: Test split
│   │   │       ├── *.svg                      # Phase 3: SVG floorplan files
│   │   │       └── *.json                     # Phase 3: Annotation files
│   │   │
│   │   └── CubiCasa-5k/                       # Phase 3: CubiCasa-5k dataset
│   │       ├── svg/                           # Phase 3: SVG files directory
│   │       ├── labels/                        # Phase 3: Label files directory
│   │       ├── train.txt                      # Phase 3: Training split file
│   │       ├── val.txt                        # Phase 3: Validation split file
│   │       └── test.txt                       # Phase 3: Test split file
│   │
│   ├── processed/                              # Phase 2: Processed data cache
│   │   ├── R2V/                               # Phase 3: Processed R2V data
│   │   │   ├── train_metadata.json            # Phase 3: Training metadata
│   │   │   ├── test_metadata.json             # Phase 3: Test metadata
│   │   │   └── *.pkl                          # Phase 3: Cached processed samples
│   │   │
│   │   └── CubiCasa-5k/                       # Phase 3: Processed CubiCasa data
│   │       ├── train_metadata.json            # Phase 3: Training metadata
│   │       ├── val_metadata.json              # Phase 3: Validation metadata
│   │       ├── test_metadata.json             # Phase 3: Test metadata
│   │       └── *.pkl                          # Phase 3: Cached processed samples
│   │
│   └── datasets/                               # Phase 2: Dataset-specific utilities
│       ├── __init__.py                        # Phase 2: Datasets package init
│       └── download_scripts.py                # Phase 2: Dataset download utilities
│
├── models/                                     # Phase 2: Model storage directory
│   └── pretrained/                            # Phase 2: Pretrained backbone models
│       ├── resnet101-torch.pth                # Phase 2: ResNet-101 pretrained weights
│       ├── resnet50-torch.pth                 # Phase 2: ResNet-50 pretrained weights
│       └── vgg16-torch.pth                    # Phase 2: VGG-16 pretrained weights
│
├── configs/                                    # Phase 2: Configuration files
│   ├── baseline.yaml                          # Phase 2: Baseline configuration
│   ├── lightweight.yaml                       # Phase 2: Lightweight configuration
│   ├── debug.yaml                             # Phase 2: Debug configuration
│   ├── production.yaml                        # Phase 5: Production configuration
│   └── fast_training.yaml                     # Phase 5: Fast training configuration
│
├── outputs/                                    # Phase 2: Output directory
│   ├── checkpoints/                           # Phase 2: Model checkpoints
│   │   ├── best_model.pth                     # Phase 5: Best model checkpoint
│   │   ├── checkpoint_epoch_*.pth             # Phase 5: Epoch checkpoints
│   │   └── latest_checkpoint.pth              # Phase 5: Latest checkpoint
│   │
│   ├── logs/                                  # Phase 2: Training logs
│   │   ├── *.log                              # Phase 5: Training log files
│   │   └── tensorboard/                       # Phase 5: TensorBoard logs
│   │
│   ├── visualizations/                        # Phase 2: Result visualizations
│   │   ├── training_curves.png               # Phase 5: Training progress plots
│   │   ├── *_primal.png                      # Phase 3: Primal graph visualizations
│   │   ├── *_dual.png                        # Phase 3: Dual graph visualizations
│   │   └── attention_maps/                    # Phase 5: Attention visualization
│   │
│   ├── predictions/                           # Phase 5: Model predictions
│   │   ├── batch_*_predictions.json          # Phase 5: Batch prediction files
│   │   └── evaluation_metrics.json           # Phase 5: Evaluation results
│   │
│   └── inference/                             # Phase 5: Inference results
│       ├── *_predictions.json                # Phase 5: Single file predictions
│       └── *_segmentation.png                # Phase 5: Segmentation visualizations
│
├── experiments/                                # Phase 2: Experiment configurations
│   ├── exp_001_baseline/                      # Phase 5: Baseline experiment
│   ├── exp_002_ablation/                      # Phase 5: Ablation studies
│   └── exp_003_custom/                        # Phase 5: Custom experiments
│
├── notebooks/                                  # Phase 2: Jupyter notebooks
│   ├── data_exploration.ipynb                # Phase 3: Data analysis notebook
│   ├── model_analysis.ipynb                  # Phase 4: Model architecture analysis
│   ├── training_analysis.ipynb               # Phase 5: Training results analysis
│   └── visualization_examples.ipynb          # Phase 5: Visualization examples
│
├── tests/                                      # Phase 2: Unit tests
│   ├── __init__.py                            # Phase 2: Tests package init
│   ├── test_data_processing.py               # Phase 3: Data processing tests
│   ├── test_model_components.py              # Phase 4: Model component tests
│   ├── test_training.py                      # Phase 5: Training pipeline tests
│   └── test_inference.py                     # Phase 5: Inference pipeline tests
│
├── scripts/                                    # Phase 5: Utility scripts
│   ├── quick_start.py                         # Phase 5: Quick start workflow
│   ├── download_datasets.py                  # Phase 2: Dataset download script
│   ├── setup_environment.py                  # Phase 1: Environment setup script
│   └── benchmark.py                          # Phase 5: Performance benchmarking
│
├── docs/                                       # Phase 2: Documentation
│   ├── installation.md                       # Phase 1: Installation guide
│   ├── getting_started.md                    # Phase 5: Getting started guide
│   ├── api_reference.md                      # Phase 4: API documentation
│   └── troubleshooting.md                    # Phase 5: Troubleshooting guide
│
├── tools/                                      # Phase 2: Development tools
│   ├── visualize_graphs.py                   # Phase 3: Graph visualization tool
│   ├── convert_datasets.py                   # Phase 3: Dataset conversion utilities
│   └── model_profiler.py                     # Phase 4: Model performance profiling
│
├── examples/                                   # Phase 5: Usage examples
│   ├── basic_usage.py                        # Phase 5: Basic usage example
│   ├── custom_dataset.py                     # Phase 3: Custom dataset example
│   ├── inference_demo.py                     # Phase 5: Inference demonstration
│   └── sample_floorplans/                    # Phase 5: Sample input files
│       ├── simple_room.svg                   # Phase 5: Simple test floorplan
│       └── complex_layout.svg                # Phase 5: Complex test floorplan
│
# Phase-specific scripts (in project root)
├── setup_custom_pyg.py                       # Phase 1: Custom PyG setup script
├── verify_installation.py                    # Phase 1: Installation verification
├── download_backbone.py                      # Phase 2: Pretrained model download
├── setup_project.py                          # Phase 2: Project structure setup
├── preprocess_data.py                        # Phase 3: Data preprocessing script
├── test_model.py                              # Phase 4: Model testing script
├── train_vectorfloorseg.py                   # Phase 5: Main training script
└── evaluate_model.py                         # Phase 5: Model evaluation script
```

## Phase Summary

### Phase 1: Environment Setup and Dependencies
- **Virtual environment**: `vectorfloorseg_env/`
- **Custom PyG setup**: `torch_geometric/`, `graphgym/`
- **Setup scripts**: `setup_custom_pyg.py`, `verify_installation.py`
- **Documentation**: Installation guides and dependency management

### Phase 2: Project Structure Setup and Pretrained Models
- **Source organization**: `src/` with modular package structure
- **Configuration system**: `configs/` with YAML configuration files
- **Output management**: `outputs/` with organized subdirectories
- **Pretrained models**: `models/pretrained/` with backbone weights
- **Utilities**: Logging, visualization, and project management tools

### Phase 3: Data Preparation Pipeline
- **Data processing**: `src/data/` with SVG processing and dataset classes
- **Raw datasets**: `data/raw/` for R2V and CubiCasa-5k
- **Processed cache**: `data/processed/` for preprocessed graph data
- **Preprocessing script**: `preprocess_data.py` for complete data pipeline

### Phase 4: Model Implementation
- **Core models**: `src/models/` with modulated GAT and two-stream architecture
- **Model testing**: `test_model.py` for component verification
- **Custom losses**: `src/losses/` for specialized loss functions
- **Backbone networks**: CNN feature extractors with pretrained weights

### Phase 5: Training Script and Evaluation
- **Training system**: `src/training/` with comprehensive trainer class
- **Main scripts**: `train_vectorfloorseg.py` and `evaluate_model.py`
- **Quick start**: `scripts/quick_start.py` for automated workflow
- **Production configs**: Production-ready configuration templates
- **Examples**: Complete usage examples and demos

## Key Features by Phase

| Phase | Key Components | Primary Purpose |
|-------|----------------|-----------------|
| 1 | Environment, PyG setup | Foundation and dependencies |
| 2 | Project structure, configs | Organization and infrastructure |
| 3 | Data pipeline, SVG processing | Graph construction from vector data |
| 4 | Model architecture, GAT layers | Core VectorFloorSeg implementation |
| 5 | Training, evaluation, deployment | Complete ML pipeline |

This structure provides a complete, production-ready implementation of VectorFloorSeg with clear separation of concerns and comprehensive tooling for development, training, and deployment.