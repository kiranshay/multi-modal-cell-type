```python
"""
Configuration settings for Multi-Modal Cell Type Classifier
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Electrophysiology encoder
    ephys_input_dim: int = 50  # Number of ephys features
    ephys_hidden_dims: List[int] = None
    
    # Morphology encoder
    morph_input_dim: int = 100  # Number of morphology features
    morph_hidden_dims: List[int] = None
    
    # Transcriptomics encoder
    transcr_input_dim: int = 2000  # Number of gene features (top variable genes)
    transcr_hidden_dims: List[int] = None
    
    # Fusion network
    fusion_dim: int = 256
    fusion_hidden_dims: List[int] = None
    
    # Output
    num_classes: int = 20  # Number of cell types
    dropout_rate: float = 0.3
    
    def __post_init__(self):
        if self.ephys_hidden_dims is None:
            self.ephys_hidden_dims = [128, 64]
        if self.morph_hidden_dims is None:
            self.morph_hidden_dims = [256, 128]
        if self.transcr_hidden_dims is None:
            self.transcr_hidden_dims = [1024, 512, 256]
        if self.fusion_hidden_dims is None:
            self.fusion_hidden_dims = [512, 256]

@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    weight_decay: float = 1e-4
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.1
    early_stopping_patience: int = 15
    
    # Loss weights for multi-modal training
    modality_weights: Dict[str, float] = None
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.modality_weights is None:
            self.modality_weights = {
                "ephys": 1.0,
                "morph": 1.0, 
                "transcr": 1.0,
                "fusion": 2.0
            }

@dataclass
class DataConfig:
    """Data loading and preprocessing configuration"""
    data_dir: str = "data/"
    cache_dir: str = "data/cache/"
    
    # Feature selection
    top_genes: int = 2000
    min_gene_expression: float = 0.1
    
    # Train/val/test splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Data augmentation
    ephys_noise_std: float = 0.01
    morph_jitter_std: float = 0.005
    transcr_dropout_rate: float = 0.1
    
    random_seed: int = 42
```
