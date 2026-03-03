```python
"""
Multi-modal dataset class for cell type classification
"""
import os
import pickle
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from allensdk.core.cell_types_cache import CellTypesCache
import logging

logger = logging.getLogger(__name__)

class MultiModalCellDataset(Dataset):
    """
    Dataset class for multi-modal cell type classification.
    Handles electrophysiology, morphology, and transcriptomics data.
    """
    
    def __init__(
        self,
        data_dir: str,
        cache_dir: str,
        split: str = "train",
        top_genes: int = 2000,
        transform: Optional[Dict] = None,
        missing_modality_prob: float = 0.0
    ):
        """
        Initialize multi-modal dataset.
        
        Args:
            data_dir: Directory containing data files
            cache_dir: Directory for caching processed data
            split: Dataset split ("train", "val", "test")
            top_genes: Number of top variable genes to keep
            transform: Dictionary of transforms for each modality
            missing_modality_prob: Probability of randomly masking modalities
        """
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.split = split
        self.top_genes = top_genes
        self.transform = transform or {}
        self.missing_modality_prob = missing_modality_prob
        
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load and preprocess data
        self.data = self._load_data()
        self.scalers = self._fit_scalers() if split == "train" else self._load_scalers()
        self._normalize_features()
        
        logger.info(f"Loaded {len(self.data)} samples for {split} split")
    
    def _load_data(self) -> Dict:
        """Load multi-modal data from Allen Cell Types Database"""
        cache_file = os.path.join(self.cache_dir, f"{self.split}_data.pkl")
        
        if os.path.exists(cache_file):
            logger.info(f"Loading cached data from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        logger.info("Loading data from Allen Cell Types Database...")
        
        # Initialize Allen SDK cache
        ctc = CellTypesCache(manifest_file=os.path.join(self.cache_dir, 'manifest.json'))
        
        # Get cell metadata
        cells = ctc.get_cells()
        ephys_features = ctc.get_ephys_features()
        morph_features = ctc.get_morphology_features()
        
        data = {
            'ephys_data': [],
            'morph_data': [],
            'transcr_data': [],
            'cell_types': [],
            'cell_ids': []
        }
        
        # Load electrophysiology data
        for cell in cells:
            cell_id = cell['id']
            
            try:
                # Get ephys features
                ephys_row = ephys_features[ephys_features.specimen_id == cell_id]
                if not ephys_row.empty:
                    ephys_feats = ephys_row.select_dtypes(include=[np.number]).values.flatten()
                    data['ephys_data'].append(ephys_feats)
                else:
                    data['ephys_data'].append(None)
                
                # Get morphology features  
                morph_row = morph_features[morph_features.specimen_id == cell_id]
                if not morph_row.empty:
                    morph_feats = morph_row.select_dtypes(include=[np.number]).values.flatten()
                    data['morph_data'].append(morph_feats)
                else:
                    data['morph_data'].append(None)
                
                # Get transcriptomics (placeholder - would need actual RNA-seq data)
                # For demo, we'll simulate this
                transcr_feats = np.random.randn(self.top_genes) if np.random.rand() > 0.3 else None
                data['transcr_data'].append(transcr_feats)
                
                data['cell_types'].append(cell.get('transgenic_line', 'Unknown'))
                data['cell_ids'].append(cell_id)
                
            except Exception as e:
                logger.warning(f"Error processing cell {cell_id}: {e}")
                continue
        
        # Convert to arrays and filter complete cases
        data = self._filter_and_convert_data(data)
        
        # Cache processed data
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        
        return data
    
    def _filter_and_convert_data(self, data: Dict) -> Dict:
        """Filter out samples with all missing modalities and convert to arrays"""
        filtered_data = {
            'ephys_data': [],
            'morph_data': [], 
            'transcr_data': [],
            'cell_types': [],
            'cell_ids': []
        }
        
        for i in range(len(data['cell_ids'])):
            # Keep samples with at least one modality present
            has_data = any([
                data['ephys_data'][i] is not None,
                data['morph_data'][i] is not None,
                data['transcr_data'][i] is not None
            ])
            
            if has_data:
                filtered_data['ephys_data'].append(data['ephys_data'][i])
                filtered_data['morph_data'].append(data['morph_data'][i])
                filtered_data['transcr_data'].append(data['transcr_data'][i])
                filtered_data['cell_types'].append(data['cell_types'][i])
                filtered_data['cell_ids'].append(data['cell_ids'][i])
        
        # Encode cell types
        self.label_encoder = LabelEncoder()
        filtered_data['encoded_labels'] = self.label_encoder.fit_transform(filtered_data['cell_types'])
        
        return filtered_data
    
    def _fit_scalers(self) -> Dict:
        """Fit scalers on training data"""
        scalers = {}
        
        # Ephys scaler
        ephys_data = [x for x in self.data['ephys_data'] if x is not None]
        if ephys_data:
            ephys_array = np.vstack(ephys_data)
            scalers['ephys'] = StandardScaler().fit(ephys_array)
        
        # Morphology scaler  
        morph_data = [x for x in self.data['morph_data'] if x is not None]
        if morph_data:
            morph_array = np.vstack(morph_data)
            scalers['morph'] = StandardScaler().fit(morph_array)
        
        # Transcriptomics scaler
        transcr_data = [x for x in self.data['transcr_data'] if x is not None]
        if transcr_data:
            transcr_array = np.vstack(transcr_data)
            scalers['transcr'] = StandardScaler().fit(transcr_array)
        
        # Save scalers
        scaler_file = os.path.join(self.cache_dir, 'scalers.pkl')
        with open(scaler_file, 'wb') as f:
            pickle.dump(scalers, f)
        
        return scalers
    
    def _load_scalers(self) -> Dict:
        """Load pre-fitted scalers"""
        scaler_file = os.path.join(self.cache_dir, 'scalers.pkl')
        with open(scaler_file, 'rb') as f:
            return pickle.load(f)
    
    def _normalize_features(self):
        """Apply normalization to features"""
        for i in range(len(self.data['ephys_data'])):
            if self.data['ephys_data'][i] is not None and 'ephys' in self.scalers:
                self.data['ephys_data'][i] = self.scalers['ephys'].transform(
                    self.data['ephys_data'][i].reshape(1, -1)
                ).flatten()
            
            if self.data['morph_data'][i] is not None and 'morph' in self.scalers:
                self.data['morph_data'][i] = self.scalers['morph'].transform(
                    self.data['morph_data'][i].reshape(1, -1)
                ).flatten()
            
            if self.data['transcr_data'][i] is not None and 'transcr' in self.scalers:
                self.data['transcr_data'][i] = self.scalers['transcr'].transform(
                    self.data['transcr_data'][i].reshape(1, -1)
                ).flatten()
    
    def __len__(self) -> int:
        return len(self.data['cell_ids'])
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset"""
        sample = {
            'cell_id': self.data['cell_ids'][idx],
            'label': torch.tensor(self.data['encoded_labels'][idx], dtype=torch.long)
        }
        
        # Get modality data with masking for missing data handling
        modalities = ['ephys', 'morph', 'transcr']
        data_keys = ['ephys_data', 'morph_data', 'transcr_data']
        
        for modality, data_key in zip(modalities, data_keys):
            data = self.data[data_key][idx]
            
            if data is not None:
                # Apply transforms if specified
                if modality in self.transform:
                    data = self.transform[modality](data)
                
                # Random modality masking during training
                if self.split == "train" and np.random.rand() < self.missing_modality_prob:
                    sample[f'{modality}_data'] = torch.zeros_like(torch.tensor(data, dtype=torch.float32))
                    sample[f'{modality}_mask'] = torch.tensor(0.0)
                else:
                    sample[f'{modality}_data'] = torch.tensor(data, dtype=torch.float32)
                    sample[f'{modality}_mask'] = torch.tensor(1.0)
            else:
                # Handle naturally missing data
                expected_dims = {
                    'ephys': 50, 'morph': 100, 'transcr': self.top_genes
                }
                sample[f'{modality}_data'] = torch.zeros(expected_dims[modality], dtype=torch.float32)
                sample[f'{modality}_mask'] = torch.tensor(0.0)
        
        return sample
    
    def get_class_names(self) -> List[str]:
        """Get list of class names"""
        return list(self.label_encoder.classes_)
    
    def get_num_classes(self) -> int:
        """Get number of classes"""
        return len(self.label_encoder.classes_)
```
