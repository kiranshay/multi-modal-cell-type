```python
"""
Multi-modal neural network for cell type classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ModalityEncoder(nn.Module):
    """Encoder network for a single modality"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout_rate: float = 0.3,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Final projection layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
        
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function"""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU()
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with masking for missing modalities
        
        Args:
            x: Input tensor [batch_size, input_dim]
            mask: Mask tensor [batch_size] (1 = present, 0 = missing)
        
        Returns:
            Encoded representation [batch_size, output_dim]
        """
        encoded = self.encoder(x)
        
        # Apply mask (zero out representations for missing modalities)
        mask = mask.unsqueeze(1).expand_as(encoded)
        encoded = encoded * mask
        
        return encoded

class AttentionFusion(nn.Module):
    """Attention-based fusion of multiple modalities"""
    
    def __init__(self, input_dim: int, attention_dim: int = 128):
        super().__init__()
        
        self.attention_dim = attention_dim
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, input_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(
        self, 
        modality_features: List[torch.Tensor], 
        modality_masks: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Fuse modality features using attention
        
        Args:
            modality_features: List of encoded features for each modality
            modality_masks: List of masks indicating available modalities
        
        Returns:
            Fused representation
        """
        batch_size = modality_features[0].size(0)
        
        # Stack modality features [batch_size, num_modalities, feature_dim]
        stacked_features = torch.stack(modality_features, dim=1)
        stacked_masks = torch.stack(modality_masks, dim=1)  # [batch_size, num_modalities]
        
        # Compute attention
        Q = self.query(stacked_features)  # [batch_size, num_modalities, attention_dim]
        K = self.key(stacked_features)    # [batch_size, num_modalities, attention_dim]
        V = self.value(stacked_features)  # [batch_size, num_modalities, feature_dim]
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.attention_dim ** 0.5)
        
        # Apply mask to attention scores (set masked positions to large negative values)
        mask_expanded = stacked_masks.unsqueeze(1).expand_as(attention_scores)
        attention_scores = attention_scores.masked_fill(~mask_expanded.bool(), -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        fused_features = torch.matmul(attention_weights, V)  # [batch_size, num_modalities, feature_dim]
        
        # Global pooling across modalities (weighted by availability)
        modality_contributions = stacked_masks.unsqueeze(-1).float()  # [batch_size, num_modalities, 1]
        fused_features = (fused_features * modality_contributions).sum(dim=1)  # [batch_size, feature_dim]
        
        # Normalize by number of available modalities
        num_available = stacked_masks.sum(dim=1, keepdim=True).float().clamp(min=1.0)
        fused_features = fused_features / num_available
        
        return fused_features

class MultiModalClassifier(nn.Module):
    """Multi-modal neural network for cell type classification"""
    
    def __init__(
        self,
        ephys_input_dim: int,
        morph_input_dim: int,
        transcr_input_dim: int,
        ephys_hidden_dims: List[int],
        morph_hidden_dims: List[int],
        transcr_hidden_dims: List[int],
        fusion_dim: int,
        fusion_hidden_dims: List[int],
        num_classes: int,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        self.fusion_dim = fusion_dim
        self.num_classes = num_classes
        
        # Modality encoders
        self.ephys_encoder = ModalityEncoder(
            ephys_input_dim, ephys_hidden_dims, fusion_dim, dropout_rate
        )
        self.morph_encoder = ModalityEncoder(
            morph_input_dim, morph_hidden_dims, fusion_dim, dropout_rate
        )
        self.transcr_encoder = ModalityEncoder(
            transcr_input_dim, transcr_hidden_dims, fusion_dim, dropout_rate
        )
        
        # Attention-based fusion
        self.fusion = AttentionFusion(fusion_dim)
        
        # Classification head
        classifier_layers = []
        prev_dim = fusion_dim
        
        for hidden_dim in fusion_hidden_dims:
            classifier_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        classifier_layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-modal network
        
        Args:
            batch: Dictionary containing modality data and masks
        
        Returns:
            Dictionary with predictions and intermediate representations
        """
        # Encode each modality
        ephys_features = self.ephys_encoder(batch['ephys_data'], batch['ephys_mask'])
        morph_features = self.morph_encoder(batch['morph_data'], batch['morph_mask'])
        transcr_features = self.transcr_encoder(batch['transcr_data'], batch['transcr_mask'])
        
        # Fuse modalities using attention
        modality_features = [ephys_features, morph_features, transcr_features]
        modality_masks = [batch['ephys_mask'], batch['morph_mask'], batch['transcr_mask']]
        
        fused_features = self.fusion(modality_features, modality_masks)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return {
            'logits': logits,
            'predictions': torch.softmax(logits, dim=1),
            'ephys_features': ephys_features,
            'morph_features': morph_features,
            'transcr_features': transcr_features,
            'fused_features': fused_features
        }
    
    def get_feature_representations(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract feature representations without classification"""
        with torch.no_grad():
            outputs = self.forward(batch)
            return {
                'ephys_features': outputs['ephys_features'],
                'morph_features': outputs['morph_features'], 
                'transcr_features': outputs['transcr_features'],
                'fused_features': outputs['fused_features']
            }
```
