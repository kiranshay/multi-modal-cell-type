```python
"""
Training loop for multi-modal cell type classifier
"""
import os
import logging
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
import json

from ..models.multimodal_classifier import MultiModalClassifier
from ..config import ModelConfig, TrainingConfig

logger = logging.getLogger(__name__)

class MultiModalTrainer:
    """Trainer for multi-modal cell type classifier"""
    
    def __init__(
        self,
        model: MultiModalClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        save_dir: str,
        class_names: Optional[list] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.save_dir = save_dir
        self.class_names = class_names or [f"Class_{i}" for i in range(model.num_classes)]
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup device
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma
        )
        
        # Setup loss function with class weights if needed
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup tensorboard
        self.writer = SummaryWriter(os.path.join(save_dir, 'tensorboard'))
        
        # Training state
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.patience_counter = 0
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
        
        logger.info(f"Trainer initialized. Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            
            # Compute loss
            loss = self.criterion(outputs['logits'], batch['label'])
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            batch_size = batch['label'].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Collect predictions
            predictions = torch.argmax(outputs['predictions'], dim=1).cpu().numpy()
            labels = batch['label'].cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'avg_loss': total_loss / total_samples,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Log batch metrics
            if batch_idx % 100 == 0:
                self.writer.add_scalar(
                    'Train/BatchLoss',
                    loss.item(),
                    self.current_epoch * len(self.train_loader) + batch_idx
                )
        
        # Compute epoch metrics
        avg_loss = total_loss / total_samples
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_score': f1
        }
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch+1} [Val]")
            
            for batch in pbar:
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch)
                loss = self.criterion(outputs['logits'], batch['label'])
                
                # Update metrics
                batch_size = batch['label'].size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Collect predictions
                predictions = torch.argmax(outputs['predictions'], dim=1).cpu().numpy()
                labels = batch['label'].cpu().numpy()
                all_predictions.extend(predictions)
                all_labels.extend(labels)
                
                pbar.set_postfix({'val_loss': loss.item()})
        
        # Compute metrics
        avg_loss = total_loss / total_samples
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        # Compute per-class metrics
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': conf_matrix
        }
        
        return metrics
    
    def train(self) -> Dict:
        """Main training loop"""
        logger.info("Starting training...")
        
        for epoch