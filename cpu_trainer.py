import os
import torch
import yaml
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning_modules.detector import DeepfakeDetector
from dataset_manager import dataset_manager
from sampling_strategy import SamplingStrategy
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiDatasetTrainer:
    """CPU-optimized trainer for multi-dataset deepfake detection"""
    
    def __init__(self, config_path: str = "config_multi.yaml"):
        self.config = self._load_config(config_path)
        self.dataset_manager = dataset_manager
        self.sampling_strategy = SamplingStrategy(self.config)
        
        # Setup memory monitoring
        self.dataset_manager.memory_monitor.log_memory_status("trainer_init")
        
    def _load_config(self, config_path: str) -> dict:
        """Load training configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_model(self):
        """Setup EfficientNet model for deepfake detection"""
        logger.info("🏗️ Setting up EfficientNet-B0 model...")
        
        # Load pretrained EfficientNet
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = efficientnet_b0(weights=weights)
        
        # Modify classifier for binary classification
        features = backbone.classifier[1].in_features
        backbone.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(features, 2)  # 2 classes: real (0), fake (1)
        )
        
        # Create Lightning module
        model = DeepfakeDetector(backbone, lr=self.config['lr'])
        
        logger.info("✅ Model setup complete")
        return model
    
    def setup_callbacks(self):
        """Setup training callbacks"""
        callbacks = []
        
        # Model checkpointing
        checkpoint = ModelCheckpoint(
            monitor=self.config.get('monitor_metric', 'val_loss'),
            dirpath="models",
            filename="multi_dataset_model-{epoch:02d}-{val_loss:.2f}",
            save_top_k=self.config.get('save_top_k', 3),
            mode='min',
            save_last=True
        )
        callbacks.append(checkpoint)
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor=self.config.get('monitor_metric', 'val_loss'),
            patience=self.config.get('early_stopping_patience', 3),
            mode='min',
            verbose=True
        )
        callbacks.append(early_stop)
        
        # Learning rate monitoring
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)
        
        return callbacks
    
    def setup_dataloaders(self):
        """Setup training and validation dataloaders"""
        logger.info("📊 Setting up dataloaders...")
        
        # Get dataset paths
        train_paths = self.dataset_manager.get_dataset_paths('train')
        val_paths = self.dataset_manager.get_dataset_paths('val')
        
        if not train_paths:
            raise ValueError("No training datasets found")
        
        if not val_paths:
            raise ValueError("No validation datasets found")
        
        # Create dataloaders using sampling strategy
        train_loader = self.sampling_strategy.create_train_dataloader(train_paths)
        val_loader = self.sampling_strategy.create_val_dataloader(val_paths)
        
        logger.info(f"✅ Dataloaders ready - Train: {len(train_loader)} batches, "
                   f"Val: {len(val_loader)} batches")
        
        return train_loader, val_loader
    
    def setup_trainer(self):
        """Setup PyTorch Lightning trainer with CPU optimizations"""
        logger.info("🚀 Setting up PyTorch Lightning trainer...")
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Configure trainer for CPU
        trainer_config = {
            'max_epochs': self.config.get('num_epochs', 10),
            'accelerator': 'cpu',  # Force CPU training
            'devices': 1,
            'callbacks': callbacks,
            'enable_progress_bar': True,
            'enable_model_summary': True,
            'log_every_n_steps': self.config.get('log_every_n_steps', 10),
            'deterministic': True,  # For reproducibility
        }
        
        # Add gradient accumulation
        if 'gradient_accumulation_steps' in self.config:
            trainer_config['accumulate_grad_batches'] = self.config['gradient_accumulation_steps']
        
        trainer = pl.Trainer(**trainer_config)
        
        logger.info("✅ Trainer setup complete")
        return trainer
    
    def train(self):
        """Run the complete training pipeline"""
        try:
            logger.info("🎯 Starting multi-dataset training...")
            
            # Memory check before training
            if not self.dataset_manager.memory_monitor.check_memory_limit():
                logger.error("❌ Memory usage too high, cannot start training")
                return
            
            # Validate datasets
            if not self.dataset_manager.validate_datasets():
                logger.error("❌ Dataset validation failed")
                return
            
            # Setup components
            model = self.setup_model()
            train_loader, val_loader = self.setup_dataloaders()
            trainer = self.setup_trainer()
            
            # Log final memory status
            self.dataset_manager.memory_monitor.log_memory_status("before_training")
            
            # Start training
            logger.info("🏃‍♂️ Beginning training...")
            trainer.fit(model, train_loader, val_loader)
            
            # Log completion
            self.dataset_manager.memory_monitor.log_memory_status("training_complete")
            logger.info("✅ Training completed successfully!")
            
            # Save final model path
            best_model_path = trainer.checkpoint_callback.best_model_path
            logger.info(f"🏆 Best model saved at: {best_model_path}")
            
            return best_model_path
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            self.dataset_manager.memory_monitor.cleanup_memory()
            raise
    
    def train_single_epoch(self, model, train_loader, val_loader, epoch):
        """Alternative training method for more control"""
        logger.info(f"📚 Training epoch {epoch}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Memory check
            if batch_idx % 50 == 0:
                self.dataset_manager.memory_monitor.check_memory_limit()
            
            # Forward pass
            outputs = model(images)
            loss = model.loss_fn(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.get('gradient_accumulation_steps', 1) == 0:
                model.optimizer.step()
                model.optimizer.zero_grad()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = model.loss_fn(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Log epoch results
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}, "
                   f"Train Acc: {train_acc:.2f}%, "
                   f"Val Loss: {avg_val_loss:.4f}, "
                   f"Val Acc: {val_acc:.2f}%")
        
        return avg_train_loss, avg_val_loss, train_acc, val_acc

def main():
    """Main training function"""
    try:
        # Initialize trainer
        trainer = MultiDatasetTrainer()
        
        # Run training
        best_model_path = trainer.train()
        
        print(f"\n🎉 Training completed!")
        print(f"🏆 Best model: {best_model_path}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
