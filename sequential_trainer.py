import os
import torch
import yaml
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning_modules.detector import DeepfakeDetector
from dataset_streamer import create_streaming_dataloader
from dataset_manager import dataset_manager
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SequentialTrainer:
    """Sequential multi-dataset trainer with streaming support"""
    
    def __init__(self, config_path: str = "config_multi.yaml"):
        self.config = self._load_config(config_path)
        self.dataset_manager = dataset_manager
        self.memory_monitor = self.dataset_manager.memory_monitor
        
        # Setup logging
        self.setup_logging()
        
    def _load_config(self, config_path: str) -> dict:
        """Load training configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_logging(self):
        """Setup detailed logging"""
        logger.info("🚀 Sequential Multi-Dataset Trainer Initialized")
        logger.info(f"📊 Memory limit: {self.config.get('memory_limit_gb', 6)}GB")
        logger.info(f"📦 Batch size: {self.config.get('batch_size', 2)}")
        logger.info(f"🔄 Gradient accumulation: {self.config.get('gradient_accumulation_steps', 8)}")
    
    def setup_model(self):
        """Setup EfficientNet model"""
        logger.info("🏗️ Setting up EfficientNet-B0 model...")
        
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = efficientnet_b0(weights=weights)
        
        # Modify classifier for binary classification
        features = backbone.classifier[1].in_features
        backbone.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(features, 2)
        )
        
        model = DeepfakeDetector(backbone, lr=self.config['lr'])
        
        logger.info("✅ Model setup complete")
        return model
    
    def get_sequential_stages(self) -> list:
        """Get sequential training stages from config"""
        
        stages = []
        
        # Default stages if not configured
        if 'sequential_training' not in self.config:
            stages = [
                {
                    'name': 'kaggle_baseline',
                    'datasets': ['kaggle_140k'],
                    'epochs': 10,
                    'description': 'Baseline training on Kaggle dataset'
                }
            ]
        else:
            stages = self.config['sequential_training']['stages']
        
        logger.info(f"📋 Found {len(stages)} training stages")
        for stage in stages:
            logger.info(f"  🎯 {stage['name']}: {stage.get('description', '')}")
        
        return stages
    
    def get_stage_datasets(self, stage_config: dict) -> list:
        """Get dataset paths for a specific stage"""
        
        dataset_names = stage_config['datasets']
        dataset_paths = []
        dataset_weights = []
        
        for dataset_name in dataset_names:
            if dataset_name in self.config['datasets']:
                dataset_config = self.config['datasets'][dataset_name]
                
                if dataset_config.get('enabled', False):
                    # Get paths for each split
                    base_path = dataset_config['path']
                    subsets = dataset_config['subsets']
                    
                    # Use train paths for training
                    train_path = os.path.join(base_path, subsets['train'])
                    if os.path.exists(train_path):
                        dataset_paths.append((train_path, None))
                        dataset_weights.append(dataset_config.get('weight', 1.0))
                        logger.info(f"  📁 Added {dataset_name}: {train_path}")
                    else:
                        logger.warning(f"  ⚠️ Path not found: {train_path}")
                else:
                    logger.warning(f"  ⚠️ Dataset disabled: {dataset_name}")
            else:
                logger.warning(f"  ⚠️ Dataset not found: {dataset_name}")
        
        return dataset_paths, dataset_weights
    
    def create_stage_dataloader(self, dataset_paths: list, dataset_weights: list):
        """Create streaming dataloader for stage"""
        
        if not dataset_paths:
            raise ValueError("No valid datasets for this stage")
        
        # Create transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Create streaming dataloader
        dataloader = create_streaming_dataloader(
            dataset_paths=dataset_paths,
            batch_size=self.config.get('batch_size', 2),
            chunk_size=self.config.get('chunk_size', 1000),
            memory_limit_mb=self.config.get('memory_limit_gb', 6) * 1024,
            transform=transform,
            dataset_weights=dataset_weights,
            shuffle=True
        )
        
        return dataloader
    
    def setup_callbacks(self, stage_name: str):
        """Setup training callbacks for stage"""
        
        callbacks = []
        
        # Model checkpointing
        checkpoint = ModelCheckpoint(
            monitor=self.config.get('monitor_metric', 'val_loss'),
            dirpath=f"models/sequential_{stage_name}",
            filename=f"{stage_name}-{{epoch:02d}}-{{val_loss:.2f}}",
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
    
    def setup_trainer(self, stage_name: str, max_epochs: int):
        """Setup PyTorch Lightning trainer"""
        
        callbacks = self.setup_callbacks(stage_name)
        
        trainer_config = {
            'max_epochs': max_epochs,
            'accelerator': 'cpu',
            'devices': 1,
            'callbacks': callbacks,
            'enable_progress_bar': True,
            'enable_model_summary': True,
            'log_every_n_steps': self.config.get('log_every_n_steps', 10),
            'deterministic': True,
        }
        
        # Add gradient accumulation
        if 'gradient_accumulation_steps' in self.config:
            trainer_config['accumulate_grad_batches'] = self.config['gradient_accumulation_steps']
        
        trainer = pl.Trainer(**trainer_config)
        
        logger.info(f"✅ Trainer setup for {stage_name}")
        return trainer
    
    def train_stage(self, stage_config: dict):
        """Train a single stage"""
        
        stage_name = stage_config['name']
        epochs = stage_config['epochs']
        description = stage_config.get('description', '')
        
        logger.info(f"\n🎯 Starting Stage: {stage_name}")
        logger.info(f"📝 Description: {description}")
        logger.info(f"🔄 Epochs: {epochs}")
        
        # Memory check
        if not self.memory_monitor.check_memory_limit():
            logger.warning("⚠️ High memory usage, performing cleanup...")
            self.memory_monitor.cleanup_memory()
        
        try:
            # Get datasets for this stage
            dataset_paths, dataset_weights = self.get_stage_datasets(stage_config)
            
            if not dataset_paths:
                logger.error(f"❌ No valid datasets for stage: {stage_name}")
                return False
            
            # Create dataloader
            train_loader = self.create_stage_dataloader(dataset_paths, dataset_weights)
            
            # Create validation dataloader (use all available validation data)
            val_paths = []
            val_weights = []
            for dataset_name in stage_config['datasets']:
                if dataset_name in self.config['datasets']:
                    dataset_config = self.config['datasets'][dataset_name]
                    if dataset_config.get('enabled', False):
                        base_path = dataset_config['path']
                        subsets = dataset_config['subsets']
                        val_path = os.path.join(base_path, subsets['val'])
                        if os.path.exists(val_path):
                            val_paths.append((val_path, None))
                            val_weights.append(dataset_config.get('weight', 1.0))
            
            val_loader = None
            if val_paths:
                val_loader = self.create_stage_dataloader(val_paths, val_weights)
            
            # Setup model and trainer
            model = self.setup_model()
            trainer = self.setup_trainer(stage_name, epochs)
            
            # Log memory status
            self.memory_monitor.log_memory_status(f"before_{stage_name}")
            
            # Train stage
            logger.info(f"🏃‍♂️ Training {stage_name}...")
            trainer.fit(model, train_loader, val_loader)
            
            # Log completion
            self.memory_monitor.log_memory_status(f"after_{stage_name}")
            
            # Get best model path
            best_model_path = trainer.checkpoint_callback.best_model_path
            logger.info(f"✅ Stage {stage_name} completed!")
            logger.info(f"🏆 Best model: {best_model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Stage {stage_name} failed: {e}")
            return False
    
    def train_sequential(self):
        """Run complete sequential training"""
        
        logger.info("🚀 Starting Sequential Multi-Dataset Training")
        logger.info("=" * 60)
        
        # Get training stages
        stages = self.get_sequential_stages()
        
        if not stages:
            logger.error("❌ No training stages configured")
            return False
        
        # Train each stage
        successful_stages = []
        failed_stages = []
        
        for i, stage_config in enumerate(stages, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"STAGE {i}/{len(stages)}: {stage_config['name']}")
            logger.info(f"{'='*60}")
            
            success = self.train_stage(stage_config)
            
            if success:
                successful_stages.append(stage_config['name'])
            else:
                failed_stages.append(stage_config['name'])
                
                # Decide whether to continue
                if i < len(stages):
                    logger.warning(f"⚠️ Stage failed, continuing to next stage...")
        
        # Final summary
        logger.info(f"\n{'='*60}")
        logger.info("🎉 SEQUENTIAL TRAINING COMPLETE!")
        logger.info(f"{'='*60}")
        
        logger.info(f"✅ Successful stages: {len(successful_stages)}")
        for stage in successful_stages:
            logger.info(f"  🎯 {stage}")
        
        if failed_stages:
            logger.warning(f"❌ Failed stages: {len(failed_stages)}")
            for stage in failed_stages:
                logger.warning(f"  ⚠️ {stage}")
        
        logger.info(f"📊 Success rate: {len(successful_stages)/len(stages)*100:.1f}%")
        
        return len(successful_stages) > 0

def main():
    """Main training function"""
    
    try:
        # Initialize trainer
        trainer = SequentialTrainer()
        
        # Run sequential training
        success = trainer.train_sequential()
        
        if success:
            logger.info("\n🎉 Sequential training completed successfully!")
            logger.info("🎯 Models saved in models/sequential_*/ directories")
        else:
            logger.error("\n❌ Sequential training failed")
            
    except Exception as e:
        logger.error(f"❌ Training failed with exception: {e}")
        raise

if __name__ == "__main__":
    main()
