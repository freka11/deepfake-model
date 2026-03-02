import os
import psutil
import torch
import gc
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

class MemoryMonitor:
    """Monitor and manage memory usage during training"""
    
    def __init__(self, memory_limit_gb: float = 6.0):
        self.memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
        self.process = psutil.Process()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        memory_info = self.process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return {
            'process_mb': memory_info.rss / 1024 / 1024,
            'system_used_percent': system_memory.percent,
            'system_available_gb': system_memory.available / 1024 / 1024 / 1024,
            'process_percent': (memory_info.rss / self.memory_limit_bytes) * 100
        }
    
    def check_memory_limit(self) -> bool:
        """Check if we're approaching memory limit"""
        usage = self.get_memory_usage()
        
        if usage['process_mb'] > (self.memory_limit_bytes / 1024 / 1024) * 0.9:
            self.logger.warning(f"⚠️ Memory usage high: {usage['process_mb']:.1f}MB")
            return False
        
        return True
    
    def cleanup_memory(self):
        """Force garbage collection and cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("🧹 Memory cleanup completed")
    
    def log_memory_status(self, step: str = ""):
        """Log current memory status"""
        usage = self.get_memory_usage()
        self.logger.info(f"📊 Memory {step}: "
                        f"Process: {usage['process_mb']:.1f}MB "
                        f"({usage['process_percent']:.1f}%), "
                        f"System: {usage['system_used_percent']:.1f}% used, "
                        f"{usage['system_available_gb']:.1f}GB available")

class DatasetManager:
    """Manage multiple datasets with memory-efficient loading"""
    
    def __init__(self, config_path: str = "config_multi.yaml"):
        self.config = self._load_config(config_path)
        self.memory_monitor = MemoryMonitor(
            memory_limit_gb=self.config.get('memory_limit_gb', 6.0)
        )
        self.datasets = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_kaggle_dataset(self, kaggle_path: str):
        """Setup Kaggle dataset paths in config"""
        if not kaggle_path or not os.path.exists(kaggle_path):
            raise ValueError("Invalid Kaggle dataset path")
        
        # Update config with actual paths
        self.config['kaggle_dataset_path'] = kaggle_path
        self.config['train_paths'] = [os.path.join(kaggle_path, 'train')]
        self.config['val_paths'] = [os.path.join(kaggle_path, 'valid')]
        self.config['test_paths'] = [os.path.join(kaggle_path, 'test')]
        
        # Update datasets section
        if 'datasets' in self.config and 'kaggle_140k' in self.config['datasets']:
            self.config['datasets']['kaggle_140k']['path'] = kaggle_path
            self.config['datasets']['kaggle_140k']['enabled'] = True
        
        self._save_config()
        self.memory_monitor.log_memory_status("after_kaggle_setup")
    
    def _save_config(self):
        """Save current configuration"""
        with open('config_multi.yaml', 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def get_enabled_datasets(self) -> List[str]:
        """Get list of enabled datasets"""
        enabled = []
        if 'datasets' in self.config:
            for name, dataset_config in self.config['datasets'].items():
                if dataset_config.get('enabled', False):
                    enabled.append(name)
        return enabled
    
    def get_dataset_paths(self, split: str) -> List[Tuple[str, Optional[int]]]:
        """Get dataset paths for a specific split"""
        paths = []
        
        # Use legacy config for backward compatibility
        if f'{split}_paths' in self.config:
            for path in self.config[f'{split}_paths']:
                paths.append((path, None))
        
        # Use new datasets config
        if 'datasets' in self.config:
            for name, dataset_config in self.config['datasets'].items():
                if dataset_config.get('enabled', False):
                    dataset_path = dataset_config['path']
                    subset_path = os.path.join(dataset_path, dataset_config['subsets'][split])
                    if os.path.exists(subset_path):
                        paths.append((subset_path, None))
        
        return paths
    
    def validate_datasets(self) -> bool:
        """Validate all enabled datasets"""
        self.memory_monitor.log_memory_status("dataset_validation_start")
        
        all_valid = True
        for dataset_name in self.get_enabled_datasets():
            if not self._validate_single_dataset(dataset_name):
                all_valid = False
        
        self.memory_monitor.log_memory_status("dataset_validation_end")
        return all_valid
    
    def _validate_single_dataset(self, dataset_name: str) -> bool:
        """Validate a single dataset structure"""
        if dataset_name not in self.config['datasets']:
            return False
        
        dataset_config = self.config['datasets'][dataset_name]
        dataset_path = dataset_config['path']
        
        # Check if dataset path exists
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset path not found: {dataset_path}")
            return False
        
        # Check required subsets
        for subset_name in ['train', 'val', 'test']:
            subset_path = os.path.join(dataset_path, dataset_config['subsets'][subset_name])
            if not os.path.exists(subset_path):
                print(f"❌ Subset not found: {subset_path}")
                return False
            
            # Check for real/fake folders
            real_path = os.path.join(subset_path, 'real')
            fake_path = os.path.join(subset_path, 'fake')
            
            if not (os.path.exists(real_path) and os.path.exists(fake_path)):
                print(f"❌ Real/fake folders not found in {subset_path}")
                return False
        
        print(f"✅ Dataset {dataset_name} validated")
        return True

# Global instance for easy access
dataset_manager = DatasetManager()
