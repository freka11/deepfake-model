import torch
import random
import numpy as np
from typing import List, Tuple, Dict, Optional
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from datasets.hybrid_loader import HybridDeepfakeDataset
import logging

class MultiDatasetSampler:
    """Handle sampling from multiple datasets with different strategies"""
    
    def __init__(self, strategy: str = "weighted", balance_datasets: bool = True):
        self.strategy = strategy
        self.balance_datasets = balance_datasets
        self.logger = logging.getLogger(__name__)
    
    def create_combined_dataloader(
        self, 
        datasets: List[HybridDeepfakeDataset], 
        batch_size: int = 2,
        num_workers: int = 2,
        dataset_weights: Optional[List[float]] = None
    ) -> DataLoader:
        """Create a combined dataloader from multiple datasets"""
        
        if self.strategy == "weighted":
            return self._create_weighted_dataloader(
                datasets, batch_size, num_workers, dataset_weights
            )
        elif self.strategy == "sequential":
            return self._create_sequential_dataloader(
                datasets, batch_size, num_workers
            )
        else:
            return self._create_random_dataloader(
                datasets, batch_size, num_workers
            )
    
    def _create_weighted_dataloader(
        self, 
        datasets: List[HybridDeepfakeDataset], 
        batch_size: int,
        num_workers: int,
        dataset_weights: Optional[List[float]]
    ) -> DataLoader:
        """Create weighted sampler for balanced multi-dataset training"""
        
        # Calculate dataset weights if not provided
        if dataset_weights is None:
            if self.balance_datasets:
                # Equal weight for each dataset
                dataset_weights = [1.0 / len(datasets)] * len(datasets)
            else:
                # Weight by dataset size
                total_size = sum(len(ds) for ds in datasets)
                dataset_weights = [len(ds) / total_size for ds in datasets]
        
        # Create combined dataset indices with weights
        combined_indices = []
        combined_weights = []
        
        for dataset_idx, (dataset, weight) in enumerate(zip(datasets, dataset_weights)):
            dataset_size = len(dataset)
            indices = list(range(dataset_size))
            
            # Add dataset offset to indices
            if dataset_idx > 0:
                offset = sum(len(ds) for ds in datasets[:dataset_idx])
                indices = [i + offset for i in indices]
            
            combined_indices.extend(indices)
            combined_weights.extend([weight] * dataset_size)
        
        # Create weighted sampler
        sampler = WeightedRandomSampler(
            weights=combined_weights,
            num_samples=len(combined_indices),
            replacement=True
        )
        
        # Create combined dataset
        combined_dataset = torch.utils.data.ConcatDataset(datasets)
        
        return DataLoader(
            combined_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=False,  # CPU training
            persistent_workers=True if num_workers > 0 else False
        )
    
    def _create_sequential_dataloader(
        self, 
        datasets: List[HybridDeepfakeDataset], 
        batch_size: int,
        num_workers: int
    ) -> DataLoader:
        """Create sequential dataloader (datasets one after another)"""
        
        combined_dataset = torch.utils.data.ConcatDataset(datasets)
        
        return DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False
        )
    
    def _create_random_dataloader(
        self, 
        datasets: List[HybridDeepfakeDataset], 
        batch_size: int,
        num_workers: int
    ) -> DataLoader:
        """Create random dataloader (simple concatenation)"""
        
        combined_dataset = torch.utils.data.ConcatDataset(datasets)
        
        return DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False
        )

class SamplingStrategy:
    """Main class for handling multi-dataset sampling strategies"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.sampler = MultiDatasetSampler(
            strategy=config.get('sampling_strategy', 'weighted'),
            balance_datasets=config.get('balance_datasets', True)
        )
        self.logger = logging.getLogger(__name__)
    
    def create_train_dataloader(self, dataset_paths: List[Tuple[str, Optional[int]]]) -> DataLoader:
        """Create training dataloader with multi-dataset sampling"""
        
        # Create individual datasets
        datasets = []
        dataset_weights = []
        
        for path, override_label in dataset_paths:
            try:
                from torchvision import transforms
                
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
                
                dataset = HybridDeepfakeDataset([(path, override_label)], transform=transform)
                datasets.append(dataset)
                
                # Calculate dataset weight based on config
                dataset_weight = self._calculate_dataset_weight(path)
                dataset_weights.append(dataset_weight)
                
                self.logger.info(f"📁 Loaded dataset: {path} ({len(dataset)} images, weight: {dataset_weight:.2f})")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to load dataset {path}: {e}")
                continue
        
        if not datasets:
            raise ValueError("No valid datasets found")
        
        # Create combined dataloader
        batch_size = self.config.get('batch_size', 2)
        num_workers = self.config.get('max_workers', 2)
        
        dataloader = self.sampler.create_combined_dataloader(
            datasets, batch_size, num_workers, dataset_weights
        )
        
        self.logger.info(f"✅ Created training dataloader: {len(datasets)} datasets, "
                        f"total samples: {sum(len(ds) for ds in datasets)}")
        
        return dataloader
    
    def create_val_dataloader(self, dataset_paths: List[Tuple[str, Optional[int]]]) -> DataLoader:
        """Create validation dataloader (simple concatenation)"""
        
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        datasets = []
        for path, override_label in dataset_paths:
            try:
                dataset = HybridDeepfakeDataset([(path, override_label)], transform=transform)
                datasets.append(dataset)
                self.logger.info(f"📁 Loaded validation dataset: {path} ({len(dataset)} images)")
            except Exception as e:
                self.logger.error(f"❌ Failed to load validation dataset {path}: {e}")
                continue
        
        if not datasets:
            raise ValueError("No valid validation datasets found")
        
        combined_dataset = torch.utils.data.ConcatDataset(datasets)
        
        batch_size = self.config.get('batch_size', 2)
        num_workers = self.config.get('max_workers', 2)
        
        return DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False
        )
    
    def _calculate_dataset_weight(self, dataset_path: str) -> float:
        """Calculate weight for a dataset based on configuration"""
        
        # Check if this path matches any configured dataset
        if 'datasets' in self.config:
            for name, dataset_config in self.config['datasets'].items():
                if dataset_config.get('enabled', False):
                    config_path = dataset_config['path']
                    if dataset_path.startswith(config_path):
                        return dataset_config.get('weight', 1.0)
        
        # Default weight
        return 1.0
