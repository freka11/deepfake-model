import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import psutil
import gc
from typing import List, Tuple, Optional, Iterator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamingDataset:
    """Memory-efficient streaming dataset for large datasets"""
    
    def __init__(
        self, 
        dataset_path: str, 
        chunk_size: int = 1000,
        memory_limit_mb: int = 2000,
        transform=None
    ):
        self.dataset_path = dataset_path
        self.chunk_size = chunk_size
        self.memory_limit_mb = memory_limit_mb
        self.transform = transform
        self.memory_monitor = MemoryMonitor()
        
        # Discover all images
        self.image_paths = self._discover_images()
        self.total_samples = len(self.image_paths)
        
        # Initialize streaming
        self.current_chunk = 0
        self.total_chunks = (self.total_samples + chunk_size - 1) // chunk_size
        self.loaded_chunk = None
        self.loaded_indices = []
        
        logger.info(f"📊 Streaming dataset: {self.total_samples} images, "
                   f"{self.total_chunks} chunks of {chunk_size}")
    
    def _discover_images(self) -> List[str]:
        """Discover all images in dataset path"""
        image_paths = []
        
        if os.path.isdir(self.dataset_path):
            # Single directory
            for file in os.listdir(self.dataset_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(self.dataset_path, file))
        
        elif os.path.isfile(self.dataset_path):
            # Single file
            image_paths.append(self.dataset_path)
        
        return sorted(image_paths)
    
    def _load_chunk(self, chunk_idx: int) -> Tuple[List[torch.Tensor], List[int]]:
        """Load a specific chunk of images"""
        
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.total_samples)
        
        chunk_paths = self.image_paths[start_idx:end_idx]
        images = []
        labels = []
        
        # Determine label from path
        for path in chunk_paths:
            label = self._get_label_from_path(path)
            
            try:
                from PIL import Image
                image = Image.open(path).convert('RGB')
                
                if self.transform:
                    image = self.transform(image)
                
                images.append(image)
                labels.append(label)
                
            except Exception as e:
                logger.warning(f"⚠️ Could not load {path}: {e}")
                continue
        
        return images, labels
    
    def _get_label_from_path(self, path: str) -> int:
        """Extract label from file path"""
        path_lower = path.lower()
        if 'real' in path_lower:
            return 0
        elif 'fake' in path_lower:
            return 1
        else:
            # Default to fake for PGGAN
            return 1
    
    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item by index with chunk loading"""
        
        # Check if we need to load a new chunk
        chunk_idx = idx // self.chunk_size
        
        if self.loaded_chunk != chunk_idx:
            # Check memory before loading new chunk
            if not self.memory_monitor.check_memory_limit():
                self.memory_monitor.cleanup_memory()
            
            # Load new chunk
            self.loaded_chunk = chunk_idx
            images, labels = self._load_chunk(chunk_idx)
            
            # Store loaded data
            self.loaded_images = images
            self.loaded_labels = labels
            self.loaded_indices = list(
                range(chunk_idx * self.chunk_size, 
                       min((chunk_idx + 1) * self.chunk_size, self.total_samples))
            )
        
        # Get item from loaded chunk
        local_idx = idx - self.loaded_indices[0]
        return self.loaded_images[local_idx], self.loaded_labels[local_idx]

class MemoryMonitor:
    """Monitor and manage memory usage"""
    
    def __init__(self, memory_limit_gb: float = 6.0):
        self.memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
        self.process = psutil.Process()
    
    def get_memory_usage(self) -> dict:
        """Get current memory usage"""
        memory_info = self.process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return {
            'process_mb': memory_info.rss / 1024 / 1024,
            'process_bytes': memory_info.rss,
            'system_used_percent': system_memory.percent,
            'system_available_gb': system_memory.available / 1024 / 1024 / 1024,
            'limit_bytes': self.memory_limit_bytes
        }
    
    def check_memory_limit(self) -> bool:
        """Check if we're within memory limit"""
        usage = self.get_memory_usage()
        return usage['process_bytes'] < self.memory_limit_bytes
    
    def cleanup_memory(self):
        """Force garbage collection and cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("🧹 Memory cleanup completed")

class StreamingMultiDataset:
    """Combine multiple streaming datasets"""
    
    def __init__(
        self, 
        datasets: List[Tuple[str, Optional[int]]], 
        chunk_size: int = 1000,
        memory_limit_mb: int = 2000,
        transform=None,
        dataset_weights: Optional[List[float]] = None
    ):
        self.datasets = []
        self.dataset_weights = dataset_weights or [1.0] * len(datasets)
        
        # Create streaming datasets
        for dataset_path, override_label in datasets:
            if os.path.exists(dataset_path):
                streaming_ds = StreamingDataset(
                    dataset_path, chunk_size, memory_limit_mb, transform
                )
                self.datasets.append(streaming_ds)
                logger.info(f"✅ Added streaming dataset: {dataset_path}")
            else:
                logger.warning(f"⚠️ Dataset path not found: {dataset_path}")
        
        self.total_samples = sum(len(ds) for ds in self.datasets)
        
        # Create sampling indices
        self._create_sampling_indices()
    
    def _create_sampling_indices(self):
        """Create weighted sampling indices"""
        indices = []
        
        for dataset_idx, (dataset, weight) in enumerate(zip(self.datasets, self.dataset_weights)):
            dataset_size = len(dataset)
            dataset_indices = [(dataset_idx, i) for i in range(dataset_size)]
            
            # Repeat based on weight
            repeat_factor = max(1, int(weight * 10))
            indices.extend(dataset_indices * repeat_factor)
        
        # Shuffle indices
        np.random.shuffle(indices)
        self.sampling_indices = indices
        self.current_epoch = 0
    
    def __len__(self) -> int:
        return len(self.sampling_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item from sampling indices"""
        dataset_idx, sample_idx = self.sampling_indices[idx]
        return self.datasets[dataset_idx][sample_idx]
    
    def shuffle_epoch(self):
        """Shuffle for new epoch"""
        np.random.shuffle(self.sampling_indices)
        self.current_epoch += 1

class StreamingDataLoader:
    """Memory-efficient data loader for streaming datasets"""
    
    def __init__(
        self,
        dataset: StreamingMultiDataset,
        batch_size: int = 2,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Initialize for first epoch
        if shuffle:
            dataset.shuffle_epoch()
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over dataset in batches"""
        
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(indices)
        
        # Create batches
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            batch_images = []
            batch_labels = []
            
            for idx in batch_indices:
                image, label = self.dataset[idx]
                batch_images.append(image)
                batch_labels.append(label)
            
            # Stack into tensors
            if batch_images:
                images_tensor = torch.stack(batch_images)
                labels_tensor = torch.tensor(batch_labels)
                
                yield images_tensor, labels_tensor
    
    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

def create_streaming_dataloader(
    dataset_paths: List[Tuple[str, Optional[int]]],
    batch_size: int = 2,
    chunk_size: int = 1000,
    memory_limit_mb: int = 2000,
    transform=None,
    dataset_weights: Optional[List[float]] = None,
    shuffle: bool = True
) -> StreamingDataLoader:
    """Create streaming dataloader from multiple dataset paths"""
    
    # Create streaming multi-dataset
    streaming_dataset = StreamingMultiDataset(
        dataset_paths, chunk_size, memory_limit_mb, transform, dataset_weights
    )
    
    # Create dataloader
    dataloader = StreamingDataLoader(
        streaming_dataset, batch_size, shuffle, num_workers=0
    )
    
    logger.info(f"✅ Created streaming dataloader: {len(dataloader)} batches, "
               f"{len(streaming_dataset)} total samples")
    
    return dataloader

# Test function
def test_streaming():
    """Test streaming functionality"""
    
    logger.info("🧪 Testing streaming dataset...")
    
    # Create sample transform
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Test with sample dataset
    dataset_paths = [
        ("sample_kaggle_dataset/train/real", 0),
        ("sample_kaggle_dataset/train/fake", 1)
    ]
    
    try:
        # Create streaming dataloader
        dataloader = create_streaming_dataloader(
            dataset_paths, batch_size=2, chunk_size=50, transform=transform
        )
        
        # Test iteration
        for i, (images, labels) in enumerate(dataloader):
            if i >= 3:  # Test first 3 batches only
                break
            
            logger.info(f"Batch {i}: {images.shape}, {labels.shape}")
        
        logger.info("✅ Streaming test successful!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Streaming test failed: {e}")
        return False

if __name__ == "__main__":
    test_streaming()
