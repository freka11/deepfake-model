# Multi-Dataset Deepfake Training Framework

A comprehensive, CPU-optimized training framework for deepfake detection that supports multiple datasets with memory-efficient sampling strategies.

## 🚀 Quick Start

### 1. Setup Framework
```bash
python setup_manual.py
```

### 2. Train Model
```bash
python cpu_trainer.py
```

### 3. Add Kaggle Dataset (when downloaded)
```bash
python manage_datasets.py
# Then use: add_kaggle_dataset("path/to/kaggle/dataset")
```

## 📁 Framework Components

### Core Files
- **`cpu_trainer.py`** - Main training pipeline with multi-dataset support
- **`dataset_manager.py`** - Central dataset coordination and memory monitoring
- **`sampling_strategy.py`** - Multi-dataset sampling algorithms
- **`config_multi.yaml`** - Multi-dataset configuration
- **`manage_datasets.py`** - Dataset management utilities

### Setup Scripts
- **`setup_manual.py`** - Initial framework setup with test dataset
- **`setup_kaggle_first.py`** - Automatic Kaggle dataset download (requires internet)

## 🎯 Features

### Multi-Dataset Support
- **Weighted Sampling** - Balance datasets by size or custom weights
- **Sequential Training** - Train on datasets one after another
- **Memory Optimization** - CPU-friendly for systems with limited RAM
- **Progressive Loading** - Add datasets as you go

### CPU Optimizations
- **Small Batch Sizes** - 2-4 samples per batch
- **Gradient Accumulation** - Effective larger batches (16 samples)
- **Memory Monitoring** - Automatic RAM usage tracking
- **Efficient Data Loading** - Optimized for CPU processing

### Dataset Management
- **Easy Addition** - Add new datasets through config or script
- **Validation** - Automatic dataset structure checking
- **Flexible Configuration** - Enable/disable datasets per training run

## 📊 Supported Datasets

### Currently Configured
1. **Kaggle 140k Real and Fake Faces** - 140k images
2. **Test Dataset** - Small dataset for framework testing

### Ready to Add
1. **FaceForensics++** - High-quality video manipulation dataset
2. **DFDC** - Deepfake Detection Challenge dataset  
3. **Celeb-DF** - Celebrity deepfake dataset

## ⚙️ Configuration

### Training Parameters (config_multi.yaml)
```yaml
lr: 0.0001                    # Learning rate
batch_size: 2                 # Small batch for CPU
num_epochs: 10               # Training epochs
gradient_accumulation_steps: 8  # Effective batch size = 16
memory_limit_gb: 6           # RAM usage limit
```

### Dataset Configuration
```yaml
datasets:
  kaggle_140k:
    enabled: true
    weight: 1.0
    path: "/path/to/kaggle/dataset"
    subsets:
      train: "train"
      val: "valid"
      test: "test"
```

## 🛠️ Usage Examples

### Adding a New Dataset
```python
from manage_datasets import add_dataset
add_dataset("faceforensics_plus", "/path/to/ff++", weight=0.5)
```

### Training with Specific Datasets
```python
from manage_datasets import enable_dataset, disable_dataset
enable_dataset("kaggle_140k")
disable_dataset("test_dataset")
```

### Custom Training
```python
from cpu_trainer import MultiDatasetTrainer
trainer = MultiDatasetTrainer("config_multi.yaml")
trainer.train()
```

## 📈 Training Strategies

### 1. Weighted Multi-Dataset Training
- Samples from all enabled datasets
- Balances based on dataset weights
- Best for generalization

### 2. Sequential Training
- Train on one dataset at a time
- Transfer learning between datasets
- Good for specialized training

### 3. Progressive Training
- Start with small dataset
- Gradually add more data
- Curriculum learning approach

## 🔧 Memory Management

### Monitoring
- Real-time RAM usage tracking
- Automatic cleanup when memory is high
- Configurable memory limits

### Optimization
- Lazy dataset loading
- Efficient data pipelines
- Gradient accumulation for larger effective batches

## 📝 Model Loading

The framework automatically tries to load models in this order:
1. `models/multi_dataset_model.ckpt` (newest)
2. `models/best_model-v3.pt`
3. `models/best_model.pt`

## 🎯 Next Steps

1. **Download Kaggle Dataset** - Use the provided script or manual download
2. **Add More Datasets** - FaceForensics++, DFDC, Celeb-DF
3. **Fine-tune Hyperparameters** - Adjust learning rates, batch sizes
4. **Experiment with Sampling** - Try different sampling strategies

## 🚨 Important Notes

- **CPU Training**: Optimized for CPU, but can use GPU if available
- **Memory Limits**: Configured for 8GB RAM, adjust as needed
- **Dataset Size**: Start with smaller subsets for testing
- **Training Time**: CPU training takes longer, be patient

## 📞 Troubleshooting

### Memory Issues
- Reduce batch size to 1-2
- Lower gradient accumulation steps
- Enable memory monitoring

### Dataset Loading Issues
- Check dataset structure (train/val/test with real/fake subfolders)
- Verify image formats (.jpg, .jpeg, .png)
- Use dataset validation tools

### Training Issues
- Check model path in config
- Verify dataset paths are correct
- Monitor memory usage during training

## 🏆 Results

The framework successfully trains on multiple datasets with:
- ✅ Memory-efficient CPU training
- ✅ Multi-dataset sampling strategies  
- ✅ Automatic model checkpointing
- ✅ Real-time progress monitoring
- ✅ Easy dataset management

Ready for production use and further dataset additions!
