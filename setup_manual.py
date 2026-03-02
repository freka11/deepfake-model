import os
import shutil
import yaml
from pathlib import Path
from PIL import Image
import numpy as np

def create_test_dataset():
    """Create a small test dataset to validate the multi-dataset framework"""
    
    print("🧪 Creating test dataset for framework validation...")
    
    # Create test dataset structure
    base_path = "test_dataset"
    splits = ['train', 'valid', 'test']
    labels = ['real', 'fake']
    
    for split in splits:
        for label in labels:
            folder_path = os.path.join(base_path, split, label)
            os.makedirs(folder_path, exist_ok=True)
            
            # Create 10 dummy images per folder
            for i in range(10):
                # Create a simple colored image (real=blue, fake=red)
                color = (100, 150, 255) if label == 'real' else (255, 100, 100)
                image = Image.fromarray(np.full((224, 224, 3), color, dtype=np.uint8))
                image.save(os.path.join(folder_path, f"image_{i:03d}.jpg"))
    
    print(f"✅ Test dataset created at: {base_path}")
    
    # Update config with test dataset paths
    config_updates = {
        'test_dataset_path': os.path.abspath(base_path),
        'train_paths': [os.path.join(os.path.abspath(base_path), 'train')],
        'val_paths': [os.path.join(os.path.abspath(base_path), 'valid')],
        'test_paths': [os.path.join(os.path.abspath(base_path), 'test')]
    }
    
    # Update config file
    config_path = 'config_multi.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    config.update(config_updates)
    
    # Update datasets section
    if 'datasets' in config and 'kaggle_140k' in config['datasets']:
        # Disable kaggle for now, enable test dataset
        config['datasets']['kaggle_140k']['enabled'] = False
        
        # Add test dataset
        config['datasets']['test_dataset'] = {
            'enabled': True,
            'weight': 1.0,
            'path': os.path.abspath(base_path),
            'subsets': {
                'train': 'train',
                'val': 'valid',
                'test': 'test'
            }
        }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Config updated: {config_path}")
    return os.path.abspath(base_path)

def manual_kaggle_setup(kaggle_path):
    """Manually setup Kaggle dataset if downloaded"""
    
    print(f"🔧 Manual setup for Kaggle dataset at: {kaggle_path}")
    
    if not os.path.exists(kaggle_path):
        print(f"❌ Path does not exist: {kaggle_path}")
        return None
    
    # Validate structure
    expected_structure = {
        'train': ['real', 'fake'],
        'valid': ['real', 'fake'], 
        'test': ['real', 'fake']
    }
    
    for split, subfolders in expected_structure.items():
        split_path = os.path.join(kaggle_path, split)
        if not os.path.exists(split_path):
            print(f"❌ Missing split folder: {split}")
            return None
        
        for subfolder in subfolders:
            subfolder_path = os.path.join(split_path, subfolder)
            if not os.path.exists(subfolder_path):
                print(f"❌ Missing subfolder: {split}/{subfolder}")
                return None
            
            # Count images
            images = [f for f in os.listdir(subfolder_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"📁 {split}/{subfolder}: {len(images)} images")
    
    # Update config
    config_updates = {
        'kaggle_dataset_path': os.path.abspath(kaggle_path),
        'train_paths': [os.path.join(os.path.abspath(kaggle_path), 'train')],
        'val_paths': [os.path.join(os.path.abspath(kaggle_path), 'valid')],
        'test_paths': [os.path.join(os.path.abspath(kaggle_path), 'test')]
    }
    
    # Update config file
    config_path = 'config_multi.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config.update(config_updates)
    
    # Update datasets section
    if 'datasets' in config:
        config['datasets']['kaggle_140k']['enabled'] = True
        config['datasets']['kaggle_140k']['path'] = os.path.abspath(kaggle_path)
        
        # Disable test dataset if it exists
        if 'test_dataset' in config['datasets']:
            config['datasets']['test_dataset']['enabled'] = False
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Kaggle dataset configured in config")
    return os.path.abspath(kaggle_path)

def test_framework():
    """Test the multi-dataset framework"""
    
    print("🧪 Testing multi-dataset framework...")
    
    try:
        # Test dataset manager
        from dataset_manager import dataset_manager
        print("✅ Dataset manager imported successfully")
        
        # Test memory monitor
        memory_status = dataset_manager.memory_monitor.get_memory_usage()
        print(f"✅ Memory monitor working: {memory_status['process_mb']:.1f}MB used")
        
        # Test dataset validation
        if dataset_manager.validate_datasets():
            print("✅ Dataset validation passed")
        else:
            print("❌ Dataset validation failed")
            return False
        
        # Test sampling strategy
        from sampling_strategy import SamplingStrategy
        train_paths = dataset_manager.get_dataset_paths('train')
        val_paths = dataset_manager.get_dataset_paths('val')
        
        if train_paths and val_paths:
            print(f"✅ Dataset paths found: Train={len(train_paths)}, Val={len(val_paths)}")
            
            # Test creating dataloaders
            strategy = SamplingStrategy(dataset_manager.config)
            try:
                train_loader = strategy.create_train_dataloader(train_paths)
                val_loader = strategy.create_val_dataloader(val_paths)
                print(f"✅ Dataloaders created: Train={len(train_loader)}, Val={len(val_loader)}")
                
                # Test loading a batch
                train_batch = next(iter(train_loader))
                print(f"✅ Batch loaded: {train_batch[0].shape}, {train_batch[1].shape}")
                
            except Exception as e:
                print(f"❌ Dataloader creation failed: {e}")
                return False
        else:
            print("❌ No dataset paths found")
            return False
        
        print("✅ Framework test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Framework test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Multi-dataset framework setup...")
    
    # Option 1: Create test dataset
    test_path = create_test_dataset()
    
    # Option 2: Manual Kaggle setup (uncomment and provide path)
    # kaggle_path = input("Enter Kaggle dataset path: ").strip()
    # if kaggle_path:
    #     manual_kaggle_setup(kaggle_path)
    
    # Test the framework
    if test_framework():
        print("\n🎉 Framework setup complete!")
        print("🎯 Ready to run: python cpu_trainer.py")
    else:
        print("\n❌ Framework setup failed")
