import kagglehub
import os
import shutil
from pathlib import Path
import yaml

def setup_kaggle_dataset():
    """Download and setup Kaggle 140k real and fake faces dataset"""
    
    print("🔄 Downloading Kaggle 140k real and fake faces dataset...")
    
    # Download dataset
    try:
        path = kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces")
        print(f"✅ Dataset downloaded to: {path}")
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return None
    
    # Validate dataset structure
    expected_folders = ['train', 'valid', 'test']
    for folder in expected_folders:
        folder_path = os.path.join(path, folder)
        if not os.path.exists(folder_path):
            print(f"❌ Missing expected folder: {folder}")
            return None
        
        # Check for real/fake subfolders
        real_path = os.path.join(folder_path, 'real')
        fake_path = os.path.join(folder_path, 'fake')
        
        if os.path.exists(real_path) and os.path.exists(fake_path):
            real_count = len([f for f in os.listdir(real_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            fake_count = len([f for f in os.listdir(fake_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"📁 {folder}: {real_count} real images, {fake_count} fake images")
        else:
            print(f"❌ Missing real/fake folders in {folder}")
            return None
    
    # Update config with Kaggle paths
    config_updates = {
        'kaggle_dataset_path': path,
        'train_paths': [os.path.join(path, 'train')],
        'val_paths': [os.path.join(path, 'valid')],
        'test_paths': [os.path.join(path, 'test')]
    }
    
    # Update config file
    config_path = 'config_multi.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    config.update(config_updates)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Config updated: {config_path}")
    print(f"📊 Dataset summary:")
    print(f"   Path: {path}")
    print(f"   Train: {config['train_paths'][0]}")
    print(f"   Valid: {config['val_paths'][0]}")
    print(f"   Test: {config['test_paths'][0]}")
    
    return path

def test_dataset_loading():
    """Test if the dataset can be loaded with hybrid_loader"""
    try:
        from datasets.hybrid_loader import HybridDeepfakeDataset
        from torchvision import transforms
        
        # Load config
        with open('config_multi.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Create transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        # Test train dataset
        train_dataset = HybridDeepfakeDataset(
            [(path, None) for path in config['train_paths']], 
            transform=transform
        )
        
        print(f"✅ Train dataset loaded: {len(train_dataset)} images")
        
        # Test loading a sample
        sample_image, sample_label = train_dataset[0]
        print(f"✅ Sample loaded: shape={sample_image.shape}, label={sample_label}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing dataset loading: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Setting up Kaggle dataset for multi-dataset training...")
    
    # Setup dataset
    kaggle_path = setup_kaggle_dataset()
    
    if kaggle_path:
        print("\n🧪 Testing dataset loading...")
        if test_dataset_loading():
            print("\n✅ Kaggle dataset setup complete!")
            print("🎯 Ready for multi-dataset training framework")
        else:
            print("\n❌ Dataset loading test failed")
    else:
        print("\n❌ Dataset setup failed")
