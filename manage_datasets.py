import os
import yaml
from dataset_manager import dataset_manager

def add_kaggle_dataset(kaggle_path):
    """Add Kaggle dataset to the multi-dataset framework"""
    
    print(f"🔧 Adding Kaggle dataset from: {kaggle_path}")
    
    # Validate Kaggle dataset structure
    expected_structure = {
        'train': ['real', 'fake'],
        'valid': ['real', 'fake'], 
        'test': ['real', 'fake']
    }
    
    for split, subfolders in expected_structure.items():
        split_path = os.path.join(kaggle_path, split)
        if not os.path.exists(split_path):
            print(f"❌ Missing split folder: {split}")
            return False
        
        for subfolder in subfolders:
            subfolder_path = os.path.join(split_path, subfolder)
            if not os.path.exists(subfolder_path):
                print(f"❌ Missing subfolder: {split}/{subfolder}")
                return False
            
            # Count images
            images = [f for f in os.listdir(subfolder_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"📁 {split}/{subfolder}: {len(images)} images")
    
    # Update config
    config_path = 'config_multi.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update paths
    config['kaggle_dataset_path'] = os.path.abspath(kaggle_path)
    config['train_paths'] = [os.path.join(os.path.abspath(kaggle_path), 'train')]
    config['val_paths'] = [os.path.join(os.path.abspath(kaggle_path), 'valid')]
    config['test_paths'] = [os.path.join(os.path.abspath(kaggle_path), 'test')]
    
    # Update datasets section
    config['datasets']['kaggle_140k']['enabled'] = True
    config['datasets']['kaggle_140k']['path'] = os.path.abspath(kaggle_path)
    
    # Disable test dataset
    if 'test_dataset' in config['datasets']:
        config['datasets']['test_dataset']['enabled'] = False
    
    # Save config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✅ Kaggle dataset added to configuration")
    
    # Test the setup
    print("🧪 Testing Kaggle dataset integration...")
    if dataset_manager.validate_datasets():
        print("✅ All datasets validated successfully!")
        return True
    else:
        print("❌ Dataset validation failed")
        return False

def add_dataset(name, path, weight=1.0):
    """Add a new dataset to the framework"""
    
    print(f"🔧 Adding dataset '{name}' from: {path}")
    
    config_path = 'config_multi.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add dataset to configuration
    config['datasets'][name] = {
        'enabled': True,
        'weight': weight,
        'path': os.path.abspath(path),
        'subsets': {
            'train': 'train',
            'val': 'valid',
            'test': 'test'
        }
    }
    
    # Save config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Dataset '{name}' added successfully")

def list_datasets():
    """List all configured datasets"""
    
    config_path = 'config_multi.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("📋 Configured Datasets:")
    print("-" * 50)
    
    for name, dataset_config in config['datasets'].items():
        status = "✅ Enabled" if dataset_config['enabled'] else "❌ Disabled"
        weight = dataset_config['weight']
        path = dataset_config['path']
        print(f"{name:15} | {status:10} | Weight: {weight:.1f} | {path}")

def enable_dataset(name):
    """Enable a specific dataset"""
    
    config_path = 'config_multi.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if name in config['datasets']:
        config['datasets'][name]['enabled'] = True
        
        # Disable other datasets if you want exclusive training
        # for other_name in config['datasets']:
        #     if other_name != name:
        #         config['datasets'][other_name]['enabled'] = False
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ Dataset '{name}' enabled")
    else:
        print(f"❌ Dataset '{name}' not found")

def disable_dataset(name):
    """Disable a specific dataset"""
    
    config_path = 'config_multi.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if name in config['datasets']:
        config['datasets'][name]['enabled'] = False
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"❌ Dataset '{name}' disabled")
    else:
        print(f"❌ Dataset '{name}' not found")

if __name__ == "__main__":
    print("🔧 Dataset Management Tool")
    print("=" * 40)
    
    # Example usage:
    # add_kaggle_dataset("path/to/kaggle/dataset")
    # add_dataset("faceforensics_plus", "path/to/ff++", weight=0.5)
    # list_datasets()
    # enable_dataset("kaggle_140k")
    
    list_datasets()
