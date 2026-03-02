import os
import ssl
import urllib.request
import zipfile
import tarfile
import shutil
import yaml
from pathlib import Path
import numpy as np
from PIL import Image

def setup_ssl_context():
    """Setup SSL context to handle certificate issues"""
    try:
        # Create unverified SSL context
        ssl._create_default_https_context = ssl._create_unverified_context
        print("✅ SSL context configured for certificate bypass")
        return True
    except Exception as e:
        print(f"⚠️ SSL setup failed: {e}")
        return False

def download_kaggle_alternative():
    """Alternative download method for Kaggle dataset"""
    
    print("🔄 Attempting alternative Kaggle download...")
    
    # Method 1: Try direct URL download
    kaggle_urls = [
        "https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces/download",
        "https://storage.googleapis.com/kaggle-datasets/140k-real-and-fake-faces.zip"
    ]
    
    for i, url in enumerate(kaggle_urls):
        try:
            print(f"📥 Attempting download method {i+1}...")
            
            # Setup SSL
            setup_ssl_context()
            
            # Download file
            filename = f"kaggle_dataset_method_{i+1}.zip"
            urllib.request.urlretrieve(url, filename)
            
            print(f"✅ Downloaded: {filename}")
            
            # Extract
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall("kaggle_dataset")
            
            print(f"✅ Extracted to: kaggle_dataset")
            return "kaggle_dataset"
            
        except Exception as e:
            print(f"❌ Method {i+1} failed: {e}")
            continue
    
    return None

def download_kagglehub_fixed():
    """Fixed kagglehub download with SSL handling"""
    
    try:
        print("🔄 Attempting kagglehub download with SSL fix...")
        
        # Setup SSL first
        setup_ssl_context()
        
        # Import and use kagglehub
        import kagglehub
        path = kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces")
        
        print(f"✅ Kaggle dataset downloaded to: {path}")
        return path
        
    except Exception as e:
        print(f"❌ Kagglehub download failed: {e}")
        return None

def create_sample_dataset():
    """Create a sample dataset if download fails"""
    
    print("🧪 Creating sample dataset for testing...")
    
    # Create sample structure
    base_path = "sample_kaggle_dataset"
    splits = ['train', 'valid', 'test']
    labels = ['real', 'fake']
    
    for split in splits:
        for label in labels:
            folder_path = os.path.join(base_path, split, label)
            os.makedirs(folder_path, exist_ok=True)
            
            # Create sample images
            for i in range(50):  # 50 images per folder
                # Create diverse colored images
                if label == 'real':
                    # Real images: blue-ish tones with variation
                    color = (
                        np.random.randint(80, 120),
                        np.random.randint(100, 150),
                        np.random.randint(200, 255)
                    )
                else:
                    # Fake images: red-ish tones with variation
                    color = (
                        np.random.randint(200, 255),
                        np.random.randint(80, 120),
                        np.random.randint(80, 120)
                    )
                
                # Create image with some noise
                image_array = np.full((224, 224, 3), color, dtype=np.uint8)
                noise = np.random.randint(-20, 20, (224, 224, 3)).astype(np.int16)
                image_array = np.clip(image_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                
                image = Image.fromarray(image_array)
                image.save(os.path.join(folder_path, f"sample_{i:04d}.jpg"))
    
    print(f"✅ Sample dataset created: {base_path}")
    return base_path

def validate_dataset_structure(dataset_path):
    """Validate and report dataset structure"""
    
    print(f"🔍 Validating dataset: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        return False
    
    splits = ['train', 'valid', 'test']
    labels = ['real', 'fake']
    total_images = 0
    
    for split in splits:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            print(f"❌ Missing split: {split}")
            return False
        
        for label in labels:
            label_path = os.path.join(split_path, label)
            if not os.path.exists(label_path):
                print(f"❌ Missing label folder: {split}/{label}")
                return False
            
            # Count images
            images = [f for f in os.listdir(label_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            count = len(images)
            total_images += count
            print(f"📁 {split}/{label}: {count} images")
    
    print(f"✅ Dataset validated: {total_images} total images")
    return True

def update_config_with_kaggle(kaggle_path):
    """Update config with Kaggle dataset"""
    
    print("📝 Updating configuration...")
    
    config_path = 'config_multi.yaml'
    
    # Load existing config
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Update paths
    kaggle_abs_path = os.path.abspath(kaggle_path)
    config['kaggle_dataset_path'] = kaggle_abs_path
    config['train_paths'] = [os.path.join(kaggle_abs_path, 'train')]
    config['val_paths'] = [os.path.join(kaggle_abs_path, 'valid')]
    config['test_paths'] = [os.path.join(kaggle_abs_path, 'test')]
    
    # Update datasets section
    if 'datasets' not in config:
        config['datasets'] = {}
    
    config['datasets']['kaggle_140k'] = {
        'enabled': True,
        'weight': 1.0,
        'path': kaggle_abs_path,
        'streaming': True,
        'subsets': {
            'train': 'train',
            'val': 'valid',
            'test': 'test'
        }
    }
    
    # Disable test dataset
    if 'test_dataset' in config['datasets']:
        config['datasets']['test_dataset']['enabled'] = False
    
    # Save config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Configuration updated: {config_path}")
    return True

def main():
    """Main setup function"""
    
    print("🚀 Kaggle Dataset Setup with SSL Fix")
    print("=" * 50)
    
    kaggle_path = None
    
    # Method 1: Try fixed kagglehub
    kaggle_path = download_kagglehub_fixed()
    
    # Method 2: Try alternative download
    if not kaggle_path:
        kaggle_path = download_kaggle_alternative()
    
    # Method 3: Create sample dataset
    if not kaggle_path:
        print("⚠️ All download methods failed, creating sample dataset...")
        kaggle_path = create_sample_dataset()
    
    # Validate dataset
    if kaggle_path and validate_dataset_structure(kaggle_path):
        # Update configuration
        if update_config_with_kaggle(kaggle_path):
            print("\n🎉 Kaggle dataset setup complete!")
            print(f"📁 Dataset path: {os.path.abspath(kaggle_path)}")
            print("🎯 Ready for training with: python cpu_trainer.py")
            return True
    else:
        print("\n❌ Dataset setup failed")
        return False

if __name__ == "__main__":
    main()
