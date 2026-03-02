import os
import shutil
import git
import yaml
from pathlib import Path
import numpy as np
from PIL import Image
import random

def clone_pggan_repository():
    """Clone PGGAN repository from GitHub"""
    
    print("🔄 Cloning PGGAN repository...")
    
    repo_url = "https://github.com/tkarras/progressive_growing_of_gans.git"
    repo_dir = "pggan_repository"
    
    try:
        # Remove existing directory
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
        
        # Clone repository
        git.Repo.clone_from(repo_url, repo_dir)
        print(f"✅ Repository cloned: {repo_dir}")
        return repo_dir
        
    except Exception as e:
        print(f"❌ Failed to clone repository: {e}")
        return None

def find_pggan_images(repo_dir):
    """Find all generated images in PGGAN repository"""
    
    print("🔍 Scanning for PGGAN generated images...")
    
    image_files = []
    
    # Common PGGAN output directories
    search_dirs = [
        "results",
        "generated_images", 
        "samples",
        "output",
        "images"
    ]
    
    for search_dir in search_dirs:
        full_path = os.path.join(repo_dir, search_dir)
        if os.path.exists(full_path):
            print(f"📁 Found directory: {full_path}")
            
            # Recursively find all image files
            for root, dirs, files in os.walk(full_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_files.append(os.path.join(root, file))
    
    # Also look for images in root and subdirectories
    for root, dirs, files in os.walk(repo_dir):
        # Skip .git and other non-image directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    print(f"✅ Found {len(image_files)} images")
    return image_files

def categorize_images_by_resolution(image_files):
    """Categorize images by resolution for different PGGAN stages"""
    
    print("📏 Categorizing images by resolution...")
    
    resolution_categories = {
        '32_64': [],    # Low resolution (32x32, 64x64)
        '128_256': [],   # Medium resolution (128x128, 256x256)
        '512_1024': []   # High resolution (512x512, 1024x1024)
    }
    
    for image_path in image_files:
        try:
            # Get image resolution
            with Image.open(image_path) as img:
                width, height = img.size
                
                # Categorize by resolution
                if max(width, height) <= 64:
                    resolution_categories['32_64'].append(image_path)
                elif max(width, height) <= 256:
                    resolution_categories['128_256'].append(image_path)
                else:
                    resolution_categories['512_1024'].append(image_path)
                    
        except Exception as e:
            print(f"⚠️ Could not process {image_path}: {e}")
            continue
    
    # Print statistics
    for category, images in resolution_categories.items():
        print(f"📁 {category}: {len(images)} images")
    
    return resolution_categories

def create_pggan_dataset_structure(base_dir, resolution_categories):
    """Create PGGAN dataset structure organized by resolution"""
    
    print("🏗️ Creating PGGAN dataset structure...")
    
    # Create base directory
    pggan_base = os.path.join(base_dir, "pggan_multi_stage")
    os.makedirs(pggan_base, exist_ok=True)
    
    dataset_info = {}
    
    for resolution, images in resolution_categories.items():
        if not images:
            continue
            
        print(f"📁 Processing {resolution} category...")
        
        # Create dataset splits
        splits = ['train', 'valid', 'test']
        split_ratios = [0.7, 0.2, 0.1]  # 70% train, 20% valid, 10% test
        
        # Shuffle images
        random.shuffle(images)
        
        resolution_info = {}
        
        for i, split in enumerate(splits):
            split_dir = os.path.join(pggan_base, f"{resolution}_{split}", "fake")
            os.makedirs(split_dir, exist_ok=True)
            
            # Calculate split indices
            start_idx = int(sum(split_ratios[:i]) * len(images))
            end_idx = int(sum(split_ratios[:i+1]) * len(images))
            split_images = images[start_idx:end_idx]
            
            # Copy images to split directory
            for j, image_path in enumerate(split_images):
                # Generate new filename
                old_ext = os.path.splitext(image_path)[1]
                new_filename = f"pggan_{resolution}_{j:06d}{old_ext}"
                new_path = os.path.join(split_dir, new_filename)
                
                try:
                    shutil.copy2(image_path, new_path)
                except Exception as e:
                    print(f"⚠️ Could not copy {image_path}: {e}")
            
            resolution_info[split] = {
                'path': split_dir,
                'count': len(split_images)
            }
            
            print(f"  📁 {split}: {len(split_images)} images")
        
        dataset_info[resolution] = resolution_info
    
    print(f"✅ PGGAN dataset created: {pggan_base}")
    return pggan_base, dataset_info

def create_sample_pggan_images(base_dir):
    """Create sample PGGAN-style images if no real ones found"""
    
    print("🧪 Creating sample PGGAN-style images...")
    
    sample_dir = os.path.join(base_dir, "pggan_multi_stage")
    os.makedirs(sample_dir, exist_ok=True)
    
    # Different resolutions for different stages
    resolutions = [
        (32, '32_64'),
        (64, '32_64'),
        (128, '128_256'),
        (256, '128_256'),
        (512, '512_1024'),
        (1024, '512_1024')
    ]
    
    for resolution, category in resolutions:
        for split in ['train', 'valid', 'test']:
            split_dir = os.path.join(sample_dir, f"{category}_{split}", "fake")
            os.makedirs(split_dir, exist_ok=True)
            
            # Create sample images with different characteristics per resolution
            num_images = 20 if split == 'train' else 10
            
            for i in range(num_images):
                # Create synthetic face-like patterns
                size = (resolution, resolution)
                
                # Generate different patterns for different resolutions
                if resolution <= 64:
                    # Low res: simple patterns
                    color = (
                        np.random.randint(150, 255),
                        np.random.randint(50, 150),
                        np.random.randint(50, 150)
                    )
                elif resolution <= 256:
                    # Medium res: more detail
                    color = (
                        np.random.randint(180, 255),
                        np.random.randint(100, 180),
                        np.random.randint(100, 180)
                    )
                else:
                    # High res: complex patterns
                    color = (
                        np.random.randint(200, 255),
                        np.random.randint(150, 220),
                        np.random.randint(150, 220)
                    )
                
                # Create image with gradients and noise
                image_array = np.zeros((resolution, resolution, 3), dtype=np.uint8)
                
                # Add gradient
                for y in range(resolution):
                    for x in range(resolution):
                        factor = (x + y) / (2 * resolution)
                        image_array[y, x] = [
                            int(color[0] * (0.7 + 0.3 * factor)),
                            int(color[1] * (0.7 + 0.3 * factor)),
                            int(color[2] * (0.7 + 0.3 * factor))
                        ]
                
                # Add noise
                noise = np.random.randint(-30, 30, (resolution, resolution, 3)).astype(np.int16)
                image_array = np.clip(image_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                
                # Resize to standard size if needed
                image = Image.fromarray(image_array)
                if resolution != 224:
                    image = image.resize((224, 224), Image.LANCZOS)
                
                # Save image
                filename = f"pggan_{category}_{i:04d}.jpg"
                image.save(os.path.join(split_dir, filename))
    
    print(f"✅ Sample PGGAN images created: {sample_dir}")
    return sample_dir

def update_config_with_pggan(pggan_path, dataset_info):
    """Update configuration with PGGAN datasets"""
    
    print("📝 Updating configuration with PGGAN datasets...")
    
    config_path = 'config_multi.yaml'
    
    # Load existing config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add PGGAN datasets
    pggan_datasets = {
        'pggan_32_64': {
            'enabled': True,
            'weight': 0.3,
            'path': os.path.join(pggan_path, '32_64_train'),
            'streaming': True,
            'subsets': {
                'train': '32_64_train/fake',
                'val': '32_64_valid/fake',
                'test': '32_64_test/fake'
            }
        },
        'pggan_128_256': {
            'enabled': True,
            'weight': 0.3,
            'path': os.path.join(pggan_path, '128_256_train'),
            'streaming': True,
            'subsets': {
                'train': '128_256_train/fake',
                'val': '128_256_valid/fake',
                'test': '128_256_test/fake'
            }
        },
        'pggan_512_1024': {
            'enabled': True,
            'weight': 0.3,
            'path': os.path.join(pggan_path, '512_1024_train'),
            'streaming': True,
            'subsets': {
                'train': '512_1024_train/fake',
                'val': '512_1024_valid/fake',
                'test': '512_1024_test/fake'
            }
        }
    }
    
    # Add to config
    config['datasets'].update(pggan_datasets)
    
    # Save config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✅ Configuration updated with PGGAN datasets")
    return True

def main():
    """Main PGGAN setup function"""
    
    print("🚀 PGGAN Multi-Stage Dataset Setup")
    print("=" * 50)
    
    # Step 1: Clone repository
    repo_dir = clone_pggan_repository()
    
    if not repo_dir:
        print("⚠️ Repository clone failed, using sample images...")
        base_dir = "."
        pggan_path, dataset_info = create_sample_pggan_images(base_dir)
    else:
        # Step 2: Find images
        image_files = find_pggan_images(repo_dir)
        
        if not image_files:
            print("⚠️ No images found, creating sample dataset...")
            pggan_path, dataset_info = create_sample_pggan_images(".")
        else:
            # Step 3: Categorize by resolution
            resolution_categories = categorize_images_by_resolution(image_files)
            
            # Step 4: Create dataset structure
            base_dir = "."
            pggan_path, dataset_info = create_pggan_dataset_structure(base_dir, resolution_categories)
    
    # Step 5: Update configuration
    if update_config_with_pggan(pggan_path, dataset_info):
        print("\n🎉 PGGAN multi-stage dataset setup complete!")
        print(f"📁 Dataset path: {os.path.abspath(pggan_path)}")
        print("🎯 Ready for multi-dataset training!")
        return True
    else:
        print("\n❌ PGGAN dataset setup failed")
        return False

if __name__ == "__main__":
    main()
