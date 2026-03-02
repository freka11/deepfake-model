import os
import yaml
from dataset_manager import dataset_manager
import torch

def show_framework_status():
    """Display complete framework status"""
    
    print("🎯 Multi-Dataset Deepfake Framework Status")
    print("=" * 60)
    
    # Configuration Status
    print("\n📋 Configuration:")
    try:
        with open('config_multi.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"  ✅ Config loaded: config_multi.yaml")
        print(f"  📊 Batch size: {config.get('batch_size', 'N/A')}")
        print(f"  🎓 Learning rate: {config.get('lr', 'N/A')}")
        print(f"  🔄 Epochs: {config.get('num_epochs', 'N/A')}")
        print(f"  💾 Memory limit: {config.get('memory_limit_gb', 'N/A')}GB")
        
    except Exception as e:
        print(f"  ❌ Config error: {e}")
    
    # Dataset Status
    print("\n📁 Dataset Status:")
    try:
        enabled_datasets = dataset_manager.get_enabled_datasets()
        print(f"  ✅ Enabled datasets: {len(enabled_datasets)}")
        
        for dataset_name in enabled_datasets:
            if dataset_name in config['datasets']:
                dataset_config = config['datasets'][dataset_name]
                print(f"    📊 {dataset_name}: weight={dataset_config['weight']}")
        
        # Validation
        if dataset_manager.validate_datasets():
            print("  ✅ All datasets validated")
        else:
            print("  ❌ Dataset validation failed")
            
    except Exception as e:
        print(f"  ❌ Dataset error: {e}")
    
    # Memory Status
    print("\n💾 Memory Status:")
    try:
        memory_status = dataset_manager.memory_monitor.get_memory_usage()
        print(f"  📊 Process: {memory_status['process_mb']:.1f}MB")
        print(f"  📊 System: {memory_status['system_used_percent']:.1f}% used")
        print(f"  📊 Available: {memory_status['system_available_gb']:.1f}GB")
        
        if dataset_manager.memory_monitor.check_memory_limit():
            print("  ✅ Memory usage OK")
        else:
            print("  ⚠️ Memory usage high")
            
    except Exception as e:
        print(f"  ❌ Memory error: {e}")
    
    # Model Status
    print("\n🤖 Model Status:")
    model_paths = [
        "models/multi_dataset_model.ckpt",
        "models/multi_dataset_model-epoch=02-val_loss=0.38.ckpt",
        "models/best_model-v3.pt",
        "models/best_model.pt"
    ]
    
    models_found = []
    for model_path in model_paths:
        if os.path.exists(model_path):
            models_found.append(model_path)
            print(f"  ✅ Found: {model_path}")
    
    if not models_found:
        print("  ❌ No trained models found")
    
    # Training Readiness
    print("\n🚀 Training Readiness:")
    try:
        from sampling_strategy import SamplingStrategy
        from cpu_trainer import MultiDatasetTrainer
        
        # Check if we can create dataloaders
        train_paths = dataset_manager.get_dataset_paths('train')
        val_paths = dataset_manager.get_dataset_paths('val')
        
        if train_paths and val_paths:
            print("  ✅ Dataset paths available")
            
            # Test dataloader creation (lightweight test)
            strategy = SamplingStrategy(config)
            try:
                # Just test the first few samples
                train_loader = strategy.create_train_dataloader(train_paths[:1])
                print(f"  ✅ Training dataloader: {len(train_loader)} batches")
            except Exception as e:
                print(f"  ⚠️ Dataloader test: {e}")
        else:
            print("  ❌ Missing dataset paths")
            
        print("  ✅ Framework ready for training")
        
    except Exception as e:
        print(f"  ❌ Training setup error: {e}")
    
    # Next Steps
    print("\n📝 Next Steps:")
    if enabled_datasets and 'test_dataset' in enabled_datasets:
        print("  🔧 Add real datasets using manage_datasets.py")
        print("    python manage_datasets.py")
        print("    # Then: add_kaggle_dataset('path/to/kaggle/dataset')")
    
    if models_found:
        print("  🎯 Test inference with:")
        print("    python classify.py path/to/image.jpg")
        print("    python web-app.py  # for web interface")
    
    print("  🏃‍♂️ Start training:")
    print("    python cpu_trainer.py")
    
    print("\n" + "=" * 60)
    print("🎉 Multi-Dataset Framework Setup Complete!")

if __name__ == "__main__":
    show_framework_status()
