#!/usr/bin/env python3
"""
FINAL USAGE GUIDE - Multi-Dataset Deepfake Framework

This script provides the complete usage guide for the implemented framework.
"""

import os
import subprocess
import sys

def run_command(cmd, description):
    """Run a command and show results"""
    print(f"\n🔧 {description}")
    print(f"Command: {cmd}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=".")
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Failed to run: {e}")
        return False

def show_usage_menu():
    """Show interactive usage menu"""
    
    print("🎯 MULTI-DATASET DEEPFAKE FRAMEWORK - USAGE GUIDE")
    print("=" * 60)
    
    menu_options = [
        ("1", "Check Framework Status", "python framework_status.py"),
        ("2", "Add Real Kaggle Dataset", "python setup_kaggle_fixed.py"),
        ("3", "Add PGGAN Dataset", "python setup_pggan_stages.py"),
        ("4", "Run Sequential Training", "python sequential_trainer.py"),
        ("5", "Manage Datasets", "python manage_datasets.py"),
        ("6", "Test Image Classification", "python classify.py test_image.jpg"),
        ("7", "Start Web Interface", "python web-app.py"),
        ("8", "Test Video Inference", "python inference/video_inference.py"),
        ("9", "View Implementation Summary", "python IMPLEMENTATION_SUMMARY.py"),
        ("10", "Quick Start Guide", "python QUICK_START.py"),
        ("0", "Exit", "")
    ]
    
    while True:
        print("\n📋 USAGE MENU:")
        print("-" * 30)
        
        for option, description, _ in menu_options:
            print(f"{option}. {description}")
        
        choice = input(f"\nEnter choice (0-{len(menu_options)-1}): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
        
        # Find matching option
        selected = None
        for option, description, command in menu_options:
            if option == choice:
                selected = (description, command)
                break
        
        if selected:
            description, command = selected
            
            if command:  # Skip exit command
                print(f"\n🚀 Running: {description}")
                user_confirm = input("Press Enter to continue or 'c' to cancel: ")
                
                if user_confirm.lower() != 'c':
                    if choice in ["6", "7", "8"]:  # Inference commands need special handling
                        handle_inference_commands(choice)
                    else:
                        run_command(command, description)
        else:
            print("❌ Invalid choice. Please try again.")

def handle_inference_commands(choice):
    """Handle inference commands with proper setup"""
    
    if choice == "6":  # Image classification
        print("\n📸 To test image classification:")
        print("1. Place an image file in the current directory")
        print("2. Run: python classify.py your_image.jpg")
        print("3. The model will classify as REAL or FAKE with confidence")
        
        # Check if there are any test images
        test_images = []
        for file in os.listdir('.'):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(file)
        
        if test_images:
            print(f"\n📁 Found test images: {', '.join(test_images)}")
            use_existing = input("Use existing test image? (y/n): ")
            
            if use_existing.lower() == 'y':
                run_command(f"python classify.py {test_images[0]}", "Image Classification Test")
    
    elif choice == "7":  # Web interface
        print("\n🌐 Starting web interface...")
        print("The web app will open in your browser")
        print("You can upload images or videos for deepfake detection")
        run_command("python web-app.py", "Web Interface")
    
    elif choice == "8":  # Video inference
        print("\n🎥 To test video inference:")
        print("1. Place a video file in the current directory")
        print("2. Run: python inference/video_inference.py your_video.mp4")
        print("3. The model will analyze frames and classify")
        
        # Check for video files
        video_files = []
        for file in os.listdir('.'):
            if file.lower().endswith(('.mp4', '.mov', '.avi')):
                video_files.append(file)
        
        if video_files:
            print(f"\n📁 Found video files: {', '.join(video_files)}")
            use_existing = input("Use existing video? (y/n): ")
            
            if use_existing.lower() == 'y':
                run_command(f"python inference/video_inference.py {video_files[0]}", "Video Inference Test")

def show_quick_commands():
    """Show quick reference commands"""
    
    print("\n⚡ QUICK COMMANDS REFERENCE")
    print("=" * 50)
    
    commands = [
        ("Status Check", "python framework_status.py"),
        ("Add Kaggle", "python setup_kaggle_fixed.py"),
        ("Add PGGAN", "python setup_pggan_stages.py"),
        ("Train Model", "python sequential_trainer.py"),
        ("Manage Datasets", "python manage_datasets.py"),
        ("Test Image", "python classify.py image.jpg"),
        ("Web App", "python web-app.py"),
        ("Test Video", "python inference/video_inference.py"),
        ("Summary", "python IMPLEMENTATION_SUMMARY.py")
    ]
    
    for desc, cmd in commands:
        print(f"{desc:15} : {cmd}")

def show_next_steps():
    """Show recommended next steps"""
    
    print("\n📝 RECOMMENDED NEXT STEPS")
    print("=" * 50)
    
    steps = [
        "1. 🔄 Download real Kaggle dataset when internet is available",
        "2. 🎯 Add more PGGAN images from the cloned repository",
        "3. 📊 Experiment with different dataset weights in config_multi.yaml",
        "4. 🏃‍♂️ Run sequential training with real datasets",
        "5. 🧪 Test model performance on real deepfake samples",
        "6. 📈 Monitor training progress and adjust hyperparameters",
        "7. 🔧 Add FaceForensics++ dataset when ready",
        "8. 🔧 Add DFDC dataset when ready",
        "9. 🎨 Fine-tune model architecture if needed",
        "10. 📝 Document your training results and findings"
    ]
    
    for step in steps:
        print(f"  {step}")

def show_troubleshooting():
    """Show troubleshooting tips"""
    
    print("\n🔧 TROUBLESHOOTING TIPS")
    print("=" * 50)
    
    issues = [
        ("Memory Issues", "Reduce batch_size in config_multi.yaml to 1"),
        ("SSL Errors", "Run setup_kaggle_fixed.py with SSL bypass"),
        ("Dataset Not Found", "Check paths in config_multi.yaml"),
        ("Training Slow", "Increase gradient_accumulation_steps"),
        ("Model Not Loading", "Check models/ directory for .ckpt files"),
        ("Web App Not Working", "Install gradio: pip install gradio"),
        ("High Memory Usage", "Run: python -c \"import gc; gc.collect()\""),
        ("Dataset Validation Failed", "Check folder structure: train/real, train/fake")
    ]
    
    for issue, solution in issues:
        print(f"❌ {issue:20} : 💡 {solution}")

def main():
    """Main function"""
    
    print("🎯 MULTI-DATASET DEEPFAKE FRAMEWORK - FINAL USAGE GUIDE")
    print("=" * 70)
    
    # Show current status
    print("\n📊 CURRENT FRAMEWORK STATUS:")
    run_command("python framework_status.py", "Framework Status Check")
    
    # Show menu options
    print("\n📋 AVAILABLE OPTIONS:")
    print("1. Interactive Usage Menu")
    print("2. Quick Commands Reference")
    print("3. Recommended Next Steps")
    print("4. Troubleshooting Tips")
    print("5. Exit")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        show_usage_menu()
    elif choice == "2":
        show_quick_commands()
    elif choice == "3":
        show_next_steps()
    elif choice == "4":
        show_troubleshooting()
    elif choice == "5":
        print("👋 Framework ready for use!")
    else:
        print("Invalid choice. Showing quick commands...")
        show_quick_commands()

if __name__ == "__main__":
    main()
