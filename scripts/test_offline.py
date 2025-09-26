"""
Test script that works without downloading models/datasets
"""

import torch
import os
import sys

def test_environment():
    """Test if basic environment is working"""
    print("=== Environment Test ===")
    
    # Test PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    # Test imports
    try:
        from transformers import __version__ as transformers_version
        print(f"Transformers version: {transformers_version}")
    except ImportError as e:
        print(f"Transformers import error: {e}")
    
    try:
        from datasets import __version__ as datasets_version
        print(f"Datasets version: {datasets_version}")
    except ImportError as e:
        print(f"Datasets import error: {e}")
    
    print("✓ Environment test completed")

def test_config():
    """Test configuration loading"""
    print("\n=== Configuration Test ===")
    
    try:
        import yaml
        with open("configs/training_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        print("✓ Config file loaded successfully")
        print(f"  Model: {config['model']['pretrained_name']}")
        print(f"  Batch size: {config['training']['batch_size']}")
    except Exception as e:
        print(f"✗ Config test failed: {e}")

def test_directory_structure():
    """Test directory structure"""
    print("\n=== Directory Structure Test ===")
    
    required_dirs = [
        "data/raw", "data/processed", "data/splits",
        "models/pretrained", "models/checkpoints", "models/final",
        "src", "configs", "scripts", "logs"
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} - missing")

if __name__ == "__main__":
    test_environment()
    test_config()
    test_directory_structure()
    
    print("\n=== Next Steps ===")
    print("1. Create virtual environment: python -m venv venv")
    print("2. Activate venv: source venv/bin/activate")
    print("3. Install requirements: pip install -r requirements.txt")
    print("4. Run offline test: python scripts/test_offline.py")
    print("5. When ready to download, run: ./scripts/train.sh")
