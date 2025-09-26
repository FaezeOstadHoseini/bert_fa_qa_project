import subprocess
import sys
import os

def run_pipeline():
    """Run the complete pipeline"""
    print("🚀 Starting QA Training Pipeline...")
    
    # Step 1: Data preprocessing
    print("\n📊 Step 1: Data Preprocessing")
    result = subprocess.run([sys.executable, "src/data_preprocessor.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"❌ Data preprocessing failed: {result.stderr}")
        return
    
    # Step 2: Training
    print("\n🎯 Step 2: Model Training")
    result = subprocess.run([sys.executable, "src/train.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"❌ Training failed: {result.stderr}")
        return
    
    print("✅ Pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()