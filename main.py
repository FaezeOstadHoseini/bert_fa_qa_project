import subprocess
import sys

def run_pipeline():
    """Run the complete pipeline"""
    print("🚀 Starting QA Training Pipeline...")
    
    # Step 1: Data preprocessing
    print("\n📊 Step 1: Data Preprocessing")
    print("-" * 50)
    
    # Simply run without capturing - output will be displayed in real-time
    result = subprocess.run([sys.executable, "src/data_preprocessor.py"])
    
    if result.returncode != 0:
        print(f"❌ Data preprocessing failed")
        return
    
    # Step 2: Training
    print("\n🎯 Step 2: Model Training")
    print("-" * 50)
    
    result = subprocess.run([sys.executable, "src/train.py"])
    
    if result.returncode != 0:
        print(f"❌ Training failed")
        return
    
    print("✅ Pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()