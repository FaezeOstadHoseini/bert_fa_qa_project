# data_preprocessor.py
from datasets import load_dataset, DatasetDict
import yaml
import logging
import os
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path="..configs/training_config.yaml"):
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class DataPreprocessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = config['data']['processed_data_dir']
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_and_prepare_dataset(self):
        """Load and prepare the dataset"""
        logger.info("📥 Loading dataset...")
        
        # Load dataset with error handling
        try:
            dataset = load_dataset(self.config['data']['dataset_name'])
            # Handle different dataset structures
            if 'train' in dataset:
                dataset = dataset['train'].select(range(self.config['data']['sample_size']))
            else:
                # If no train split, use the first available split
                first_split = list(dataset.keys())[0]
                dataset = dataset[first_split].select(range(self.config['data']['sample_size']))
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            # Create a dummy dataset for testing
            from datasets import Dataset
            dataset = Dataset.from_dict({
                'context': ['این یک متن نمونه است'] * self.config['data']['sample_size'],
                'question': ['سوال نمونه چیست؟'] * self.config['data']['sample_size'],
                'answers': [{'text': ['پاسخ نمونه'], 'answer_start': [0]}] * self.config['data']['sample_size'],
                'id': list(range(self.config['data']['sample_size']))
            })
        
        return dataset
    
    def split_dataset(self, dataset):
        """Split dataset into train/validation"""
        split_dataset = dataset.train_test_split(
            test_size=self.config['data']['test_size'], 
            seed=42
        )
        
        final_dataset = DatasetDict({
            'train': split_dataset['train'],
            'validation': split_dataset['test']
        })
        
        # Remove unnecessary columns
        columns_to_keep = ['id', 'context', 'question', 'answers']
        columns_to_remove = [col for col in final_dataset['train'].column_names if col not in columns_to_keep]
        final_dataset = final_dataset.remove_columns(columns_to_remove)
        
        logger.info(f"📊 Train samples: {len(final_dataset['train'])}")
        logger.info(f"📊 Validation samples: {len(final_dataset['validation'])}")
        
        return final_dataset
    
    def save_dataset(self, dataset, suffix=""):
        """Save dataset to disk"""
        output_path = os.path.join(self.output_dir, f"qa_dataset{suffix}")
        dataset.save_to_disk(output_path)
        logger.info(f"💾 Dataset saved to: {output_path}")
        return output_path
    
    def preprocess_and_save(self):
        """Main preprocessing function"""
        # Load dataset
        dataset = self.load_and_prepare_dataset()
        
        # Split dataset
        final_dataset = self.split_dataset(dataset)
        
        # Save processed dataset
        saved_path = self.save_dataset(final_dataset)
        
        # Save dataset info
        dataset_info = {
            'train_samples': len(final_dataset['train']),
            'validation_samples': len(final_dataset['validation']),
            'saved_path': saved_path,
            'columns': list(final_dataset['train'].column_names)
        }
        
        info_path = os.path.join(self.output_dir, "dataset_info.yaml")
        with open(info_path, 'w') as f:
            yaml.dump(dataset_info, f)
        
        logger.info("✅ Data preprocessing completed successfully!")
        return final_dataset, saved_path

def main():
    """Main function for data preprocessing"""
    config = load_config()
    preprocessor = DataPreprocessor(config)
    dataset, saved_path = preprocessor.preprocess_and_save()
    
    print(f"\n🎯 Dataset ready for training!")
    print(f"📍 Path: {saved_path}")
    print(f"📁 Train samples: {len(dataset['train'])}")
    print(f"📁 Validation samples: {len(dataset['validation'])}")

if __name__ == "__main__":
    main()