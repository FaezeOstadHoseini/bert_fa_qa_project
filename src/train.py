# train.py (optimized with checkpoint resume)
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import DefaultDataCollator
from datasets import load_from_disk
import torch
import numpy as np
import os
import yaml
import logging
import glob
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="../configs/training_config.yaml"):
    """Load training configuration"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"✅ Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"❌ Config file not found: {config_path}")
        raise

def setup_device():
    """Setup device (CPU/GPU)"""
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"✅ Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        logger.info("✅ Using CPU")
    return device

def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint directory"""
    if not os.path.exists(output_dir):
        return None
    
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoint_dirs:
        return None
    
    # Find checkpoint with highest number
    latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split("-")[-1]))
    logger.info(f"🔍 Found checkpoint: {latest_checkpoint}")
    return latest_checkpoint

def load_model_and_tokenizer(config, device):
    """Load model and tokenizer with checkpoint support"""
    best_model_dir = config['output']['best_model_dir']
    output_dir = config['output']['output_dir']
    
    # First, check if best model exists
    if os.path.exists(best_model_dir):
        logger.info("🏆 Loading best model from previous training...")
        tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
        model = AutoModelForQuestionAnswering.from_pretrained(best_model_dir)
    else:
        # Check for checkpoint to resume
        checkpoint_path = find_latest_checkpoint(output_dir)
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"🔄 Resuming from checkpoint: {checkpoint_path}")
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            model = AutoModelForQuestionAnswering.from_pretrained(checkpoint_path)
        else:
            # Load base model
            logger.info("🚀 Loading base model...")
            model_name = config['model']['pretrained_name']
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    model = model.to(device)
    return model, tokenizer

def load_preprocessed_data(config):
    """Load preprocessed data"""
    data_path = config['data']['processed_data_path']
    
    if not os.path.exists(data_path):
        logger.error(f"❌ Processed data not found at: {data_path}")
        logger.info("📝 Please run data_preprocessor.py first to prepare the data")
        raise FileNotFoundError(f"Processed data not found at {data_path}")
    
    logger.info(f"📥 Loading preprocessed data from: {data_path}")
    dataset = load_from_disk(data_path)
    
    logger.info(f"📊 Train samples: {len(dataset['train'])}")
    logger.info(f"📊 Validation samples: {len(dataset['validation'])}")
    
    return dataset

def prepare_train_features(examples, tokenizer, config):
    """Preprocess training features - optimized version"""
    # Clean questions
    examples["question"] = [q.lstrip() if q else "" for q in examples["question"]]
    pad_on_right = tokenizer.padding_side == "right"

    # Tokenize with optimized settings
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=config['model']['max_length'],
        stride=config['model']['stride'],
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",  # Changed to max_length for consistency
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        
        # Find CLS token index
        cls_index = 0
        for special_token_id in [tokenizer.cls_token_id, tokenizer.bos_token_id]:
            if special_token_id and special_token_id in input_ids:
                cls_index = input_ids.index(special_token_id)
                break

        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        # Handle unanswerable questions
        if (not answers or 'text' not in answers or not answers['text'] or len(answers['text']) == 0):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
            continue

        answer_text = answers['text'][0]
        answer_start = answers.get('answer_start', [0])[0] if answers.get('answer_start') else 0
        start_char = answer_start
        end_char = start_char + len(answer_text)

        # Find token positions
        token_start_index = 0
        while (token_start_index < len(sequence_ids) and 
               sequence_ids[token_start_index] != (1 if pad_on_right else 0)):
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while (token_end_index >= 0 and 
               sequence_ids[token_end_index] != (1 if pad_on_right else 0)):
            token_end_index -= 1

        # Check if answer is within the context
        if (token_start_index >= len(offsets) or token_end_index < 0 or
            not (offsets[token_start_index][0] <= start_char and 
                 offsets[token_end_index][1] >= end_char)):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Find start position
            while (token_start_index < len(offsets) and 
                   offsets[token_start_index][0] <= start_char):
                token_start_index += 1
            start_position = max(token_start_index - 1, 0)

            # Find end position
            token_end_index_copy = token_end_index
            while (token_end_index_copy >= 0 and 
                   offsets[token_end_index_copy][1] >= end_char):
                token_end_index_copy -= 1
            end_position = min(token_end_index_copy + 1, len(offsets) - 1)

            tokenized_examples["start_positions"].append(start_position)
            tokenized_examples["end_positions"].append(end_position)

    return tokenized_examples

def compute_metrics(eval_pred):
    """Compute evaluation metrics - optimized"""
    start_logits, end_logits = eval_pred.predictions
    start_positions, end_positions = eval_pred.label_ids

    # Use vectorized operations for better performance
    start_preds = np.argmax(start_logits, axis=1)
    end_preds = np.argmax(end_logits, axis=1)

    start_accuracy = np.mean(start_preds == start_positions)
    end_accuracy = np.mean(end_preds == end_positions)

    return {
        "start_accuracy": start_accuracy,
        "end_accuracy": end_accuracy,
        "avg_accuracy": (start_accuracy + end_accuracy) / 2
    }

def main():
    """Main training function with checkpoint support"""
    try:
        # Load configuration
        config = load_config()
        
        # Setup device
        device = setup_device()
        
        # Create output directories
        os.makedirs(config['output']['output_dir'], exist_ok=True)
        os.makedirs(config['output']['best_model_dir'], exist_ok=True)
        
        # Load model and tokenizer (with checkpoint support)
        model, tokenizer = load_model_and_tokenizer(config, device)
        
        # Load preprocessed data
        final_dataset = load_preprocessed_data(config)
        
        # Tokenize datasets with progress bar
        logger.info("🔄 Tokenizing data...")
        
        def tokenize_function(examples):
            return prepare_train_features(examples, tokenizer, config)
        
        # Use smaller batch size for tokenization to avoid memory issues
        tokenized_datasets = final_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=final_dataset["train"].column_names,
            batch_size=4,  # Reduced for memory efficiency
            desc="Tokenizing"
        )
        
        logger.info(f"✅ Tokenized - Train: {len(tokenized_datasets['train'])}, Validation: {len(tokenized_datasets['validation'])}")
        
        # Find checkpoint for resuming
        resume_from_checkpoint = find_latest_checkpoint(config['output']['output_dir'])
        
        # Optimized training arguments
        training_args = TrainingArguments(
            output_dir=config['output']['output_dir'],
            overwrite_output_dir=False,  # Important for checkpoint resuming
            
            # Checkpoint settings
            resume_from_checkpoint=resume_from_checkpoint,
            save_strategy="epoch",
            save_total_limit=2,  # Only keep 2 checkpoints to save space
            save_only_model=True,
            
            # Best model tracking
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Training parameters
            num_train_epochs=config['training']['num_epochs'],
            per_device_train_batch_size=config['training']['batch_size'],
            per_device_eval_batch_size=config['training']['eval_batch_size'],
            gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            warmup_steps=config['training']['warmup_steps'],
            
            # Evaluation
            evaluation_strategy="epoch",
            logging_strategy="steps",
            
            # Memory optimization
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=True,  # Memory optimization
            
            # Logging
            logging_dir=config['logging']['logging_dir'],
            logging_steps=config['logging']['logging_steps'],
            report_to="none",
            run_name="bert-fa-qa-training",
            
            # Prevent OOM errors
            dataloader_drop_last=True,
            remove_unused_columns=False,
        )
        
        # Setup trainer
        data_collator = DefaultDataCollator()
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        # Start training
        if resume_from_checkpoint:
            logger.info(f"🔄 Resuming training from checkpoint...")
        else:
            logger.info("🚀 Starting new training...")
            
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save best model
        trainer.save_model(config['output']['best_model_dir'])
        tokenizer.save_pretrained(config['output']['best_model_dir'])
        
        logger.info("✅ Training completed successfully!")
        logger.info(f"📊 Final training loss: {train_result.training_loss:.4f}")
        
        # Final evaluation
        eval_results = trainer.evaluate()
        logger.info("📋 Final evaluation results:")
        for key, value in eval_results.items():
            logger.info(f"   {key}: {value:.4f}")
            
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()