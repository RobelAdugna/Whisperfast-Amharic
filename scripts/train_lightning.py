#!/usr/bin/env python3
"""
Lightning AI training script for Amharic Whisper fine-tuning.
"""

import argparse
import yaml
import os
from pathlib import Path
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Audio
from dataclasses import dataclass
from typing import Dict, List, Optional
import evaluate


class WhisperDataModule(L.LightningDataModule):
    def __init__(self, config: dict, processor):
        super().__init__()
        self.config = config
        self.processor = processor
        
    def setup(self, stage: Optional[str] = None):
        # Load custom dataset from JSON metadata
        from datasets import Dataset
        import json
        
        def load_custom_dataset(split_dir):
            metadata_path = Path(split_dir) / 'metadata.json'
            with open(metadata_path, 'r') as f:
                data = json.load(f)
            
            # Convert to Dataset format
            dataset = Dataset.from_dict({
                'audio': [str(Path(split_dir) / item['audio']) for item in data],
                'text': [item['text'] for item in data]
            })
            
            # Cast audio column
            dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))
            return dataset
        
        self.train_dataset = load_custom_dataset(self.config['data']['train_dir'])
        self.val_dataset = load_custom_dataset(self.config['data']['val_dir'])
        
        # Preprocess datasets
        self.train_dataset = self.train_dataset.map(
            self._prepare_dataset,
            remove_columns=self.train_dataset.column_names
        )
        self.val_dataset = self.val_dataset.map(
            self._prepare_dataset,
            remove_columns=self.val_dataset.column_names
        )
    
    def _prepare_dataset(self, batch):
        # Load and process audio
        audio = batch['audio']
        
        # Process audio to input features
        input_features = self.processor(
            audio['array'],
            sampling_rate=audio['sampling_rate'],
            return_tensors="pt"
        ).input_features[0]
        
        # Tokenize text
        labels = self.processor.tokenizer(batch['text']).input_ids
        
        return {
            'input_features': input_features,
            'labels': labels
        }
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=4,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=4,
            collate_fn=self._collate_fn
        )
    
    @staticmethod
    def _collate_fn(batch):
        # Pad inputs and labels
        input_features = [item['input_features'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        # Pad to max length in batch
        input_features = torch.stack(input_features)
        
        # Pad labels
        max_label_len = max(len(l) for l in labels)
        labels_padded = []
        for l in labels:
            padded = l + [-100] * (max_label_len - len(l))
            labels_padded.append(padded)
        
        return {
            'input_features': input_features,
            'labels': torch.tensor(labels_padded)
        }


class WhisperLightningModule(L.LightningModule):
    def __init__(self, config: dict, processor):
        super().__init__()
        self.config = config
        self.processor = processor
        
        # Load model
        self.model = WhisperForConditionalGeneration.from_pretrained(
            config['model']['base_model']
        )
        
        # Apply LoRA if enabled
        if config['lora']['enabled']:
            self.model = prepare_model_for_kbit_training(self.model)
            
            lora_config = LoraConfig(
                r=config['lora']['r'],
                lora_alpha=config['lora']['lora_alpha'],
                target_modules=config['lora']['target_modules'],
                lora_dropout=config['lora']['lora_dropout'],
                bias="none"
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        # Set language and task
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=config['model']['language'],
            task=config['model']['task']
        )
        
        # Metrics
        self.wer_metric = evaluate.load("wer")
        
        self.save_hyperparameters()
    
    def forward(self, input_features, labels):
        return self.model(input_features=input_features, labels=labels)
    
    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        
        # Calculate WER
        predictions = torch.argmax(outputs.logits, dim=-1)
        decoded_preds = self.processor.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.processor.batch_decode(batch['labels'], skip_special_tokens=True)
        
        wer = self.wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_wer', wer, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['optimization']['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=self.config['training']['warmup_steps']
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }


def main(config_path: str):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize processor
    processor = WhisperProcessor.from_pretrained(config['model']['base_model'])
    
    # Create data module
    data_module = WhisperDataModule(config, processor)
    
    # Create model
    model = WhisperLightningModule(config, processor)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=config['training']['output_dir'],
            filename='whisper-amharic-{epoch:02d}-{val_loss:.2f}',
            save_top_k=config['checkpoint']['save_total_limit'],
            monitor='val_loss',
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            mode='min'
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Logger
    logger = None
    if config['monitoring']['use_wandb']:
        logger = WandbLogger(
            project=config['monitoring']['project_name'],
            name='whisper-amharic-finetune'
        )
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=config['training']['num_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=config['lightning']['num_gpus'] if config['lightning']['enabled'] else 1,
        strategy=config['lightning']['strategy'] if config['lightning']['enabled'] else 'auto',
        precision=config['lightning']['precision'] if config['lightning']['enabled'] else '32',
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=config['optimization']['max_grad_norm'],
        accumulate_grad_batches=config['training']['gradient_accumulation_steps'],
        log_every_n_steps=config['evaluation']['logging_steps'],
        val_check_interval=config['evaluation']['eval_steps']
    )
    
    # Train
    trainer.fit(model, data_module)
    
    # Save final model
    output_dir = Path(config['training']['output_dir']) / 'final'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if config['lora']['enabled']:
        model.model.save_pretrained(output_dir)
    else:
        model.model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    print(f"Training complete! Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/amharic_config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    main(args.config)
