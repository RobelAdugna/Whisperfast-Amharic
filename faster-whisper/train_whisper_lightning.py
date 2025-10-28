import lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperTokenizer
)
from datasets import load_from_disk
import evaluate
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.strategies import DeepSpeedStrategy
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Dict, Optional, Callable
import os

class GradioProgressCallback(L.Callback):
    """Custom callback to update Gradio progress during training"""
    def __init__(self, progress_callback: Optional[Callable] = None):
        super().__init__()
        self.progress_callback = progress_callback
    
    def on_train_epoch_end(self, trainer, pl_module):
        if self.progress_callback:
            progress = (trainer.current_epoch + 1) / trainer.max_epochs
            self.progress_callback(
                progress * 0.9,  # Reserve 0.9-1.0 for final steps
                desc=f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}"
            )

class WhisperLightningModule(L.LightningModule):
    def __init__(self, config: Dict):
        super().__init__()
        self.save_hyperparameters(config)
        
        # Load model
        self.model = WhisperForConditionalGeneration.from_pretrained(
            config["model_name"]
        )
        
        # Apply LoRA if requested
        if config.get("use_lora", False):
            peft_config = LoraConfig(
                r=config.get("lora_r", 16),
                lora_alpha=config.get("lora_alpha", 32),
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none"
            )
            self.model = get_peft_model(self.model, peft_config)
            print(f"LoRA applied. Trainable params: {self.model.print_trainable_parameters()}")
        
        # Load processor
        self.processor = WhisperProcessor.from_pretrained(
            config["model_name"],
            language=config.get("language", "am"),
            task="transcribe"
        )
        
        # Metrics
        self.wer_metric = evaluate.load("wer")
        
        # Config
        self.learning_rate = config.get("learning_rate", 1e-5)
        
    def forward(self, input_features, labels):
        return self.model(
            input_features=input_features,
            labels=labels
        )
    
    def training_step(self, batch, batch_idx):
        input_features = batch["input_features"]
        labels = batch["labels"]
        
        outputs = self(input_features, labels)
        loss = outputs.loss
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_features = batch["input_features"]
        labels = batch["labels"]
        
        outputs = self(input_features, labels)
        loss = outputs.loss
        
        # Generate predictions for WER
        with torch.no_grad():
            generated_ids = self.model.generate(input_features)
            predictions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Replace -100 in labels for decoding
            labels_for_decode = torch.where(labels != -100, labels, self.processor.tokenizer.pad_token_id)
            references = self.processor.batch_decode(labels_for_decode, skip_special_tokens=True)
            
            wer = self.wer_metric.compute(predictions=predictions, references=references)
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_wer", wer, on_epoch=True, prog_bar=True)
        
        return {"val_loss": loss, "val_wer": wer}
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

class AmharicDataModule(L.LightningDataModule):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.batch_size = config.get("batch_size", 16)
        self.num_workers = config.get("num_workers", 4)
        
        # Load processor
        self.processor = WhisperProcessor.from_pretrained(
            config["model_name"],
            language=config.get("language", "am"),
            task="transcribe"
        )
    
    def prepare_dataset(self, batch):
        # Load and process audio
        audio = batch["audio"]
        
        # Compute input features
        input_features = self.processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt"
        ).input_features[0]
        
        # Encode text
        batch["input_features"] = input_features
        
        # Tokenize labels
        labels = self.processor.tokenizer(batch["text"]).input_ids
        batch["labels"] = labels
        
        return batch
    
    def setup(self, stage=None):
        # Load dataset from disk
        dataset = load_from_disk(self.config["data_path"])
        
        # Process datasets
        self.train_dataset = dataset["train"].map(
            self.prepare_dataset,
            remove_columns=dataset["train"].column_names
        )
        
        self.val_dataset = dataset["validation"].map(
            self.prepare_dataset,
            remove_columns=dataset["validation"].column_names
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
    
    def collate_fn(self, batch):
        # Pad inputs and labels
        input_features = [item["input_features"] for item in batch]
        labels = [item["labels"] for item in batch]
        
        # Stack input features
        input_features = torch.stack(input_features)
        
        # Pad labels
        max_label_len = max(len(label) for label in labels)
        padded_labels = []
        for label in labels:
            padding = [-100] * (max_label_len - len(label))
            padded_labels.append(label + padding)
        
        labels = torch.tensor(padded_labels)
        
        return {
            "input_features": input_features,
            "labels": labels
        }

def start_training(
    config: Dict,
    resume_checkpoint: Optional[str] = None,
    progress_callback: Optional[Callable] = None
) -> Dict:
    """Start training with Lightning Trainer"""
    
    # Initialize data module
    data_module = AmharicDataModule(config)
    
    # Initialize model
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        model = WhisperLightningModule.load_from_checkpoint(
            resume_checkpoint,
            config=config
        )
    else:
        model = WhisperLightningModule(config)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.get("checkpoint_dir", "./checkpoints"),
        filename="whisper-{epoch:02d}-{val_loss:.2f}-{val_wer:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=config.get("save_top_k", 3),
        save_last=True,
        every_n_train_steps=None,
        every_n_epochs=1
    )
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=config.get("early_stopping_patience", 3),
        mode="min",
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    # Add Gradio progress callback if provided
    callbacks = [checkpoint_callback, early_stopping, lr_monitor]
    if progress_callback:
        callbacks.append(GradioProgressCallback(progress_callback))
    
    # Logger
    logger = TensorBoardLogger(
        save_dir="./logs",
        name="whisper_amharic"
    )
    
    # Strategy
    strategy = "auto"
    if config.get("use_deepspeed", False):
        strategy = DeepSpeedStrategy(
            stage=2,
            offload_optimizer=True,
            cpu_checkpointing=True
        )
    
    # Trainer
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        precision=config.get("precision", 16),
        strategy=strategy,
        max_epochs=config.get("num_epochs", 20),
        gradient_clip_val=config.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=config.get("accumulate_grad_batches", 4),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=config.get("val_check_interval", 0.25),
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Train
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=resume_checkpoint if resume_checkpoint else None
    )
    
    # Return results
    return {
        "best_model_path": checkpoint_callback.best_model_path,
        "best_model_score": checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score else None,
        "metrics": {
            "final_train_loss": trainer.callback_metrics.get("train_loss", 0),
            "final_val_loss": trainer.callback_metrics.get("val_loss", 0),
            "final_val_wer": trainer.callback_metrics.get("val_wer", 0)
        },
        "log": "Training completed successfully"
    }