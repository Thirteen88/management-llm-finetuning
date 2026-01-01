#!/usr/bin/env python3
"""
Fine-tune a language model on management conversation data.

This script handles:
- Loading a base model (Mistral, Llama, etc.)
- Setting up LoRA/QLoRA for parameter-efficient fine-tuning
- Training with the management dataset
- Saving checkpoints and metrics
"""

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from transformers import TrainerCallback
import yaml


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file.

    Args:
        config_path: Path to config YAML file

    Returns:
        Configuration dictionary
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


@dataclass
class TrainingConfig:
    """Training configuration hyperparameters."""

    # Model
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    load_in_4bit: bool = True
    use_flash_attention: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    max_grad_norm: float = 1.0

    # Data
    train_path: str = "data/processed/train.parquet"
    validation_path: str = "data/processed/validation.parquet"
    max_seq_length: int = 2048

    # Output
    output_dir: str = "models/checkpoints"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3

    # Hardware
    bf16: bool = True
    fp16: bool = False


class MetricsCallback(TrainerCallback):
    """Callback to log training metrics."""

    def __init__(self):
        self.metrics_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics."""
        if logs:
            self.metrics_history.append(logs)
            if "loss" in logs:
                print(f"Step {state.global_step}: loss={logs['loss']:.4f}, lr={logs.get('learning_rate', 0):.2e}")


def load_model_and_tokenizer(config: TrainingConfig):
    """Load the base model and tokenizer.

    Args:
        config: Training configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {config.model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix for fp16 training

    # Quantization config
    if config.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_in_8bit = False
    else:
        bnb_config = None
        load_in_8bit = False

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        load_in_8bit=load_in_8bit,
        device_map="auto",
        trust_remote_code=True,
        use_flash_attention_2=config.use_flash_attention,
    )

    model.config.use_cache = False  # Disable for training

    return model, tokenizer


def setup_lora(model, config: TrainingConfig):
    """Set up LoRA adapters for the model.

    Args:
        model: Base model
        config: Training configuration

    Returns:
        Model with LoRA adapters
    """
    print("Setting up LoRA...")

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def load_datasets(config: TrainingConfig):
    """Load training and validation datasets.

    Args:
        config: Training configuration

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    print("Loading datasets...")

    train_dataset = load_dataset(
        "parquet",
        data_files=config.train_path,
        split="train",
    )

    eval_dataset = load_dataset(
        "parquet",
        data_files=config.validation_path,
        split="train",
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    return train_dataset, eval_dataset


def preprocess_function(examples, tokenizer, max_length):
    """Preprocess examples for training.

    Args:
        examples: Batch of examples
        tokenizer: Tokenizer
        max_length: Maximum sequence length

    Returns:
        Tokenized examples
    """
    # Tokenize the text
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,  # Dynamic padding
        return_tensors=None,
    )

    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Fine-tune model on management data")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )

    args = parser.parse_args()

    # Load config
    config_dict = load_config(args.config)
    config = TrainingConfig(**{
        **config_dict.get("model", {}),
        **config_dict.get("lora", {}),
        **config_dict.get("training", {}),
        **config_dict.get("data", {}),
        **config_dict.get("output", {}),
        **config_dict.get("hardware", {}),
    })

    if args.output_dir:
        config.output_dir = args.output_dir

    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Management Model Fine-Tuning")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Output: {config.output_dir}")
    print(f"Epochs: {config.num_train_epochs}")
    print(f"Batch size: {config.per_device_train_batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print("=" * 60)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Setup LoRA
    model = setup_lora(model, config)

    # Load datasets
    train_dataset, eval_dataset = load_datasets(config)

    # Preprocess
    print("Preprocessing datasets...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, config.max_seq_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
    )

    eval_dataset = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer, config.max_seq_length),
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing eval",
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        save_strategy="steps",
        evaluation_strategy="steps",
        bf16=config.bf16,
        fp16=config.fp16,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Setup trainer
    metrics_callback = MetricsCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[metrics_callback],
    )

    # Train
    print("\nStarting training...")
    train_result = trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)

    # Save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {config.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
