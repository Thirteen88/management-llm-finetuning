#!/usr/bin/env python3
"""
Evaluate a fine-tuned management model.

This script:
- Loads a trained model
- Runs evaluation on test data
- Computes metrics (loss, perplexity)
- Generates sample outputs for qualitative review
"""

import argparse
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model(model_path: str, device: str = "cuda"):
    """Load trained model and tokenizer.

    Args:
        model_path: Path to trained model directory
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.float16,
    )

    model.eval()

    return model, tokenizer


def compute_perplexity(model, tokenizer, dataset):
    """Compute perplexity on a dataset.

    Args:
        model: Trained model
        tokenizer: Model tokenizer
        dataset: Evaluation dataset

    Returns:
        Perplexity score
    """
    print("Computing perplexity...")

    total_loss = 0.0
    total_tokens = 0

    for i, example in enumerate(dataset):
        # Skip if no text field
        if "text" not in example:
            continue

        # Tokenize
        inputs = tokenizer(
            example["text"],
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(model.device)

        # Compute loss
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()

        total_loss += loss * inputs["input_ids"].size(1)
        total_tokens += inputs["input_ids"].size(1)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1} examples...")

    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return perplexity, avg_loss


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate a response from the model.

    Args:
        model: Trained model
        tokenizer: Model tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter

    Returns:
        Generated response text
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt from response
    if response.startswith(prompt):
        response = response[len(prompt):].strip()

    return response


def qualitative_evaluation(model, tokenizer, test_dataset, num_samples: int = 5):
    """Generate sample outputs for qualitative review.

    Args:
        model: Trained model
        tokenizer: Model tokenizer
        test_dataset: Test dataset
        num_samples: Number of samples to generate
    """
    print("\n" + "=" * 60)
    print("Qualitative Evaluation")
    print("=" * 60)

    for i in range(min(num_samples, len(test_dataset))):
        example = test_dataset[i]

        print(f"\n--- Sample {i + 1} ---")
        print(f"Category: {example.get('category', 'N/A')}")

        # Create prompt from situation/context
        context = example.get("context", {})
        situation = context.get("topic", "management situation")
        role_level = context.get("role_level", "manager")

        prompt = f"""You are a {role_level}. A direct report says:

"{situation}"

How do you respond?
"""

        print(f"\nPrompt:\n{prompt}")

        # Generate response
        response = generate_response(model, tokenizer, prompt)
        print(f"\nGenerated Response:\n{response}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate management model")
    parser.add_argument(
        "--model",
        type=str,
        default="models/checkpoints",
        help="Path to trained model",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/processed/test.parquet",
        help="Path to test dataset",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of qualitative samples to generate",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    inference_config = config.get("inference", {})

    # Load model
    model, tokenizer = load_model(args.model)

    # Load test data
    print(f"\nLoading test data from {args.test_data}...")
    test_dataset = load_dataset("parquet", data_files=args.test_data, split="train")
    print(f"Loaded {len(test_dataset)} test examples")

    # Compute perplexity
    perplexity, avg_loss = compute_perplexity(model, tokenizer, test_dataset)

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    print("=" * 60)

    # Qualitative evaluation
    qualitative_evaluation(
        model,
        tokenizer,
        test_dataset,
        num_samples=args.samples,
    )


if __name__ == "__main__":
    main()
