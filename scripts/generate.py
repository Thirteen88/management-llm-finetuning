#!/usr/bin/env python3
"""
Generate management responses using a fine-tuned model.

This script provides an interactive interface for generating
manager responses to various situations.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def generate_response(
    model,
    tokenizer,
    situation: str,
    context: dict,
    role_level: str = "manager",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate a management response.

    Args:
        model: Trained model
        tokenizer: Model tokenizer
        situation: The situation to respond to
        context: Additional context (dict)
        role_level: Manager's role level
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter

    Returns:
        Generated response
    """
    # Build prompt
    context_str = json.dumps(context, indent=2)
    prompt = f"""You are a {role_level}. Respond to the following situation:

Situation: {situation}
Context: {context_str}

Provide a constructive, professional response:
"""

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt
    if response.startswith(prompt):
        response = response[len(prompt):].strip()

    return response


def interactive_mode(model, tokenizer, args):
    """Run interactive generation mode.

    Args:
        model: Trained model
        tokenizer: Model tokenizer
        args: Command line arguments
    """
    print("\n" + "=" * 60)
    print("Interactive Management Response Generator")
    print("=" * 60)
    print("Enter 'quit' to exit\n")

    while True:
        # Get situation
        situation = input("Situation (what the employee said or happened): ").strip()
        if situation.lower() == "quit":
            break

        # Get context
        print("\nContext (press Enter for defaults):")
        role_level = input(f"  Role level [{args.role_level}]: ").strip() or args.role_level
        team_size = input(f"  Team size (optional): ").strip()

        context = {}
        if team_size:
            context["team_size"] = int(team_size)

        # Generate response
        print("\n" + "-" * 60)
        print("Generating response...")
        print("-" * 60 + "\n")

        response = generate_response(
            model,
            tokenizer,
            situation=situation,
            context=context,
            role_level=role_level,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        print(response)
        print("\n" + "=" * 60 + "\n")


def batch_mode(model, tokenizer, args):
    """Run batch generation from input file.

    Args:
        model: Trained model
        tokenizer: Model tokenizer
        args: Command line arguments
    """
    # Load input file
    with open(args.input) as f:
        situations = json.load(f)

    print(f"Loaded {len(situations)} situations from {args.input}")

    # Generate responses
    results = []
    for item in situations:
        response = generate_response(
            model,
            tokenizer,
            situation=item["situation"],
            context=item.get("context", {}),
            role_level=item.get("role_level", "manager"),
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        results.append({
            "situation": item["situation"],
            "response": response,
        })

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate management responses"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/checkpoints",
        help="Path to trained model",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["interactive", "batch"],
        default="interactive",
        help="Generation mode",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input JSON file for batch mode",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/responses.json",
        help="Output file for batch mode",
    )

    # Generation parameters
    parser.add_argument(
        "--role-level",
        type=str,
        default="manager",
        choices=["manager", "senior_manager", "director", "vp"],
        help="Default role level",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter",
    )

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(args.model)

    # Run mode
    if args.mode == "interactive":
        interactive_mode(model, tokenizer, args)
    else:
        if not args.input:
            print("Error: --input required for batch mode")
            return
        batch_mode(model, tokenizer, args)


if __name__ == "__main__":
    main()
