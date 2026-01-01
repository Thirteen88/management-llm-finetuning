#!/usr/bin/env python3
"""
Preprocess raw management conversation data for training.

This script:
1. Loads raw conversation data
2. Validates and cleans the data
3. Redacts PII (personally identifiable information)
4. Tokenizes for training
5. Splits into train/validation/test sets
6. Saves as Parquet files
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import Dataset
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from tqdm import tqdm


class DataPreprocessor:
    """Preprocess management conversation data for training."""

    # Categories of management conversations
    CATEGORIES = [
        "one_on_one",
        "performance_review",
        "standup",
        "strategy",
        "hiring",
        "crisis",
        "stakeholder",
    ]

    # Role levels
    ROLE_LEVELS = ["manager", "senior_manager", "director", "vp"]

    def __init__(
        self,
        input_path: str,
        output_dir: str,
        redact_pii: bool = True,
        train_split: float = 0.8,
        val_split: float = 0.1,
    ):
        """Initialize the preprocessor.

        Args:
            input_path: Path to raw JSON/JSONL data file
            output_dir: Directory to save processed data
            redact_pii: Whether to redact PII using Presidio
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.redact_pii = redact_pii
        self.train_split = train_split
        self.val_split = val_split

        # Initialize PII engines
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    def load_data(self) -> List[Dict[str, Any]]:
        """Load raw data from file.

        Returns:
            List of conversation dictionaries
        """
        print(f"Loading data from {self.input_path}...")

        if self.input_path.suffix == ".jsonl":
            data = []
            with open(self.input_path) as f:
                for line in f:
                    data.append(json.loads(line))
        elif self.input_path.suffix == ".json":
            with open(self.input_path) as f:
                data = json.load(f)
                # Handle if data is wrapped in a key
                if isinstance(data, dict) and "conversations" in data:
                    data = data["conversations"]
        else:
            raise ValueError(f"Unsupported file format: {self.input_path.suffix}")

        print(f"Loaded {len(data)} conversations")
        return data

    def validate_entry(self, entry: Dict[str, Any]) -> bool:
        """Validate a conversation entry.

        Args:
            entry: Conversation dictionary to validate

        Returns:
            True if valid, False otherwise
        """
        required_fields = ["id", "category", "conversation", "metadata"]

        # Check required fields
        if not all(field in entry for field in required_fields):
            return False

        # Validate category
        if entry["category"] not in self.CATEGORIES:
            return False

        # Validate conversation has at least 2 turns
        if len(entry.get("conversation", [])) < 2:
            return False

        return True

    def redact_conversation(self, text: str) -> str:
        """Redact PII from conversation text.

        Args:
            text: Text to redact

        Returns:
            Redacted text
        """
        if not self.redact_pii:
            return text

        # Analyze and anonymize
        results = self.analyzer.analyze(
            text=text,
            entities=["PERSON", "EMAIL", "PHONE_NUMBER", "URL", "DATE_TIME"],
            language="en",
        )

        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
        )

        return anonymized.text

    def preprocess_entry(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Preprocess a single conversation entry.

        Args:
            entry: Raw conversation entry

        Returns:
            Preprocessed entry, or None if invalid
        """
        # Validate
        if not self.validate_entry(entry):
            return None

        # Redact PII from conversation
        if self.redact_pii:
            conversation = []
            for turn in entry["conversation"]:
                redacted_text = self.redact_conversation(turn["text"])
                conversation.append({
                    "role": turn["role"],
                    "text": redacted_text,
                })
            entry["conversation"] = conversation
            entry["metadata"]["redacted"] = True

        # Add token count (approximate)
        total_text = " ".join([t["text"] for t in entry["conversation"]])
        entry["metadata"]["tokens"] = len(total_text.split())

        return entry

    def format_for_training(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Format entry for training with prompt template.

        Args:
            entry: Preprocessed entry

        Returns:
            Formatted entry with text field for training
        """
        conversation = entry["conversation"]
        context = entry.get("context", {})

        # Create formatted conversation text
        turns = []
        for turn in conversation:
            role = turn["role"].capitalize()
            text = turn["text"]
            turns.append(f"{role}: {text}")

        conversation_text = "\n".join(turns)

        # Create training prompt
        situation = context.get("topic", "management situation")
        role_level = context.get("role_level", "manager")
        context_str = json.dumps(context, indent=2)

        prompt = f"""You are a {role_level}. Respond to the following situation:

Situation: {situation}
Context: {context_str}

Conversation:
{conversation_text}
"""

        return {
            "id": entry["id"],
            "text": prompt,
            "category": entry["category"],
            "metadata": entry["metadata"],
        }

    def split_data(self, data: List[Dict[str, Any]]) -> tuple:
        """Split data into train/validation/test sets.

        Args:
            data: List of processed entries

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        # Shuffle by category for balanced splits
        import random
        random.shuffle(data)

        n = len(data)
        train_end = int(n * self.train_split)
        val_end = int(n * (self.train_split + self.val_split))

        train = data[:train_end]
        val = data[train_end:val_end]
        test = data[val_end:]

        print(f"Split: {len(train)} train, {len(val)} val, {len(test)} test")
        return train, val, test

    def save_datasets(
        self,
        train: List[Dict[str, Any]],
        val: List[Dict[str, Any]],
        test: List[Dict[str, Any]],
    ):
        """Save processed datasets as Parquet files.

        Args:
            train: Training data
            val: Validation data
            test: Test data
        """
        print("Saving datasets...")

        for split_name, data in [("train", train), ("validation", val), ("test", test)]:
            # Convert to Hugging Face Dataset
            hf_dataset = Dataset.from_list(data)

            # Save as Parquet
            output_path = self.output_dir / f"{split_name}.parquet"
            hf_dataset.to_parquet(str(output_path))
            print(f"  Saved {split_name} to {output_path}")

        # Save metadata
        metadata = {
            "total_examples": len(train) + len(val) + len(test),
            "train_examples": len(train),
            "val_examples": len(val),
            "test_examples": len(test),
            "categories": self.CATEGORIES,
            "redacted": self.redact_pii,
        }

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")

    def run(self):
        """Run the full preprocessing pipeline."""
        print("=" * 50)
        print("Data Preprocessing Pipeline")
        print("=" * 50)

        # Load
        data = self.load_data()

        # Preprocess
        print("\nPreprocessing conversations...")
        processed = []
        for entry in tqdm(data):
            preprocessed = self.preprocess_entry(entry)
            if preprocessed:
                formatted = self.format_for_training(preprocessed)
                processed.append(formatted)

        print(f"Processed {len(processed)} valid conversations")

        # Split
        print("\nSplitting data...")
        train, val, test = self.split_data(processed)

        # Save
        print("\nSaving datasets...")
        self.save_datasets(train, val, test)

        print("\n" + "=" * 50)
        print("Preprocessing complete!")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess management conversation data for training"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/conversations.json",
        help="Path to raw input data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--no-redact",
        action="store_true",
        help="Disable PII redaction",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data for training",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data for validation",
    )

    args = parser.parse_args()

    preprocessor = DataPreprocessor(
        input_path=args.input,
        output_dir=args.output,
        redact_pii=not args.no_redact,
        train_split=args.train_split,
        val_split=args.val_split,
    )

    preprocessor.run()


if __name__ == "__main__":
    main()
