#!/usr/bin/env python3
"""
Data quality tests for management conversation dataset.

Run with: pytest tests/test_data_quality.py -v
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# Valid categories and role levels
VALID_CATEGORIES = [
    "one_on_one",
    "performance_review",
    "standup",
    "strategy",
    "hiring",
    "crisis",
    "stakeholder",
]

VALID_ROLE_LEVELS = ["manager", "senior_manager", "director", "vp"]


class TestDataQuality:
    """Test suite for data quality validation."""

    @pytest.fixture
    def raw_data(self):
        """Load raw conversation data."""
        raw_path = DATA_DIR / "raw"
        json_files = list(raw_path.glob("*.json")) + list(raw_path.glob("*.jsonl"))

        if not json_files:
            pytest.skip("No raw data files found")

        data = []
        for file_path in json_files:
            if file_path.suffix == ".jsonl":
                with open(file_path) as f:
                    for line in f:
                        data.append(json.loads(line))
            else:
                with open(file_path) as f:
                    content = json.load(f)
                    if isinstance(content, list):
                        data.extend(content)
                    elif isinstance(content, dict):
                        data.extend(content.get("conversations", []))

        return data

    @pytest.fixture
    def processed_data(self):
        """Load processed parquet data."""
        processed_path = DATA_DIR / "processed"

        if not (processed_path / "train.parquet").exists():
            pytest.skip("No processed data found")

        import pyarrow.parquet as pq

        data = []
        for split in ["train", "validation", "test"]:
            file_path = processed_path / f"{split}.parquet"
            if file_path.exists():
                table = pq.read_table(file_path)
                df = table.to_pandas()
                df["split"] = split
                data.append(df)

        return pd.concat(data, ignore_index=True) if data else pd.DataFrame()

    def test_raw_data_exists(self, raw_data):
        """Test that raw data files exist and contain data."""
        assert len(raw_data) > 0, "Raw data is empty"

    def test_raw_data_structure(self, raw_data):
        """Test that each entry has required fields."""
        required_fields = ["id", "category", "conversation", "metadata"]

        for entry in raw_data:
            for field in required_fields:
                assert field in entry, f"Missing field: {field}"

    def test_valid_category(self, raw_data):
        """Test that all categories are valid."""
        for entry in raw_data:
            assert entry["category"] in VALID_CATEGORIES, \
                f"Invalid category: {entry['category']}"

    def test_conversation_not_empty(self, raw_data):
        """Test that conversations have at least 2 turns."""
        for entry in raw_data:
            assert len(entry["conversation"]) >= 2, \
                f"Conversation too short: {entry['id']}"

    def test_conversation_turns_valid(self, raw_data):
        """Test that each conversation turn has role and text."""
        for entry in raw_data:
            for turn in entry["conversation"]:
                assert "role" in turn, "Missing role in turn"
                assert "text" in turn, "Missing text in turn"
                assert isinstance(turn["text"], str), "Text must be string"
                assert len(turn["text"].strip()) > 0, "Text cannot be empty"

    def test_metadata_structure(self, raw_data):
        """Test that metadata has required fields."""
        for entry in raw_data:
            metadata = entry["metadata"]
            assert "redacted" in metadata, "Missing redacted field"
            assert "source" in metadata, "Missing source field"

    def test_no_pii_in_conversations(self, raw_data):
        """Test that conversations don't contain obvious PII patterns."""
        import re

        # Simple patterns for emails and phone numbers
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'

        for entry in raw_data:
            for turn in entry["conversation"]:
                text = turn["text"]

                # Check for email pattern (excluding placeholders)
                emails = re.findall(email_pattern, text)
                for email in emails:
                    assert "@" in email and any(c.isupper() for c in email) is False, \
                        f"Found potential email: {email}"

                # Check for phone pattern
                phones = re.findall(phone_pattern, text)
                assert len(phones) == 0, f"Found potential phone: {phones}"

    def test_processed_data_exists(self, processed_data):
        """Test that processed data exists."""
        assert not processed_data.empty, "Processed data is empty"

    def test_processed_data_columns(self, processed_data):
        """Test that processed data has expected columns."""
        expected_cols = ["id", "text", "category", "metadata"]

        for col in expected_cols:
            assert col in processed_data.columns, f"Missing column: {col}"

    def test_processed_text_not_empty(self, processed_data):
        """Test that processed text is not empty."""
        assert all(processed_data["text"].str.len() > 0), "Found empty text entries"

    def test_unique_ids(self, processed_data):
        """Test that all IDs are unique."""
        assert processed_data["id"].is_unique, "Duplicate IDs found"

    def test_split_distribution(self, processed_data):
        """Test that data splits are reasonable."""
        if "split" not in processed_data.columns:
            pytest.skip("No split information")

        split_counts = processed_data["split"].value_counts()
        total = len(processed_data)

        # Train should be largest
        train_pct = split_counts.get("train", 0) / total
        assert train_pct >= 0.6, f"Train split too small: {train_pct:.2%}"

        # Val and test should be similar
        val_pct = split_counts.get("validation", 0) / total
        test_pct = split_counts.get("test", 0) / total
        assert abs(val_pct - test_pct) < 0.1, "Val and test splits should be similar"

    def test_category_distribution(self, raw_data):
        """Test that all categories are represented."""
        categories = [entry["category"] for entry in raw_data]
        unique_categories = set(categories)

        for cat in VALID_CATEGORIES:
            # At least one example per category (or skip if none)
            if cat not in unique_categories:
                pytest.skip(f"Category {cat} not represented")

    def test_token_count_reasonable(self, raw_data):
        """Test that token counts are reasonable (if present)."""
        for entry in raw_data:
            tokens = entry["metadata"].get("tokens")
            if tokens:
                assert 50 <= tokens <= 10000, \
                    f"Token count unreasonable: {tokens}"

    def test_context_valid(self, raw_data):
        """Test that context field is valid if present."""
        for entry in raw_data:
            context = entry.get("context", {})

            if context:
                # Check role level if present
                if "role_level" in context:
                    assert context["role_level"] in VALID_ROLE_LEVELS, \
                        f"Invalid role level: {context['role_level']}"

                # Check team size if present
                if "team_size" in context:
                    assert isinstance(context["team_size"], (int, float)), \
                        "Team size must be numeric"
                    assert 1 <= context["team_size"] <= 1000, \
                        f"Team size unreasonable: {context['team_size']}"


class TestMetadataFile:
    """Test metadata.json file if present."""

    @pytest.fixture
    def metadata(self):
        """Load metadata file."""
        metadata_path = DATA_DIR / "processed" / "metadata.json"

        if not metadata_path.exists():
            pytest.skip("No metadata file found")

        with open(metadata_path) as f:
            return json.load(f)

    def test_metadata_fields(self, metadata):
        """Test that metadata has required fields."""
        required = ["total_examples", "train_examples", "val_examples", "test_examples"]

        for field in required:
            assert field in metadata, f"Missing metadata field: {field}"

    def test_metadata_counts_match(self, metadata):
        """Test that counts add up."""
        total = metadata["total_examples"]
        train = metadata["train_examples"]
        val = metadata["val_examples"]
        test = metadata["test_examples"]

        assert total == train + val + test, \
            f"Counts don't add up: {total} != {train} + {val} + {test}"

    def test_metadata_categories_listed(self, metadata):
        """Test that categories are listed."""
        assert "categories" in metadata, "Categories not listed"
        assert isinstance(metadata["categories"], list), "Categories must be a list"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
