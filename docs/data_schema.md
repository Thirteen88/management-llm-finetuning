# Data Schema Documentation

This document describes the schema and structure of the management conversation dataset.

## Overview

The dataset consists of management and leadership conversations organized into categories. Each entry represents a complete conversation or interaction with metadata.

## Entry Schema

### Top-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique identifier for the entry (UUID or slug) |
| `category` | enum | Yes | One of the defined categories |
| `context` | object | No | Additional context about the conversation |
| `conversation` | array | Yes | List of conversation turns |
| `metadata` | object | Yes | Metadata about the entry |

### Valid Categories

| Category | Description |
|----------|-------------|
| `one_on_one` | Manager-direct report 1:1 meetings |
| `performance_review` | Annual/bi-annual performance discussions |
| `standup` | Daily standups, sprint planning |
| `strategy` | OKRs, roadmaps, strategic planning |
| `hiring` | Interviews, calibration, hiring decisions |
| `crisis` | Incident responses, crisis communication |
| `stakeholder` | Executive updates, cross-functional alignment |

## Context Object

| Field | Type | Description |
|-------|------|-------------|
| `role_level` | enum | `manager`, `senior_manager`, `director`, `vp` |
| `team_size` | number | Number of direct reports (optional) |
| `topic` | string | Brief description of the topic |
| `participants` | array | List of participant roles (optional) |
| `industry` | string | Industry context (optional) |
| `tenure` | string | Employee tenure (optional) |

## Conversation Array

Each conversation is an array of turns:

```json
{
  "role": "manager",
  "text": "Conversation text..."
}
```

### Valid Roles

| Role | Description |
|------|-------------|
| `manager` | The manager/leader in the conversation |
| `employee` | Direct report or team member |
| `candidate` | Job candidate (for hiring category) |
| `stakeholder` | Other stakeholders |
| `hr` | HR representative |
| `peer` | Peer manager or colleague |

## Metadata Object

| Field | Type | Description |
|-------|------|-------------|
| `redacted` | boolean | Whether PII has been redacted |
| `synthetic` | boolean | Whether the conversation is synthetic |
| `source` | string | Source of the conversation |
| `tokens` | number | Approximate token count |
| `language` | string | Language code (default: `en`) |
| `created_at` | string | ISO timestamp of creation |

## Example Entry

```json
{
  "id": "one_on_one_career_growth_001",
  "category": "one_on_one",
  "context": {
    "role_level": "manager",
    "team_size": 5,
    "topic": "career growth discussion",
    "participants": ["manager", "senior_engineer"],
    "tenure": "2 years"
  },
  "conversation": [
    {
      "role": "employee",
      "text": "I don't feel like I'm growing in my role anymore."
    },
    {
      "role": "manager",
      "text": "I appreciate you sharing that with me. Can you tell me more about what growth would look like for you? Is it technical depth, scope of ownership, or something else like people leadership?"
    }
  ],
  "metadata": {
    "redacted": true,
    "synthetic": false,
    "source": "anonymized_real_data",
    "tokens": 847,
    "language": "en"
  }
}
```

## Processed Data Schema

After preprocessing, data is formatted for training with additional fields:

### Training Format

```json
{
  "id": "one_on_one_career_growth_001",
  "text": "You are a manager. Respond to the following situation...\n\nSituation: career growth discussion\n\nConversation:\nEmployee: I don't feel like I'm growing...\n\nManager: I appreciate you sharing...",
  "category": "one_on_one",
  "metadata": {
    "tokens": 847,
    "redacted": true
  }
}
```

The `text` field contains the full formatted prompt for training.

## File Formats

### Raw Data

- **Format**: JSON or JSONL
- **Location**: `data/raw/`
- **Structure**: Array of entry objects

### Processed Data

- **Format**: Apache Parquet
- **Location**: `data/processed/`
- **Files**:
  - `train.parquet` - Training set
  - `validation.parquet` - Validation set
  - `test.parquet` - Test set
  - `metadata.json` - Dataset metadata

## Data Quality Requirements

### Validation Rules

1. **Required Fields**: All top-level fields must be present
2. **Valid Category**: Must be one of the 7 defined categories
3. **Minimum Turns**: At least 2 conversation turns
4. **Non-empty Text**: All conversation text must be non-empty
5. **Valid Role**: Each turn must have a valid role

### PII Redaction

All personally identifiable information must be redacted:

| Element | Replacement |
|---------|-------------|
| Names | `[NAME]`, `[ENGINEER]`, `[MANAGER]` |
| Emails | `[EMAIL]` |
| Phone | `[PHONE]` |
| Companies | `[COMPANY]` |
| Locations | `[CITY]`, `[OFFICE]` |

## Statistics

The dataset metadata includes:

```json
{
  "total_examples": 10000,
  "train_examples": 8000,
  "val_examples": 1000,
  "test_examples": 1000,
  "categories": ["one_on_one", "performance_review", ...],
  "redacted": true,
  "version": "1.0.0"
}
```

## Contributing Data

When contributing new data:

1. Follow the schema exactly
2. Ensure all PII is redacted
3. Validate using `pytest tests/test_data_quality.py`
4. Include diverse perspectives and scenarios
5. Ensure professional, constructive tone

See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.
