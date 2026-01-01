# Manager & Director Training Dataset

> A curated dataset for fine-tuning Large Language Models on management, leadership, and executive decision-making skills.

## Quick Start

```bash
# Clone and setup
git clone <your-repo-url>
cd manager-training-dataset

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Preprocess data (if you have raw data)
python scripts/preprocess.py --input data/raw/conversations.json

# Train model
python scripts/train.py --config config/training_config.yaml

# Generate responses
python scripts/generate.py --mode interactive
```

## Documentation

| Document | Description |
|----------|-------------|
| [Data Schema](docs/data_schema.md) | Complete data format documentation |
| [CONTRIBUTING.md](../CONTRIBUTING_MANAGER_TRAINING_TEMPLATE.md) | Contribution guidelines |
| [LICENSE](../LICENSE_MANAGER_TRAINING_TEMPLATE.txt) | MIT License |

## Project Structure

```
manager-training-dataset/
├── data/
│   ├── raw/                    # Raw conversation data
│   ├── processed/              # Processed training data
│   └── synthetic/              # Synthetic/augmented data
├── scripts/
│   ├── preprocess.py           # Data preprocessing
│   ├── train.py                # Fine-tuning script
│   ├── evaluate.py             # Model evaluation
│   └── generate.py             # Inference/generation
├── models/
│   └── checkpoints/            # Trained model checkpoints
├── config/
│   └── training_config.yaml    # Training configuration
├── tests/
│   └── test_data_quality.py    # Data validation tests
└── docs/
    └── data_schema.md          # Schema documentation
```

## Dataset Categories

- **1:1 Conversations** - Manager-direct reports meetings, coaching
- **Performance Reviews** - Annual/bi-annual review discussions
- **Standups & Meetings** - Daily standups, retrospectives
- **Strategy Documents** - OKRs, roadmaps, planning
- **Hiring Decisions** - Interviews, calibration discussions
- **Crisis Communication** - Incident responses, announcements
- **Stakeholder Updates** - Executive summaries, progress reports

## License

MIT License - see [LICENSE](../LICENSE_MANAGER_TRAINING_TEMPLATE.txt) for details.

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{manager_training_dataset_2025,
  author       = {Your Name},
  title        = {Manager & Director Training Dataset for LLM Fine-Tuning},
  year         = 2025,
  version      = {1.0.0},
  publisher    = {GitHub},
  url          = {https://github.com/YOUR-ORG/manager-training-dataset}
}
```
