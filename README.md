# NLP Complaint Classification Pipeline

## Overview
Production-ready NLP classification pipeline built using strict Test-Driven Development (TDD) methodology.

## Project Structure
```
nlp-complaint-classification-pipeline/
├── data/                   # Dataset directory
├── src/                    # Source code modules
│   ├── data_validation.py  # Data validation logic
│   ├── preprocessing.py    # Text preprocessing
│   ├── vectorization.py    # Feature extraction
│   ├── train.py           # Model training
│   ├── predict.py         # Prediction interface
│   ├── evaluate.py        # Model evaluation
│   └── pipeline.py        # End-to-end orchestration
├── tests/                 # Test suite
└── requirements.txt       # Python dependencies
```

## Development Approach
This project follows strict **RED → GREEN → REFACTOR** TDD methodology:
- ✅ Tests written first
- ✅ Minimal implementation to pass tests
- ✅ Refactor for quality and maintainability
- ✅ Leakage-safe design principles
- ✅ Production-ready code quality

## Setup
```bash
pip install -r requirements.txt
```

## Testing
```bash
pytest
```

## Status
🚧 **Project initialized** - Ready for TDD Cycle 1
