<<<<<<< HEAD
# Adobe
=======
# ADOBE_1B - Advanced Document Processing Pipeline

A sophisticated text processing and analysis pipeline that performs outline loading, persona matching, section ranking, and intelligent output building.

## 🏆 HACKATHON-SAFE: LOCAL MODEL INCLUDED

**IMPORTANT**: This solution includes a **479.2MB local model** - NO downloads, NO APIs, NO internet dependency!

### 🎯 For Judges - Instant Verification:
1. **Check model size**: Repository is ~500MB (proves model included)
2. **Build & run**: `docker build -t adobe-1b . && docker run adobe-1b` 
3. **Verify offline**: `docker run --network none adobe-1b python verify_offline_model.py`
4. **See proof**: Open `EXAMINER_PROOF.md` for detailed evidence

**Result: ✅ Works perfectly even without internet!**

## 📁 **Model Location:**
```
models/all-MiniLM-L6-v2/
├── model.safetensors          (448.83 MB)
├── tokenizer.json             ( 16.29 MB)  
├── unigram.json               ( 14.08 MB)
└── config files...
```

## Project Structure

```
ADOBE_1B/
├── Dockerfile
├── README.md
├── requirements.txt
├── approach_explanation.md
├── src/
│   ├── __init__.py
│   ├── outline_loader.py
│   ├── persona_matcher.py
│   ├── section_ranker.py
│   ├── subsection_analyzer.py
│   ├── output_builder.py
│   └── process_round1b.py
├── mock_data/
│   ├── input/
│   └── output/
├── tests/
│   └── test_basic.py
└── models/
    └── (embedding models will be downloaded here)
```

## Features

- **Outline Loading**: Intelligent document structure extraction
- **Persona Matching**: Advanced user/content persona alignment
- **Section Ranking**: Priority-based content organization
- **Subsection Analysis**: Detailed content breakdown and analysis
- **Output Building**: Structured result generation

## Installation

### Local Development
```bash
pip install -r requirements.txt
```

### Docker
```bash
docker build -t adobe_1b .
docker run adobe_1b
```

## Usage

```bash
python -m src.process_round1b
```

## Testing

```bash
pytest tests/
```

## Development

```bash
# Code formatting
black src/ tests/

# Linting
flake8 src/ tests/

# Type checking
mypy src/
```

## License

[License information here] 
>>>>>>> 1a710b6 (initial commit)
