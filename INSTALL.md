# Installation Guide

## System Requirements

- **Python**: 3.10, 3.11, or 3.12
- **CUDA**: 12.1
- **GPU**: NVIDIA GPU with CUDA support (recommended: 16GB+ VRAM for Llama models)

## Installation Steps

### 1. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### 3. Install PyTorch with CUDA 12.1 Support

```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install Production Dependencies

**Option A: Using requirements.txt**
```bash
pip install -r requirements.txt
```

**Option B: Using pyproject.toml**
```bash
pip install -e .
```

### 5. Install Development Dependencies (Optional)

```bash
pip install -r requirements-dev.txt
# OR
pip install -e ".[dev]"
```

## Verify Installation

### Check PyTorch CUDA Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

Expected output:
```
PyTorch: 2.2.0+cu121
CUDA available: True
CUDA version: 12.1
```

### Check Transformers Installation

```bash
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### Run Tests

```bash
pytest tests/
```

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA OOM errors when loading models:

1. Use smaller models or reduce batch sizes
2. Enable gradient checkpointing
3. Use 8-bit or 4-bit quantization (install `bitsandbytes`)

### Import Errors

If you get import errors, ensure you're in the project root and have activated your virtual environment:

```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip list  # Verify installed packages
```

### CUDA Version Mismatch

If PyTorch doesn't detect CUDA:

1. Verify NVIDIA driver: `nvidia-smi`
2. Check CUDA installation: `nvcc --version`
3. Reinstall PyTorch with correct CUDA version

## Alternative: Docker Installation (Coming Soon)

A Docker image with all dependencies pre-installed will be provided in future releases.

## Development Setup

For development work, install pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

Run code quality checks:

```bash
ruff check .
mypy .
pytest tests/ --cov=. --cov-report=html
```
