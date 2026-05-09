# Contributing to CHiPSAL Multimodal Hate Detection

Thank you for your interest in contributing to our research project! This document provides guidelines for contributing to the codebase.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Research Contributions](#research-contributions)

---

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to samir@example.com.

---

## How Can I Contribute?

### 🐛 Reporting Bugs

Before submitting a bug report:
1. Check the [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
2. Search [existing issues](https://github.com/yourusername/chipsal-multimodal/issues)
3. Ensure you're using the latest version

**Good Bug Report Includes:**
- Clear, descriptive title
- Steps to reproduce
- Expected vs. actual behavior
- Environment details (OS, Python version, CUDA version, GPU model)
- Full error traceback
- Minimal reproducible code example

**Template:**
```markdown
**Environment:**
- OS: Windows 11 / Ubuntu 22.04 / macOS 13
- Python: 3.10.12
- PyTorch: 2.0.1
- CUDA: 11.8
- GPU: NVIDIA RTX 3090

**Bug Description:**
[Clear description of the bug]

**Steps to Reproduce:**
1. [First step]
2. [Second step]
3. [...]

**Expected Behavior:**
[What you expected to happen]

**Actual Behavior:**
[What actually happened]

**Error Traceback:**
```
[Paste full error traceback here]
```

**Minimal Reproducible Example:**
```python
[Minimal code that reproduces the issue]
```
```

---

### 💡 Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Clear title** describing the enhancement
- **Detailed description** of the proposed functionality
- **Use cases** explaining why this would be useful
- **Possible implementation** (optional)
- **Related work** (papers, repos) if applicable

---

### 🔧 Contributing Code

We welcome:
- **Bug fixes**
- **Performance improvements**
- **New fusion architectures**
- **Additional datasets/languages**
- **Visualization tools**
- **Documentation improvements**
- **Test coverage**

---

## Development Setup

### 1. Fork the Repository

Click "Fork" on GitHub to create your own copy.

### 2. Clone Your Fork

```bash
git clone https://github.com/YOUR_USERNAME/chipsal-multimodal-hate-detection.git
cd chipsal-multimodal-hate-detection
```

### 3. Set Up Development Environment

```bash
# Create virtual environment
python -m venv chipsal_dev
source chipsal_dev/bin/activate  # Windows: chipsal_dev\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy isort pre-commit

# Install package in editable mode
pip install -e .

# Install pre-commit hooks
pre-commit install
```

### 4. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number-description
```

**Branch Naming:**
- `feature/` - New features
- `bugfix/` - Bug fixes
- `refactor/` - Code refactoring
- `docs/` - Documentation changes
- `test/` - Test additions/improvements

---

## Coding Standards

### Python Style Guide

We follow **PEP 8** with some modifications:

- **Line length:** 100 characters (not 79)
- **Indentation:** 4 spaces
- **Quotes:** Double quotes for strings
- **Docstrings:** Google style

### Code Formatting

Use **Black** for automatic formatting:

```bash
black abiral/ samir/ tests/ --line-length 100
```

### Linting

```bash
# Flake8
flake8 abiral/ samir/ tests/ --max-line-length=100

# Type checking
mypy abiral/ --ignore-missing-imports

# Import sorting
isort abiral/ samir/ tests/
```

### Pre-commit Hooks

Automatically run checks before commit:

```bash
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        args: [--line-length=100]
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        args: [--max-line-length=100]
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: [--profile=black]
```

---

## Docstring Standards

Use **Google-style docstrings**:

```python
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 10,
    device: str = "cuda"
) -> Dict[str, List[float]]:
    """Train a multimodal classification model.
    
    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training data.
        optimizer: Optimizer (e.g., AdamW).
        num_epochs: Number of training epochs. Defaults to 10.
        device: Device to train on ('cuda' or 'cpu'). Defaults to 'cuda'.
    
    Returns:
        Dictionary containing training metrics:
            - 'train_loss': List of training losses per epoch
            - 'train_acc': List of training accuracies per epoch
    
    Raises:
        ValueError: If num_epochs < 1.
        RuntimeError: If CUDA is specified but not available.
    
    Example:
        >>> model = MultimodalClassifier()
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        >>> metrics = train_model(model, train_loader, optimizer, num_epochs=30)
        >>> print(f"Final loss: {metrics['train_loss'][-1]:.4f}")
    """
    if num_epochs < 1:
        raise ValueError("num_epochs must be >= 1")
    
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA specified but not available")
    
    # Training logic...
    return metrics
```

---

## Pull Request Process

### 1. Make Your Changes

- Write clean, readable code
- Follow coding standards
- Add tests for new functionality
- Update documentation

### 2. Test Your Changes

```bash
# Run all tests
pytest tests/ -v

# Test with coverage
pytest --cov=abiral tests/

# Run specific test
pytest tests/test_model.py::test_hybrid_fusion -v
```

### 3. Commit Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "feat: Add cross-modal attention visualization

- Implement attention heatmap generation
- Add support for multi-head attention vis
- Include example in notebooks/
- Update docs/API.md with new function"
```

**Commit Message Format:**
```
<type>: <short summary>

<optional detailed description>

<optional footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting (no logic change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `perf`: Performance improvement
- `chore`: Maintenance tasks

### 4. Push Changes

```bash
git push origin feature/your-feature-name
```

### 5. Create Pull Request

- Go to GitHub and click "New Pull Request"
- Fill out the PR template
- Link related issues (e.g., "Fixes #123")
- Request review

**PR Checklist:**
- [ ] Code follows style guidelines
- [ ] Self-review of code completed
- [ ] Commented complex logic
- [ ] Updated documentation
- [ ] Added tests with >80% coverage
- [ ] All tests pass
- [ ] No new warnings
- [ ] Updated CHANGELOG.md (if applicable)

---

## Testing Guidelines

### Unit Tests

Write tests for all new functionality:

```python
# tests/test_fusion.py
import pytest
import torch
from abiral.fusion_models import HybridFusion

def test_hybrid_fusion_forward():
    """Test HybridFusion forward pass."""
    model = HybridFusion(
        visual_dim=512,
        text_dim=1024,
        hidden_dim=256,
        num_classes=2,
        attention_heads=4
    )
    
    # Create dummy inputs
    visual_feats = torch.randn(8, 512)  # batch=8
    text_feats = torch.randn(8, 1024)
    
    # Forward pass
    logits = model(visual_feats, text_feats)
    
    # Assertions
    assert logits.shape == (8, 2), f"Expected shape (8, 2), got {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN in outputs"
    assert not torch.isinf(logits).any(), "Inf in outputs"

def test_hybrid_fusion_gating():
    """Test gating mechanism produces valid weights."""
    model = HybridFusion(visual_dim=512, text_dim=1024, hidden_dim=256)
    
    visual_feats = torch.randn(4, 512)
    text_feats = torch.randn(4, 1024)
    
    # Get gating weights
    gates = model.compute_gates(visual_feats, text_feats)
    
    # Assertions
    assert gates.shape == (4, 2), "Gates should be (batch_size, 2)"
    assert torch.allclose(gates.sum(dim=1), torch.ones(4)), "Gates should sum to 1"
    assert (gates >= 0).all() and (gates <= 1).all(), "Gates should be in [0, 1]"
```

### Integration Tests

Test end-to-end workflows:

```python
def test_training_pipeline():
    """Test full training pipeline."""
    # Create small synthetic dataset
    dataset = create_dummy_dataset(n_samples=100)
    train_loader = DataLoader(dataset, batch_size=16)
    
    # Initialize model
    model = MultimodalClassifier(model_type='hybrid')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Train for 2 epochs
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=train_loader,
        optimizer=optimizer,
        num_epochs=2,
        device='cpu'
    )
    
    metrics = trainer.train()
    
    # Check training completed
    assert len(metrics['train_loss']) == 2
    assert metrics['train_loss'][-1] < metrics['train_loss'][0], "Loss should decrease"
```

### Test Coverage

Aim for **>80% coverage** for new code:

```bash
pytest --cov=abiral --cov-report=html tests/
# Open htmlcov/index.html to view coverage report
```

---

## Documentation

### Updating Documentation

When adding features, update:

1. **Docstrings** in code
2. **API.md** for new functions/classes
3. **EXPERIMENTS.md** for new experiments
4. **README.md** if user-facing changes
5. **CHANGELOG.md** with version info

### Building Documentation (Optional)

If using Sphinx:

```bash
cd docs/
make html
```

---

## Research Contributions

We welcome research contributions:

### New Fusion Architectures

If proposing a new fusion method:

1. **Implement in** `abiral/fusion_models.py`
2. **Add config** in `abiral/experiment_configs.py`
3. **Run ablation study** comparing to M1-M8
4. **Document results** in PR description
5. **Include paper references** if based on published work

**Example:**

```python
class TransformerFusion(nn.Module):
    """Cross-modal transformer fusion.
    
    Based on: Tsai et al. "Multimodal Transformer for Unaligned
    Multimodal Language Sequences" (ACL 2019).
    """
    def __init__(self, visual_dim, text_dim, num_layers=2):
        super().__init__()
        # Implementation...
```

### Dataset Support

To add support for new languages/datasets:

1. **Create dataset class** in `abiral/dataset.py`
2. **Add data preprocessing** scripts
3. **Update experiment configs**
4. **Provide sample data** (or instructions to obtain)
5. **Document** in README and paper

---

## Code Review Process

### For Reviewers

- Be respectful and constructive
- Focus on code quality, not style (automated tools handle style)
- Test changes locally if possible
- Approve when satisfied or request changes

### For Contributors

- Address all review comments
- Be open to feedback
- Ask questions if unclear
- Update PR based on feedback

---

## Release Process

(For maintainers)

1. Update version in `setup.py`
2. Update `CHANGELOG.md`
3. Create git tag: `git tag -a v1.1.0 -m "Version 1.1.0"`
4. Push tag: `git push origin v1.1.0`
5. Create GitHub release
6. (Optional) Publish to PyPI

---

## Questions?

- **Documentation:** [docs/](docs/)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/chipsal-multimodal/discussions)
- **Issues:** [GitHub Issues](https://github.com/yourusername/chipsal-multimodal/issues)
- **Email:** samir@example.com

---

## Recognition

Contributors will be acknowledged in:
- **README.md** contributors section
- **Paper acknowledgments** (for significant contributions)
- **GitHub contributors** page

Thank you for contributing to multimodal hate speech detection research! 🙏

---

## Quick Reference

```bash
# Setup
git clone [fork-url]
cd chipsal-multimodal-hate-detection
python -m venv chipsal_dev
source chipsal_dev/bin/activate
pip install -r requirements.txt
pip install -e .

# Development
git checkout -b feature/my-feature
# Make changes
black abiral/ --line-length 100
flake8 abiral/ --max-line-length=100
pytest tests/ -v

# Commit
git add .
git commit -m "feat: Description"
git push origin feature/my-feature
# Create PR on GitHub
```

---

**Last Updated:** March 1, 2026
