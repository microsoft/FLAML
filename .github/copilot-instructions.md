# GitHub Copilot Instructions for FLAML

## Project Overview

FLAML (Fast Library for Automated Machine Learning & Tuning) is a lightweight Python library for efficient automation of machine learning and AI operations. It automates workflow based on large language models, machine learning models, etc. and optimizes their performance.

**Key Components:**

- `flaml/automl/`: AutoML functionality for classification and regression
- `flaml/tune/`: Generic hyperparameter tuning
- `flaml/default/`: Zero-shot AutoML with default configurations
- `flaml/autogen/`: Legacy autogen code (note: AutoGen has moved to a separate repository)
- `flaml/fabric/`: Microsoft Fabric integration
- `test/`: Comprehensive test suite

## Build and Test Commands

### Installation

```bash
# Basic installation
pip install -e .

# Install with test dependencies
pip install -e .[test]

# Install with automl dependencies
pip install -e .[automl]

# Install with forecast dependencies (Linux only)
pip install -e .[forecast]
```

### Running Tests

```bash
# Run all tests (excluding autogen)
pytest test/ --ignore=test/autogen --reruns 2 --reruns-delay 10

# Run tests with coverage
coverage run -a -m pytest test --ignore=test/autogen --reruns 2 --reruns-delay 10
coverage xml

# Check dependencies
python test/check_dependency.py
```

### Linting and Formatting

```bash
# Run pre-commit hooks
pre-commit run --all-files

# Format with black (line length: 120)
black . --line-length 120

# Run ruff for linting and auto-fix
ruff check . --fix
```

## Code Style and Formatting

### Python Style

- **Line length:** 120 characters (configured in both Black and Ruff)
- **Formatter:** Black (v23.3.0+)
- **Linter:** Ruff with Pyflakes and pycodestyle rules
- **Import sorting:** Use isort (via Ruff)
- **Python version:** Supports Python >= 3.10 (full support for 3.10, 3.11, 3.12 and 3.13)

### Code Quality Rules

- Follow Black formatting conventions
- Keep imports sorted and organized
- Avoid unused imports (F401) - these are flagged but not auto-fixed
- Avoid wildcard imports (F403) where possible
- Complexity: Max McCabe complexity of 10
- Use type hints where appropriate
- Write clear docstrings for public APIs

### Pre-commit Hooks

The repository uses pre-commit hooks for:

- Checking for large files, AST syntax, YAML/TOML/JSON validity
- Detecting merge conflicts and private keys
- Trailing whitespace and end-of-file fixes
- pyupgrade for Python 3.8+ syntax
- Black formatting
- Markdown formatting (mdformat with GFM and frontmatter support)
- Ruff linting with auto-fix

## Testing Strategy

### Test Organization

- Tests are in the `test/` directory, organized by module
- `test/automl/`: AutoML feature tests
- `test/tune/`: Hyperparameter tuning tests
- `test/default/`: Zero-shot AutoML tests
- `test/nlp/`: NLP-related tests
- `test/spark/`: Spark integration tests

### Test Requirements

- Write tests for new functionality
- Ensure tests pass on multiple Python versions (3.10, 3.11, 3.12 and 3.13)
- Tests should work on both Ubuntu and Windows
- Use pytest markers for platform-specific tests (e.g., `@pytest.mark.spark`)
- Tests should be idempotent and not depend on external state
- Use `--reruns 2 --reruns-delay 10` for flaky tests

### Coverage

- Aim for good test coverage on new code
- Coverage reports are generated for Python 3.11 builds
- Coverage reports are uploaded to Codecov

## Git Workflow and Best Practices

### Branching

- Main branch: `main`
- Create feature branches from `main`
- PR reviews are required before merging

### Commit Messages

- Use clear, descriptive commit messages
- Reference issue numbers when applicable
- ALWAYS run `pre-commit run --all-files` before each commit to avoid formatting issues

### Pull Requests

- Ensure all tests pass before requesting review
- Update documentation if adding new features
- Follow the PR template in `.github/PULL_REQUEST_TEMPLATE.md`

## Project Structure

```
flaml/
├── automl/         # AutoML functionality
├── tune/           # Hyperparameter tuning
├── default/        # Zero-shot AutoML
├── autogen/        # Legacy autogen (deprecated, moved to separate repo)
├── fabric/         # Microsoft Fabric integration
├── onlineml/       # Online learning
└── version.py      # Version information

test/               # Test suite
├── automl/
├── tune/
├── default/
├── nlp/
└── spark/

notebook/           # Example notebooks
website/            # Documentation website
```

## Dependencies and Package Management

### Core Dependencies

- NumPy >= 1.17
- Python >= 3.10 (officially supported: 3.10, 3.11, 3.12 and 3.13)

### Optional Dependencies

- `[automl]`: lightgbm, xgboost, scipy, pandas, scikit-learn
- `[test]`: Full test suite dependencies
- `[spark]`: PySpark and joblib dependencies
- `[forecast]`: holidays, prophet, statsmodels, hcrystalball, pytorch-forecasting, pytorch-lightning, tensorboardX
- `[hf]`: Hugging Face transformers and datasets
- See `setup.py` for complete list

### Version Constraints

- Be mindful of Python version-specific dependencies (check setup.py)
- XGBoost versions differ based on Python version
- NumPy 2.0+ only for Python >= 3.13
- Some features (like vowpalwabbit) only work with older Python versions

## Boundaries and Restrictions

### Do NOT Modify

- `.git/` directory and Git configuration
- `LICENSE` file
- Version information in `flaml/version.py` (unless explicitly updating version)
- GitHub Actions workflows without careful consideration
- Existing test files unless fixing bugs or adding coverage

### Be Cautious With

- `setup.py`: Changes to dependencies should be carefully reviewed
- `pyproject.toml`: Linting and testing configuration
- `.pre-commit-config.yaml`: Pre-commit hook configuration
- Backward compatibility: FLAML is a library with external users

### Security Considerations

- Never commit secrets or API keys
- Be careful with external data sources in tests
- Validate user inputs in public APIs
- Follow secure coding practices for ML operations

## Special Notes

### AutoGen Migration

- AutoGen has moved to a separate repository: https://github.com/microsoft/autogen
- The `flaml/autogen/` directory contains legacy code
- Tests in `test/autogen/` are ignored in the main test suite
- Direct users to the new AutoGen repository for AutoGen-related issues

### Platform-Specific Considerations

- Some tests only run on Linux (e.g., forecast tests with prophet)
- Windows and Ubuntu are the primary supported platforms
- macOS support exists but requires special libomp setup for lgbm/xgboost

### Performance

- FLAML focuses on efficient automation and tuning
- Consider computational cost when adding new features
- Optimize for low resource usage where possible

## Documentation

- Main documentation: https://microsoft.github.io/FLAML/
- Update documentation when adding new features
- Provide clear examples in docstrings
- Add notebook examples for significant new features

## Contributing

- Follow the contributing guide: https://microsoft.github.io/FLAML/docs/Contribute
- Sign the Microsoft CLA when making your first contribution
- Be respectful and follow the Microsoft Open Source Code of Conduct
- Join the Discord community for discussions: https://discord.gg/Cppx2vSPVP
