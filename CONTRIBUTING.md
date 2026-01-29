# Contributing to TorchWM

We welcome contributions to torchwm! This document outlines the guidelines for contributing to the project.

## Ways to Contribute

- **Report Bugs**: Open an issue on GitHub with a clear description of the bug, steps to reproduce, and expected behavior.
- **Suggest Features**: Propose new features or improvements via GitHub issues.
- **Submit Pull Requests**: Implement fixes or enhancements and submit a PR.

## Development Setup

1. Fork the repository and clone your fork:
   ```bash
   git clone https://github.com/your-username/torchwm.git
   cd torchwm
   ```

2. Install dependencies in editable mode with development extras:
   ```bash
   pip install -e ".[dev]"
   ```

3. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Coding Standards

- Follow PEP 8 style guidelines.
- Use type hints where possible.
- Ensure code is linted with Ruff: `ruff check .`
- Format code with Ruff: `ruff format .`

## Testing

- Run tests with: `pytest`
- Ensure all tests pass before submitting a PR.
- Add tests for new features or bug fixes.

## Pull Request Process

1. Create a feature branch from `main`: `git checkout -b feature/your-feature`
2. Make your changes and commit them.
3. Push to your fork and create a PR.
4. PRs must pass CI checks and have at least one approval.

## Code of Conduct

Please be respectful and inclusive. Harassment or discriminatory behavior will not be tolerated.

For questions, reach out via GitHub issues.
