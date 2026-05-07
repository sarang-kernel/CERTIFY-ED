# Contributing to CERTIFY-ED

Thank you for considering contributing to CERTIFY-ED!

## Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/sarang-kernel/CERTIFY-ED.git
cd certify-ed
```

3. Install in development mode:

```bash
pip install -e ".[dev]"
```

4. Create a branch for your changes:

```bash
git checkout -b feature/your-feature-name
```

## Code Standards

### Python Style

- Follow PEP 8
- Use Black for formatting: `black src tests scripts`
- Use type hints for function signatures
- Maximum line length: 100 characters

### Testing

- All new features must include tests
- Maintain >90% code coverage
- Run tests before submitting: `pytest tests/ -v`
- Run with coverage: `pytest tests/ --cov=certifyEd`

### Documentation

- Add docstrings to all public functions/classes
- Use NumPy-style docstrings
- Update README.md if adding new features

## Submitting Changes

1. Ensure all tests pass
2. Run Black and flake8:

```bash
black src tests scripts
flake8 src tests scripts
```

3. Commit your changes:

```bash
git add .
git commit -m "Brief description of changes"
```

4. Push to your fork:

```bash
git push origin feature/your-feature-name
```

5. Open a Pull Request on GitHub

## Pull Request Guidelines

- Provide a clear description of the changes
- Reference any related issues
- Ensure CI tests pass
- Request review from maintainers

## Reporting Issues

- Use GitHub Issues
- Provide:
  - Clear description of the problem
  - Minimal reproducible example
  - Python version and OS
  - Error messages/stack traces

## Questions?

Open a GitHub Discussion or Issue.

Thank you for contributing!
