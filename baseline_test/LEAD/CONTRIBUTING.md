# Contributing to LEAD

We welcome contributions! Here's how you can help.

## Getting Started

1. Fork the repository
2. Clone your fork:

```bash
git clone https://github.com/<your-username>/LEAD.git
cd LEAD
```

3. Install in development mode:

```bash
pip install -e ".[dev]"
```

4. Create a new branch:

```bash
git checkout -b feature/your-feature-name
```

## Development Workflow

### Running Tests

```bash
bash script/run_tests.sh
```

Or directly:

```bash
python -m pytest tests/ -v
```

### Code Style

- Follow PEP 8 conventions
- All functions must have docstrings (Args, Returns, Raises)
- Keep functions under 30 lines where possible
- Use descriptive variable names

### Adding New Features

1. Add your implementation in the appropriate module under `lead/`
2. Add corresponding tests in `tests/`
3. Update `lead/__init__.py` if you're adding new public APIs
4. Update `README.md` if needed

## Pull Request Process

1. Ensure all tests pass
2. Update documentation as needed
3. Write a clear PR description
4. Reference any related issues

## Reporting Issues

Please include:

- Python version and OS
- GPU model and CUDA version (if applicable)
- Steps to reproduce
- Expected vs actual behavior
- Full error traceback

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
