# Contributing to AutoShorts

First off, thank you for considering contributing to AutoShorts! It's people like you who make the open-source community such an amazing place to learn, inspire, and create.

## 🛠️ Development Setup

AutoShorts is built to be developer-friendly. We use `conda` and `make` to manage the environment.

1. **Clone the repository**:

    ```bash
    git clone https://github.com/divyaprakash0426/autoshorts.git
    cd autoshorts
    ```

2. **Install dependencies**:
    We have a comprehensive `Makefile`. To set up your local environment (including the specialized build of `decord` for CUDA):

    ```bash
    make install
    ```

3. **Activate environment**:
    - **Nushell**: `overlay use .venv/bin/activate.nu`
    - **Bash/Zsh**: `source .venv/bin/activate`

## 🧪 Testing

Before submitting a Pull Request, please ensure all tests pass. We use `pytest` for testing.

```bash
pytest tests/
```

If you add new features, please include corresponding tests in the `tests/` directory.

## 🤝 Pull Request Process

1. **Fork the repo** and create your branch from `main`.
2. If you've added code that should be tested, **add tests**.
3. Ensure the test suite passes.
4. Make sure your code follows the project's coding style (PEP8).
5. Open a Pull Request with a clear title and description of your changes.

## 📜 Coding Standards

- **Python**: We follow PEP8.
- **Imports**: Use absolute imports where possible within the `src/` directory.
- **Documentation**: Provide docstrings for all new functions and classes.
- **Type Hints**: Use type hints for all function signatures.

## 💬 Community

If you have questions or want to discuss a feature before building it, feel free to open an **Issue** or join the discussion on Hacker News/social media where the project was launched.

Thank you for contributing! 🚀
