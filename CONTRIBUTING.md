# Contributing to LinguaLabel

Thank you for your interest in contributing to LinguaLabel! We welcome contributions from developers, linguists, NLP researchers, and open-source enthusiasts.

---

## Code of Conduct

Please be respectful, constructive, and collaborative in all issues, discussions, and pull requests.

---

## Development Setup

1. **Fork and Clone the Repository**:
   ```bash
   git clone https://github.com/Arshhh26/LinguaLabel.git
   cd LinguaLabel
   ```

2. **Create a Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Set Up Local Configuration**:
   ```bash
   cp .env.example .env
   # Edit .env and supply your local development secrets
   ```

5. **Run the Test Suite**:
   ```bash
   PYTHONPATH=. pytest -v
   ```
   Ensure all 209 tests pass before making changes.

---

## Guidelines for Pull Requests

- **Preserve Test Coverage**: Every new feature or bugfix should include corresponding unit or integration tests in `tests/`.
- **Maintain Compatibility**: All 209 existing regression tests must pass cleanly.
- **Strict Human-in-the-Loop Contract**: AI predictions must remain strictly advisory; never automatically commit model outputs to disk without human confirmation.
- **Privacy First**: Never log sentence texts, personal data, or uploaded datasets in telemetry or analytics.
- **Code Style**: Follow PEP 8 guidelines for Python code (`snake_case` for functions and variables, `CamelCase` for classes).

---

## Reporting Issues

If you encounter a bug or have a feature request:
1. Search existing issues to ensure it has not already been reported.
2. Provide a clear description, reproduction steps, expected behavior, and system environment details (OS, Python version, Streamlit version).

