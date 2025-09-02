# Contributing to Lost-in-the-Middle Analyzer

First off, thank you for taking the time to contribute! We welcome contributions of all kinds — bug reports, feature requests, documentation improvements, and code changes. This guide will help you get started.

## Table of contents
- Code of Conduct
- Ways to contribute
- Reporting bugs
- Requesting features
- Development setup
- Running the app
- Testing and linting
- Coding guidelines
- Submitting a pull request (PR)
- Getting help

## Code of Conduct
Please be kind and respectful. We aim to foster an open and welcoming environment for everyone. If you experience or observe unacceptable behavior, please open a confidential issue or contact a maintainer.

## Ways to contribute
- Report a bug
- Suggest a feature or enhancement
- Improve documentation (README, examples, comments)
- Triage issues (confirm repro, suggest labels, propose scopes)
- Contribute code (small fixes to larger features)

If you’re unsure where to start, check issues labeled “good first issue” or “help wanted”.

## Reporting bugs
Before filing a bug:
- Search existing issues to avoid duplicates.
- Try the latest main branch if possible.

When opening a bug report (use the Bug Report template):
- Describe the problem clearly and concisely
- Include exact steps to reproduce
- Provide expected vs. actual behavior
- Share environment details (OS, Python, Streamlit version)
- Include logs, stack traces, or screenshots if relevant

## Requesting features
When proposing a feature (use the Feature Request template):
- Explain the problem or use case
- Describe the proposed solution
- List alternatives considered
- Provide additional context (links, prior art)
- Indicate if you’d be willing to help implement it

## Development setup
Prerequisites:
- Python 3.11+
- Git

Steps:
1. Fork the repository and clone your fork
2. Create and activate a virtual environment
   - Linux/macOS: `python -m venv .venv && source .venv/bin/activate`
   - Windows: `.venv\Scripts\activate`
3. Install dependencies: `pip install -r requirements.txt`
4. (Optional) Copy `.env.example` to `.env` and set API keys if you’ll use external providers

## Running the app
- Streamlit UI: `streamlit run app.py`
- Docker: see README’s Docker and Docker Compose sections

## Testing and linting
- If tests exist, run them with `pytest` (or `python -m pytest`).
- Keep additions small and focused; include tests where meaningful.
- Follow PEP 8. Prefer type hints where practical. Run a linter (e.g., `ruff` or `flake8`) if available.

## Coding guidelines
- Aim for clear, well‑documented functions.
- Keep public APIs stable; propose breaking changes via an issue first.
- For new methods or models, provide docstrings and small examples if possible.
- Avoid large PRs. If a change is big, propose a design in an issue first.

## Submitting a pull request (PR)
1. Create a feature branch: `git checkout -b feat/short-description` (or `fix/...`)
2. Commit with clear messages (Conventional Commits are welcome):
   - `feat(methods): add hybrid RAG variant`
   - `fix(evaluation): correct EM calculation` 
3. Push and open a PR against the `main` branch.
4. Fill out the PR description:
   - What and why
   - How it was implemented
   - Screenshots or logs if UI/behavior changed
   - Any breaking changes or migration notes
5. Ensure CI/tests pass. Address review feedback promptly and kindly.

### PR checklist
- [ ] I ran the app/tests locally
- [ ] I updated docs/README where relevant
- [ ] I added/updated tests when meaningful
- [ ] I considered backward compatibility

## Getting help
- Open an issue with the "question" label for general help.
- If you’re stuck on setup, include your OS, Python version, and console output.

We’re excited to collaborate with you. Thanks again for helping improve Lost‑in‑the‑Middle Analyzer!