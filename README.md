# MLOps Assignment 4 — ML CI Pipeline

## Overview
This project implements a GitHub Actions CI pipeline for an Arabic Handwritten Characters classification model built with PyTorch.

## Pipeline Steps
| Step | Description |
|------|-------------|
| Checkout | Clones the repository into the runner |
| Set up Python | Configures Python 3.11 |
| Install Dependencies | Installs project requirements from requirements.txt |
| Linter Check | Runs `ruff` on `train.py` to enforce code style |
| Model Dry Test | Verifies the PyTorch environment is importable |
| Upload Artifact | Uploads this README as a GitHub artifact named `project-doc` |

## Trigger Logic
- **Push to any branch except `main`**: pipeline runs automatically (uses `branches-ignore: [main]`)
- **Pull requests**: pipeline also runs on all PRs
- **Push to `main`**: intentionally excluded — `main` holds only stable, reviewed code

## Reflection

Moving from manual execution to an automated CI pipeline has improved the reliability of the workflow. 
Now every code push (outside of the main branch) is automatically checked for syntax errors via the linter, and the environment setup is verified with a dry run of the PyTorch import. This ensures that faulty code is caught early before it can be merged into the main production branch.