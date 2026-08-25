@.ai-instructions/profiles/tier-b-research.md

# Embedding Space Exploration

## Overview

Research project exploring embedding spaces. Uses pytask for workflow orchestration
and Pixi for environment management. Built on the
[econ-project-templates](https://github.com/OpenSourceEconomics/econ-project-templates).

## Build & Test

```bash
# Run the complete computational pipeline (data -> analysis -> figures/tables -> paper)
pixi run pytask

# Run tests
pixi run tests

# Run a single test file or specific test
pixi run pytest tests/analysis/test_something.py::test_name

# Run pre-commit hooks on all files
pixi run prek

# Build documentation (Jupyter Book 2.0)
pixi run -e docs docs

# View documentation and paper interactively
pixi run -e docs view-docs   # Project documentation
pixi run view-paper          # Paper (HTML with live reload)

# Regenerate the DAG visualization
pixi run -e docs recreate-dag
```

## Architecture

### Workflow Pipeline (pytask)

The project follows a task-based pipeline where each `task_*.py` file defines
computational steps:

1. **Data Management** (`src/embedding_space_exploration/data_management/`)

   - Loads and cleans raw data, saves intermediate formats to `bld/data/`

1. **Analysis** (`src/embedding_space_exploration/analysis/`)

   - Fits models and generates predictions

1. **Final Outputs** (`src/embedding_space_exploration/final/`)

   - Creates publication-ready figures (PNG via Plotly + Kaleido) and tables

1. **Documents** (`documents/task_documents.py`)

   - Compiles paper to PDF and HTML (MyST Markdown via Jupyter Book 2.0)

### Key Configuration

- `src/embedding_space_exploration/config.py`: Central path definitions (`SRC`, `ROOT`,
  `BLD`, `DOCUMENTS`)
- `pyproject.toml`: All tool configurations (Pixi, pytask, Ruff, pytest)
- `myst.yml`: Jupyter Book 2.0 configuration for PDF export (in project root)

### Directory Conventions

- `src/embedding_space_exploration/`: Source code (hand-written)
- `bld/`: Computational outputs (data, models, predictions, figures, tables)
- `_build/`: Document build outputs (HTML site, PDF exports)
- `documents/`: Academic paper sources (MyST Markdown)
- `documents/public/`: Generated figures (intermediate outputs used by documents)
- `documents/tables/`: Generated tables (intermediate outputs used by documents)
- `docs_template/source/`: Documentation of the upstream template (kept for reference)

### pytask Task Pattern

Tasks are discovered by filename pattern `task_*.py`. For iterating over groups:

```python
for group in GROUPS:

    @pytask.task(id=group)
    def task_name(depends_on=..., produces=...): ...
```

## Code Quality

- **Linting/Formatting**: Ruff with an explicit rule selection (see `[tool.ruff.lint]`)
- **Pre-commit**: see `.pre-commit-config.yaml`
- **Docstrings**: Google convention
- **Python version**: 3.14 (requires >=3.14, \<3.15)

## Testing

- Markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.end_to_end`,
  `@pytest.mark.wip`
- Uses `pdbp` as enhanced debugger (`--pdbcls=pdbp:Pdb`)
