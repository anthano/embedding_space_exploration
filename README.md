# Embedding Space Exploration

Exploration of embedding spaces.

## Getting started

[Pixi](https://pixi.sh/) is the only prerequisite. To run the full pipeline — data
cleaning, analysis, figures and tables, and the compiled paper:

```bash
pixi run pytask
```

To view the paper in a browser with live reload:

```bash
pixi run view-paper
```

## Development

```bash
pixi run tests    # run the test suite
pixi run prek     # run the pre-commit hooks on all files
```

Install the pre-commit hooks once per machine with `pixi run prek install`.

## Credits

Built on the
[econ-project-templates](https://github.com/OpenSourceEconomics/econ-project-templates)
by Hans-Martin von Gaudecker.
