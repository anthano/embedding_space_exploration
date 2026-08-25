"""All the general configuration of the project."""

from pathlib import Path

SRC: Path = Path(__file__).parent.resolve()
ROOT: Path = SRC.joinpath("..", "..").resolve()

BLD: Path = ROOT.joinpath("bld").resolve()

DOCUMENTS: Path = ROOT.joinpath("documents").resolve()

# Per-representation artifacts (embeddings, clusters, ...) under `MODELS / {key}`.
MODELS_DIR: Path = BLD.joinpath("models").resolve()
