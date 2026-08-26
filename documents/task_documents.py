"""Tasks for compiling the paper."""

import shutil
import subprocess
from pathlib import Path

import pytask

from embedding_space_exploration.config import DOCUMENTS, ROOT

for fmt, produces in {
    "pdf": ROOT / "paper.pdf",
    "html": ROOT / "_build" / "html" / "index.html",
}.items():

    @pytask.task(
        id=f"paper-{fmt}",
        kwargs={
            "sections": {
                name: DOCUMENTS / f"{name}.md"
                for name in ("introduction", "methods", "results", "discussion")
            },
            "tables": {
                name: DOCUMENTS / "tables" / f"tier0_{name}.md"
                for name in ("separation", "continuum", "confound", "rankme", "cone")
            },
        },
    )
    def task_compile_paper(
        sections: dict[str, Path],
        tables: dict[str, Path],
        paper_md: Path = DOCUMENTS / "paper.md",
        myst_yml: Path = ROOT / "myst.yml",
        refs: Path = DOCUMENTS / "refs.bib",
        produces: Path = produces,
    ) -> None:
        """Compile the paper from MyST Markdown using Jupyter Book 2.0."""
        fmt = produces.suffix.lstrip(".")
        jupyter_path = shutil.which("jupyter")
        subprocess.run(
            (jupyter_path, "book", "build", f"--{fmt}"),
            check=True,
            cwd=ROOT.absolute(),
        )
        if fmt == "pdf":
            build_pdf = ROOT / "_build" / "exports" / "paper.pdf"
            shutil.copy(build_pdf, produces)
