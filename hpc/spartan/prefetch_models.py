"""Download every checkpoint the array will need, on a node that has the internet.

Spartan's compute nodes have no outbound network. A job that reaches
``from_pretrained`` uncached does not fail fast -- it stalls on a connection that
will never open, then dies somewhere inside ``huggingface_hub``'s retry loop, at
which point the array element has burned its GPU allocation to produce a
traceback. So the download is a separate, explicit step run on a login node, and
the jobs run with ``HF_HUB_OFFLINE=1`` so a cache miss is an immediate, legible
error instead of a hang.

Run once, after ``pixi install``::

    pixi run prefetch

Idempotent: an already-cached revision is verified and skipped, so re-running it
before a re-submission costs seconds and rules out a whole class of job failure.
"""

import os
import sys

from huggingface_hub import snapshot_download

from embedding_space_exploration.data_management.run_extraction import (
    STAGE_ONE_FAMILIES,
    key_slug,
    keys_for,
)
from embedding_space_exploration.registry import CELLS


def sources(families=STAGE_ONE_FAMILIES):
    """The distinct HuggingFace repos the array's keys resolve to.

    Read off the registry rather than listed, so a family added to the array is
    a family that gets prefetched -- a hand-maintained list here would fail in
    the one place it is expensive to fail.

    Args:
        families: Family names the array will run.

    Returns:
        Dict of ``{key_slug: repo_id}``.
    """
    return {
        key_slug(key): CELLS[members[0]].source for key, members in keys_for(families)
    }


def main():
    """Cache every checkpoint, reporting where and how large."""
    home = os.environ.get("HF_HOME", "(unset -- defaults to ~/.cache/huggingface)")
    print(f"HF_HOME  {home}")
    if "HF_HOME" not in os.environ:
        print(
            "  warning: HF_HOME is unset, so this caches into your home directory.\n"
            "  Home is small and often not readable at speed from compute nodes;\n"
            "  export HF_HOME to project scratch and re-run.",
        )

    failures = []
    for slug, repo in sources().items():
        print(f"\n{slug:20s} {repo}", flush=True)
        try:
            path = snapshot_download(repo)
        except Exception as error:
            print(f"  FAILED: {type(error).__name__}: {error}")
            failures.append((repo, error))
            continue
        print(f"  cached at {path}")

    if failures:
        print(f"\n{len(failures)} of {len(sources())} checkpoints failed to download.")
        print("Gated repos need `huggingface-cli login` with an approved account.")
        return 1
    print(f"\nAll {len(sources())} checkpoints cached. Compute nodes can run offline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
