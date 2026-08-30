r"""Run one extraction key end to end: the entry point a Slurm array element calls.

Deliberately **not** a ``task_`` module. pytask discovers by filename and
registers on import, and an extraction task would put a multi-hour GPU job into
the DAG that every ``pixi run pytask`` then wants to build -- on a laptop, from a
licensed extract that is not there. The cluster runs cells; pytask consumes the
matrices they leave behind. When the battery lands it depends on
``embeddings.parquet`` as an input that already exists, which is the same
contract ``bld/ehrshot`` already has.

One process runs one **extraction key** -- ``(family, size, context)`` -- and
writes every pooling of it, because both readouts come off the same forward pass
(see ``registry.extraction_key``). So the eight GPT and Llama keys produce the
sixteen last-token and mean-pool matrices of the staged sequence's first step,
and the array is eight elements wide, not sixteen.

Resume is on by default and needs no flag: ``extract_resumable`` reads its own
journal, so a job killed at the walltime is re-queued rather than restarted, and
the resumed run is bitwise identical to an uninterrupted one.

Usage::

    # what the array elements run
    python -m embedding_space_exploration.data_management.run_extraction \\
        --anchor lastevent --key-index $SLURM_ARRAY_TASK_ID --device cuda

    # sizing, without loading a model or touching a GPU
    python -m embedding_space_exploration.data_management.run_extraction \\
        --anchor perlabel-scout --plan
"""

import argparse
import sys
import time

from embedding_space_exploration import registry
from embedding_space_exploration.config import BLD
from embedding_space_exploration.data_management import anchors
from embedding_space_exploration.data_management.extraction import (
    BATCH_SIZE,
    extract_resumable,
)

TIMELINE = BLD / "ehrshot" / "patient_timeline.parquet"

# The staged sequence's first step (Build Plan section 1.2): two families, one
# loader, eight keys, sixteen matrices. CKA over these decides whether Mamba and
# Hyena are ever built, so it is the default rather than a convenience.
STAGE_ONE_FAMILIES = ("gpt", "llama")

# Rows between progress lines. Chosen for a Slurm log rather than a terminal:
# frequent enough that a stalled job is visible within minutes, rare enough that
# a 14k-anchor run leaves tens of lines and not thousands.
LOG_EVERY = 500


def keys_for(families=STAGE_ONE_FAMILIES):
    """The extraction keys of ``families``, costliest first.

    Ordered by context descending so that a throttled array (``%4``) starts the
    long cells first. Without it the 512s finish quickly, the 4096s start last,
    and the array's wall clock becomes the sum of two waves instead of the max of
    one.

    Args:
        families: Family names to include.

    Returns:
        Tuple of ``(key, cell_ids)`` pairs, where ``key`` is
        ``registry.extraction_key``'s tuple.
    """
    groups = registry.poolings_by_extraction()
    selected = [(key, members) for key, members in groups.items() if key[0] in families]
    return tuple(sorted(selected, key=lambda item: (-item[0][2], item[0][0])))


def key_slug(key):
    """``('gpt', 'base', 512)`` -> ``'gpt-base-512'``, the CLI's name for a key."""
    return "-".join(str(part) for part in key)


def resolve_key(args):
    """The one ``(key, cell_ids)`` this invocation runs.

    Selected by index or by slug, both resolved against ``keys_for`` so the Slurm
    array and a hand-run command cannot disagree about which element is which.

    Args:
        args: Parsed arguments carrying ``families``, ``key`` and ``key_index``.

    Returns:
        Tuple of ``(key, cell_ids)``.

    Raises:
        SystemExit: If neither selector is given, or it names nothing.
    """
    available = keys_for(tuple(args.families))
    if args.key_index is not None:
        if not 0 <= args.key_index < len(available):
            raise SystemExit(
                f"--key-index {args.key_index} is out of range for "
                f"{len(available)} keys; the array must be 0-{len(available) - 1}"
            )
        return available[args.key_index]
    if args.key is None:
        raise SystemExit("pass one of --key or --key-index")
    for key, members in available:
        if key_slug(key) == args.key:
            return key, members
    raise SystemExit(
        f"--key {args.key!r} is not one of: "
        + ", ".join(key_slug(key) for key, _ in available)
    )


def progress_logger(total, every=LOG_EVERY):
    """A ``progress`` callback that logs rate and ETA to stdout.

    The only instrument the first cluster run has. Throughput here is what turns
    the walltime in the Slurm header from a guess into a measurement, so it is
    printed unconditionally rather than behind a verbosity flag.

    Args:
        total: Number of anchors in the run.
        every: Log at most one line per this many rows.

    Returns:
        Callable taking the number of rows completed.
    """
    started = time.perf_counter()
    state = {"logged": 0}

    def log(done):
        if done < state["logged"] + every and done < total:
            return
        state["logged"] = done
        elapsed = time.perf_counter() - started
        rate = done / elapsed if elapsed else 0.0
        remaining = (total - done) / rate if rate else float("nan")
        print(
            f"  {done:>7,}/{total:,} anchors  {rate:6.2f} anchors/s  "
            f"elapsed {elapsed / 60:6.1f}m  eta {remaining / 60:6.1f}m",
            flush=True,
        )

    return log


def main(argv=None):
    """Parse arguments and run (or plan) one extraction key."""
    args = _parse(argv)
    key, cell_ids = resolve_key(args)

    index = anchors.build_index(args.anchor, timeline=TIMELINE)
    if args.limit:
        # A smoke test, and the only supported way to run a partial cell. It
        # writes to its own anchor level so a truncated matrix can never be
        # mistaken for a complete one -- the same argument that separates
        # `perlabel-scout` from `perlabel`.
        index = index.head(args.limit)

    targets = {
        registry.CELLS[cell_id].pooling: registry.cell_dir(
            args.dataset, args.anchor_dir, cell_id
        )
        for cell_id in cell_ids
    }

    print(f"key      {key_slug(key)}", flush=True)
    print(f"cells    {', '.join(sorted(cell_ids))}", flush=True)
    print(f"anchor   {args.anchor_dir}  ({len(index):,} anchors)", flush=True)
    print(f"outputs  {', '.join(str(path) for path in targets.values())}", flush=True)
    if args.plan:
        return 0

    record = extract_resumable(
        _open_database(),
        index,
        registry.CELLS[cell_ids[0]],
        targets,
        batch_size=args.batch_size,
        device=args.device,
        progress=progress_logger(len(index)),
    )
    print(
        f"done in {record['seconds'] / 60:.1f}m on {record['device']}  "
        f"truncated {100 * record['truncated_share']:.1f}%  "
        f"median covered {record['median_covered']:.2f}",
        flush=True,
    )
    return 0


# ======================================================================================
# HELPER FUNCTIONS
# ======================================================================================


def _open_database():
    """Open the extract, imported here so ``--plan`` needs no ``meds_reader``."""
    from embedding_space_exploration.data_management.timeline import open_database

    return open_database()


def _parse(argv):
    """Build the parser and resolve the anchor's output directory name."""
    parser = argparse.ArgumentParser(
        description="Embed one extraction key at one anchor level.",
    )
    parser.add_argument(
        "--anchor",
        default="lastevent",
        choices=("lastevent", "perlabel-scout", "perlabel", "shared"),
        help="anchor level to build the index from",
    )
    parser.add_argument("--dataset", default="ehrshot")
    parser.add_argument("--key", help="extraction key slug, e.g. gpt-base-512")
    parser.add_argument(
        "--key-index",
        type=int,
        help="index into the key list; this is what SLURM_ARRAY_TASK_ID feeds",
    )
    parser.add_argument("--families", nargs="+", default=list(STAGE_ONE_FAMILIES))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--device",
        default=None,
        help="cuda / cpu / mps; omitted means auto-detect",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="run only the first N anchors, into a 'smoke' anchor directory",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="report what would run and exit, loading no model",
    )
    args = parser.parse_args(argv)
    args.anchor_dir = f"{args.anchor}-smoke{args.limit}" if args.limit else args.anchor
    return args


if __name__ == "__main__":
    sys.exit(main())
