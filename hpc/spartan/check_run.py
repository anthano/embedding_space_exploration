#!/usr/bin/env python
"""Verify a finished extraction array before the matrices are copied home.

Run on Spartan, from the repo root, after the array reports done::

    cd $REPO/hpc/spartan && pixi run python check_run.py

The rsync is the cheap half. What is expensive is discovering three weeks into
the battery that one cell's rows are in a different order from the other
fifteen, because every CKA and every paired contrast joins these matrices
positionally and a misalignment is silent all the way to a figure.

Everything here is a *cross-check between files the run already wrote*, not a
re-derivation: the journal is deleted only on a clean finish, `extraction.json`
records what the forward pass actually did, and sixteen cells at one anchor must
agree about their anchor index. So the checks cost seconds and need neither a
GPU nor the model weights.

Exit status is 0 when nothing failed, 1 otherwise. WARN never fails the run --
it marks a number worth reading rather than a broken file.
"""

import argparse
import json
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

ANCHORS = ("lastevent", "perlabel-scout")
FAMILIES = ("gpt", "llama")
CONTEXTS = (512, 1024, 2048, 4096)
POOLINGS = ("last", "mean")

# Rows sampled for the pairwise comparisons. The point of those is to catch a
# matrix that is a *copy* of another, which shows at any sample size; a full
# 14k x 14k neighbour computation would buy nothing for the memory.
SAMPLE = 1500
RNG = np.random.default_rng(0)

results = []


def record(level, name, detail):
    """Log one check outcome. ``level`` is PASS, WARN or FAIL."""
    results.append((level, name, detail))
    print(f"  [{level:4}] {name}: {detail}", flush=True)


def expected_cells():
    """The 16 cell ids one anchor level should carry, from the registry if present."""
    try:
        from embedding_space_exploration import registry

        return tuple(
            sorted(
                cell.id
                for cell in registry.CELLS.values()
                if cell.family in FAMILIES and cell.context in CONTEXTS
            )
        )
    except Exception:
        return tuple(
            sorted(
                f"{family}-base-{context}-{pooling}"
                for family in FAMILIES
                for context in CONTEXTS
                for pooling in POOLINGS
            )
        )


def load_cell(directory):
    """One cell's matrix, keys and provenance, or ``None`` when incomplete."""
    matrix_path = directory / "embeddings.parquet"
    record_path = directory / "extraction.json"
    if not matrix_path.exists() or not record_path.exists():
        return None
    frame = pd.read_parquet(matrix_path)
    dims = [column for column in frame.columns if column.startswith("dim_")]
    return {
        "id": directory.name,
        "keys": frame[["person_id", "cutoff"]],
        "matrix": frame[dims].to_numpy(dtype=np.float32),
        "record": json.loads(record_path.read_text()),
        "path": matrix_path,
    }


# ======================================================================================
# Checks
# ======================================================================================


def check_journals(anchor_dir):
    """A surviving journal is the run's own signal that a cell never finished.

    ``extract_resumable`` deletes ``_journal/`` as its last act, after it has
    verified the journal holds exactly as many rows as the index. So a journal
    that is still on disk means that element was killed, and its matrix is
    either absent or left over from an earlier attempt.
    """
    journals = sorted(anchor_dir.glob("cells/*/_journal"))
    if journals:
        record(
            "FAIL",
            "journals cleared",
            f"{len(journals)} cell(s) still hold a journal, so they never "
            f"completed: {', '.join(path.parent.name for path in journals)}",
        )
    else:
        record("PASS", "journals cleared", "no _journal directories remain")


def check_inventory(anchor_dir, cells):
    """Every expected cell directory exists and carries all three files."""
    wanted = set(expected_cells())
    found = {cell["id"] for cell in cells}
    missing = sorted(wanted - found)
    extra = sorted(found - wanted)
    if missing:
        record(
            "FAIL", "cell inventory", f"missing {len(missing)}: {', '.join(missing)}"
        )
    else:
        record("PASS", "cell inventory", f"all {len(wanted)} cells present")
    if extra:
        record("WARN", "unexpected cells", ", ".join(extra))
    for cell in cells:
        truncation = cell["path"].parent / "truncation.parquet"
        if not truncation.exists():
            record("FAIL", f"{cell['id']} truncation.parquet", "missing")


def check_alignment(anchor, cells):
    """Every cell at an anchor must carry the same anchors, in the same order.

    This is the check that justifies the script. Nothing downstream re-joins on
    ``person_id``: CKA, the paired contrasts and the battery all read row *i* of
    two matrices as the same patient. A cell whose index differs -- a stale
    matrix from a run at a different anchor, a resume against a changed index --
    produces numbers rather than an error.
    """
    reference = cells[0]
    keys = reference["keys"].reset_index(drop=True)
    counts = {cell["id"]: len(cell["keys"]) for cell in cells}
    if len(set(counts.values())) != 1:
        spread = ", ".join(f"{name}={n:,}" for name, n in sorted(counts.items()))
        record("FAIL", f"{anchor} row counts", f"cells disagree: {spread}")
    else:
        record("PASS", f"{anchor} row counts", f"all cells {len(keys):,} rows")

    mismatched = [
        cell["id"]
        for cell in cells[1:]
        if not cell["keys"].reset_index(drop=True).equals(keys)
    ]
    if mismatched:
        record(
            "FAIL",
            f"{anchor} anchor alignment",
            f"{len(mismatched)} cell(s) differ from {reference['id']}: "
            f"{', '.join(mismatched)}",
        )
    else:
        record("PASS", f"{anchor} anchor alignment", "identical (person_id, cutoff)")

    unsorted = []
    for cell in cells:
        cell_keys = cell["keys"].reset_index(drop=True)
        ordered = cell_keys.sort_values(
            ["person_id", "cutoff"], kind="stable", na_position="first"
        ).reset_index(drop=True)
        if not ordered.equals(cell_keys):
            unsorted.append(cell["id"])
    if unsorted:
        record(
            "FAIL",
            f"{anchor} anchor order",
            f"{len(unsorted)} cell(s) are not in `ordered_index` order, so a "
            f"resume would land on different anchors than a fresh run: "
            f"{', '.join(unsorted)}",
        )
    else:
        record("PASS", f"{anchor} anchor order", "sorted by person_id, cutoff")

    duplicates = int(keys.duplicated().sum())
    if duplicates:
        record("FAIL", f"{anchor} anchor uniqueness", f"{duplicates:,} duplicate keys")
    else:
        record("PASS", f"{anchor} anchor uniqueness", "no duplicate anchors")


def check_provenance(anchor, cells):
    """`extraction.json` says what the forward pass did. Read it, do not assume."""
    for cell in cells:
        payload = cell["record"]
        if payload.get("cell_id") != cell["id"]:
            record(
                "FAIL",
                f"{cell['id']} record identity",
                f"extraction.json names {payload.get('cell_id')!r}",
            )
        if payload.get("truncation_side") != "left":
            record(
                "FAIL",
                f"{cell['id']} truncation side",
                f"{payload.get('truncation_side')!r}, not 'left' -- the matrix "
                "describes the patient's distant past, not the anchor",
            )
        if payload.get("device") != "cuda":
            record(
                "WARN",
                f"{cell['id']} device",
                f"ran on {payload.get('device')!r}; devices do not agree "
                "bit-for-bit, so this cell is not comparable to the CUDA ones",
            )
        if payload.get("n_anchors") != len(cell["keys"]):
            record(
                "FAIL",
                f"{cell['id']} record row count",
                f"extraction.json says {payload.get('n_anchors'):,}, parquet holds "
                f"{len(cell['keys']):,}",
            )
        empty = payload.get("n_empty_histories", 0)
        if empty:
            record(
                "WARN",
                f"{cell['id']} empty histories",
                f"{empty:,} anchors embedded a single PAD token (a real vector "
                "that means nothing); legitimate a few at a time, never in bulk",
            )

    for field in ("n_anchors_dropped", "batch_size", "n_patients"):
        values = {cell["record"].get(field) for cell in cells}
        if len(values) != 1:
            record(
                "FAIL",
                f"{anchor} {field} agreement",
                f"cells disagree: {sorted(str(v) for v in values)}",
            )
    dropped = cells[0]["record"].get("n_anchors_dropped")
    record(
        "PASS",
        f"{anchor} cohort",
        f"{cells[0]['record'].get('n_patients'):,} patients, "
        f"{cells[0]['record'].get('n_anchors'):,} anchors, {dropped:,} dropped as "
        "absent from the extract",
    )


def check_truncation_trend(anchor, cells):
    """Longer context must see more of the record. If it does not, the cut is wrong.

    The failure this catches is the one the extraction module's docstring calls
    load-bearing: under right-truncation every context reads the same opening
    tokens, so `truncated_share` and `median_covered` stop moving with context
    and P1's null is built into the design rather than measured.
    """
    for family in FAMILIES:
        for pooling in POOLINGS:
            series = []
            for context in CONTEXTS:
                cell_id = f"{family}-base-{context}-{pooling}"
                match = [cell for cell in cells if cell["id"] == cell_id]
                if match:
                    series.append((context, match[0]["record"]))
            if len(series) < 2:
                continue
            shares = [payload["truncated_share"] for _, payload in series]
            covered = [payload["median_covered"] for _, payload in series]
            label = f"{anchor} {family}-{pooling} truncation trend"
            detail = "  ".join(
                f"{context}:{share:.1%}/{cov:.2f}"
                for (context, _), share, cov in zip(
                    series, shares, covered, strict=True
                )
            )
            monotone = all(a >= b for a, b in pairwise(shares)) and all(
                a <= b for a, b in pairwise(covered)
            )
            record("PASS" if monotone else "FAIL", label, detail)


def check_numerics(anchor, cells):
    """The matrix itself: finite, non-degenerate, and not a copy of another cell."""
    samples = {}
    for cell in cells:
        matrix = cell["matrix"]
        name = cell["id"]
        bad = int((~np.isfinite(matrix)).sum())
        if bad:
            record("FAIL", f"{name} finite", f"{bad:,} NaN/Inf entries")
        zero_rows = int((np.abs(matrix).sum(axis=1) == 0).sum())
        if zero_rows:
            record("FAIL", f"{name} zero rows", f"{zero_rows:,} all-zero vectors")
        unique = len(np.unique(matrix, axis=0))
        if unique < len(matrix):
            record(
                "WARN",
                f"{name} distinct vectors",
                f"{len(matrix) - unique:,} duplicate rows -- expected only where "
                "two anchors truncate to the same window",
            )
        norms = np.linalg.norm(matrix, axis=1)
        record(
            "PASS",
            f"{name} scale",
            f"dim={matrix.shape[1]}  norm median {np.median(norms):.2f} "
            f"[{norms.min():.2f}, {norms.max():.2f}]",
        )
        take = RNG.choice(len(matrix), size=min(SAMPLE, len(matrix)), replace=False)
        samples[name] = matrix[np.sort(take)]

    # Distinctness. Each of these pairs *must* differ, and an exact match means a
    # matrix was written from the wrong tensor -- the cheapest possible detector
    # for a copied or mislabelled cell.
    pairs = []
    for family in FAMILIES:
        for context in CONTEXTS:
            pairs.append(
                (f"{family}-base-{context}-last", f"{family}-base-{context}-mean")
            )
        for pooling in POOLINGS:
            for left, right in pairwise(CONTEXTS):
                pairs.append(
                    (
                        f"{family}-base-{left}-{pooling}",
                        f"{family}-base-{right}-{pooling}",
                    )
                )
    for context in CONTEXTS:
        pairs.append((f"gpt-base-{context}-last", f"llama-base-{context}-last"))

    identical = []
    for left, right in pairs:
        if left in samples and right in samples:
            a, b = samples[left], samples[right]
            if a.shape == b.shape and np.array_equal(a, b):
                identical.append(f"{left} == {right}")
    if identical:
        record("FAIL", f"{anchor} cell distinctness", "; ".join(identical))
    else:
        record(
            "PASS",
            f"{anchor} cell distinctness",
            f"all {len(pairs)} pooling/context/family pairs differ",
        )


def check_throughput(anchor, cells):
    """Report anchors/s per key. The README asks for this number; it sizes perlabel."""
    rows = []
    for family in FAMILIES:
        for context in CONTEXTS:
            match = [c for c in cells if c["id"] == f"{family}-base-{context}-last"]
            if not match:
                continue
            payload = match[0]["record"]
            seconds = payload.get("seconds") or 0.0
            rate = payload["n_anchors"] / seconds if seconds else float("nan")
            slug = f"{family}-base-{context}"
            rows.append(
                f"    {slug:<17} {seconds / 60:7.1f}m  {rate:7.2f} anchors/s  "
                f"resumed_from={payload.get('resumed_from_row', 0):,}"
            )
    if rows:
        print(f"\n  throughput at {anchor} (per extraction key):")
        print("\n".join(rows), flush=True)


# ======================================================================================


def main(argv=None):
    """Check every anchor level under ``bld/tier1/{dataset}``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ehrshot")
    parser.add_argument(
        "--tier1",
        type=Path,
        default=None,
        help="bld/tier1 directory; defaults to the registry's",
    )
    parser.add_argument("--anchors", nargs="+", default=list(ANCHORS))
    args = parser.parse_args(argv)

    tier1 = args.tier1
    if tier1 is None:
        from embedding_space_exploration.registry import TIER1_DIR

        tier1 = TIER1_DIR

    for anchor in args.anchors:
        anchor_dir = Path(tier1) / args.dataset / anchor
        print(f"\n=== {anchor_dir} ===", flush=True)
        if not anchor_dir.exists():
            record("FAIL", f"{anchor} present", "no such directory")
            continue
        check_journals(anchor_dir)
        cells = [
            cell
            for cell in (load_cell(d) for d in sorted(anchor_dir.glob("cells/*")))
            if cell is not None
        ]
        if not cells:
            record("FAIL", f"{anchor} matrices", "no complete cell directories")
            continue
        check_inventory(anchor_dir, cells)
        check_alignment(anchor, cells)
        check_provenance(anchor, cells)
        check_truncation_trend(anchor, cells)
        check_numerics(anchor, cells)
        check_throughput(anchor, cells)

    failed = [item for item in results if item[0] == "FAIL"]
    warned = [item for item in results if item[0] == "WARN"]
    print(
        f"\n{len(results)} checks: {len(results) - len(failed) - len(warned)} pass, "
        f"{len(warned)} warn, {len(failed)} fail",
        flush=True,
    )
    if failed:
        print("\nDo not copy home until these are resolved:", flush=True)
        for _, name, detail in failed:
            print(f"  - {name}: {detail}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
