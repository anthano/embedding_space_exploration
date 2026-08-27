"""EHRSHOT's labels, splits, published results and shipped reference embeddings.

The data layer for Tier 1. Everything here is a *read* of the EHRSHOT release as
downloaded -- nothing derives a cohort, a split or a label of its own. That is the
point: [[Study Design Freeze]] section 2 Rule 1 forbids redefining the EHRSHOT
cohort, because the published-baseline oracle is the cheapest correctness check
available on the whole pipeline and a re-partition silently breaks it.

Two ids are normalised at this boundary. EHRSHOT calls the patient key
``patient_id`` in the label files and ``omop_person_id`` in the split map; both
come back as ``person_id`` so that ``data_management.splits`` applies unchanged.

This module deliberately does **not** import ``meds_reader`` or any model. The
timeline path lives in ``data_management.timeline`` and needs the extract; the
probe and the oracle need only what is here, which is why they run on a laptop.
"""

import pickle

import numpy as np
import pandas as pd

from embedding_space_exploration.config import EHRSHOT_ROOT

ASSETS = EHRSHOT_ROOT / "EHRSHOT_ASSETS"
BENCHMARK_DIR = ASSETS / "benchmark"
SPLITS_CSV = ASSETS / "splits" / "person_id_map.csv"
RESULTS_DIR = ASSETS / "results"
REFERENCE_FEATURES = ASSETS / "features" / "clmbr_features.pkl"

# The 14 binary benchmark tasks. `chexpert` is excluded: it is a multi-label
# chest X-ray task scored differently, so it is not an apples-to-apples row
# against the published CLMBR number and would not belong in the oracle.
TASKS = (
    "guo_icu",
    "guo_los",
    "guo_readmission",
    "lab_anemia",
    "lab_hyperkalemia",
    "lab_hypoglycemia",
    "lab_hyponatremia",
    "lab_thrombocytopenia",
    "new_acutemi",
    "new_celiac",
    "new_hyperlipidemia",
    "new_hypertension",
    "new_lupus",
    "new_pancan",
)

# Which row of `results/{task}/all_results.csv` is the comparison target: the
# CLMBR featuriser, logistic-regression head, full training set (k == -1). This
# is the row [[Wornow2025]] reports and the one section 6 pins our protocol to.
PUBLISHED_MODEL = "clmbr"
PUBLISHED_HEAD = "lr_lbfgs"
PUBLISHED_K = -1


def load_labels(task):
    """Load one task's labels as ``person_id``, ``prediction_time``, ``y``.

    Every benchmark task is scored as a binary AUROC. The ``guo_*`` and ``new_*``
    tasks ship boolean values directly. The ``lab_*`` tasks ship four ordinal
    severity classes (0 = normal .. 3 = severe) but EHRSHOT scores them
    normal-versus-abnormal, collapsing value >= 1 to 1; we mirror that, because
    otherwise our AUROC is not the quantity the published number reports.

    Args:
        task: One of ``TASKS``.

    Returns:
        Frame with ``person_id``, ``prediction_time``, ``y``, ``label_type`` and
        ``task``, in the file's own row order.
    """
    labels = pd.read_csv(BENCHMARK_DIR / task / "labeled_patients.csv")
    labels = labels.rename(columns={"patient_id": "person_id"})
    labels["prediction_time"] = pd.to_datetime(labels["prediction_time"])
    if labels["label_type"].iloc[0] == "boolean":
        # Booleans arrive as bools or as the strings "True"/"False" depending on
        # the task file; coerce both.
        labels["y"] = labels["value"].map({True: 1, False: 0, "True": 1, "False": 0})
    else:
        labels["y"] = (labels["value"].astype(int) >= 1).astype(int)
    labels["task"] = task
    return labels[["person_id", "prediction_time", "y", "label_type", "task"]]


def load_splits():
    """EHRSHOT's native patient-level train/val/test partition.

    Returns:
        Frame with ``person_id`` and ``split``, in the contract
        ``data_management.splits`` expects.
    """
    splits = pd.read_csv(SPLITS_CSV)
    return splits.rename(columns={"omop_person_id": "person_id"})[
        ["person_id", "split"]
    ]


def load_published_auroc(task):
    """The published test AUROC this pipeline is checked against.

    Averaged over the replicates EHRSHOT ships for the CLMBR + LR + full-data
    row. Returns ``None`` when the file or the matching row is absent, so a run
    still reports our number rather than failing for want of a target.

    Args:
        task: One of ``TASKS``.

    Returns:
        Float AUROC, or ``None``.
    """
    path = RESULTS_DIR / task / "all_results.csv"
    if not path.exists():
        return None
    results = pd.read_csv(path)
    hit = results.loc[
        (results["model"] == PUBLISHED_MODEL)
        & (results["head"] == PUBLISHED_HEAD)
        & (results["k"] == PUBLISHED_K)
        & (results["score"] == "auroc"),
        "value",
    ]
    return float(hit.mean()) if len(hit) else None


def load_reference_features():
    """The CLMBR embedding matrix EHRSHOT ships, with its own index.

    406,379 label-time embeddings, one per unique ``(person_id,
    prediction_time)`` pair across the benchmark. This is the release's *own*
    output of the model our anchor cell reproduces, which is what lets the oracle
    be split in two: the probe can be validated against the published AUROCs on
    these vectors before any extraction exists, and our extraction can then be
    checked against these vectors directly rather than through a downstream
    AUROC that confounds the two.

    ~624 MB as float16. Loaded once and aligned per task via ``align_features``
    rather than re-read, which matters on a 8 GB machine.

    Returns:
        ``(index, matrix)``. ``index`` has ``person_id`` and ``prediction_time``;
        ``matrix`` is ``(n_rows, 768)`` float16, row-aligned to it.
    """
    with REFERENCE_FEATURES.open("rb") as stream:
        payload = pickle.load(stream)
    index = pd.DataFrame(
        {
            "person_id": payload["patient_ids"].astype("int64"),
            "prediction_time": pd.to_datetime(payload["labeling_time"]),
        }
    )
    return index, payload["data_matrix"]


def align_features(index, matrix, labels):
    """Select the feature rows matching ``labels``, in the labels' row order.

    A left join on ``(person_id, prediction_time)``. Label rows with no matching
    vector come back in the mask as ``False`` rather than being dropped silently,
    so a caller decides what an unmatched row means instead of discovering a
    shortened frame later.

    Args:
        index: Index frame from ``load_reference_features``.
        matrix: Feature matrix from ``load_reference_features``.
        labels: Frame with ``person_id`` and ``prediction_time``.

    Returns:
        ``(features, matched)``. ``features`` is ``(n_matched, n_dims)`` in the
        row order of the matched label rows; ``matched`` is a boolean array over
        ``labels`` marking which rows are represented.
    """
    positions = index.assign(_row=np.arange(len(index)))
    merged = labels[["person_id", "prediction_time"]].merge(
        positions, on=["person_id", "prediction_time"], how="left"
    )
    matched = merged["_row"].notna().to_numpy()
    rows = merged.loc[matched, "_row"].to_numpy().astype("int64")
    return np.asarray(matrix[rows]), matched
