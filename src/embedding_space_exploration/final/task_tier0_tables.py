"""Tier 0 result tables, generated from the calibration measurements.

Every number the results section states comes from here, so the prose cannot
drift from the run that produced it. One file per table, written as MyST with a
caption and a label, and pulled into ``results.md`` with ``{include}``.

Conditions are summarised across the three replicate seeds. A verdict is only
reported when all three seeds agree; where they do not the cell reads ``mixed``,
which is a finding rather than something to average away.
"""

import pandas as pd
import pytask

from embedding_space_exploration.battery.cluster_tendency import verdict_label
from embedding_space_exploration.config import CALIBRATION_DIR, DOCUMENTS

TABLES = DOCUMENTS / "tables"

# The arm every clustering check is read on. `raw` is reported alongside it only
# in the cone table, the one place the two arms could plausibly disagree.
PRIMARY = "spherical"


@pytask.task(
    kwargs={
        "measurements": CALIBRATION_DIR / "measurements.parquet",
        "produces": {
            "separation": TABLES / "tier0_separation.md",
            "continuum": TABLES / "tier0_continuum.md",
            "confound": TABLES / "tier0_confound.md",
            "rankme": TABLES / "tier0_rankme.md",
            "cone": TABLES / "tier0_cone.md",
        },
    }
)
def task_tier0_tables(measurements, produces):
    """Write the five Tier 0 result tables."""
    frame = pd.read_parquet(measurements)
    frame["condition"] = frame["cell_id"].str.rsplit("-s", n=1).str[0]
    primary = _derived_verdict(frame[frame["scaling"] == PRIMARY])

    _write(
        _separation_table(primary),
        produces["separation"],
        "Null-gate behaviour against planted cluster separation, in units of "
        "within-cluster SD. Four clusters, 2,000 patients, 128 ambient "
        "dimensions, intrinsic dimension 16. `k_at_max_margin` and `chosen_k` "
        "list all three replicate seeds; the true k is 4.",
        "tbl-tier0-separation",
    )
    _write(
        _continuum_table(primary),
        produces["continuum"],
        "Discrete blobs against a continuum through the same waypoints, at "
        "matched separation. The continuum has no partition to find, so any "
        "verdict other than CONTINUOUS is a false positive.",
        "tbl-tier0-continuum",
    )
    _write(
        _confound_table(primary),
        produces["confound"],
        "Confound orientation (C1, principal-component regression) beside "
        "confound decodability (linear and nonlinear probes) at matched "
        "loadings. `axis` writes the nuisance along one direction; `radial` "
        "writes it as a radius in a plane at uniform angle, so no linear "
        "direction carries it. `decoy_floor` is the highest value reached by a "
        "covariate that loads on nothing.",
        "tbl-tier0-confound",
    )
    _write(
        _rankme_table(primary),
        produces["rankme"],
        "RankMe (A5) against the planted intrinsic dimension. The upper block "
        "varies intrinsic dimension at fixed ambient noise 0.1; the lower "
        "block varies ambient noise at fixed intrinsic dimension 16. Ambient "
        "width is 128 throughout. `error` is RankMe minus the planted "
        "intrinsic dimension.",
        "tbl-tier0-rankme",
    )
    _write(
        _cone_table(frame),
        produces["cone"],
        "Anisotropy with no planted structure, on both preprocessing arms. "
        "Every cell is a single unstructured cloud pushed into a cone of the "
        "stated strength, so any verdict other than CONTINUOUS is the "
        "anisotropy alone manufacturing a partition.",
        "tbl-tier0-cone",
    )


# ======================================================================================
# HELPER FUNCTIONS
# ======================================================================================


def _separation_table(primary):
    """The gate's operating curve against planted separation."""
    rows = primary[
        primary["condition"].str.startswith("separation-")
        | (primary["condition"] == "base")
    ]
    return (
        rows.groupby("separation")
        .agg(
            verdict=("gate_verdict", _verdict),
            margin=("gate_max_headroom_margin", "mean"),
            k_at_max_margin=("gate_k_at_max_margin", _k_spread),
            ari_at_true_k=("ari_at_true_k", "mean"),
            chosen_k=("chosen_k", _k_spread),
            silhouette=("silhouette", "mean"),
        )
        .round(3)
        .reset_index()
    )


def _continuum_table(primary):
    """Blobs against a continuum through the same waypoints (D6)."""
    rows = primary[primary["separation"].isin([2.0, 3.0, 6.0])]
    rows = rows[
        rows["condition"].str.startswith(("separation-", "continuum-"))
        | (rows["condition"] == "base")
    ]
    return (
        rows.groupby(["separation", "structure"])
        .agg(
            verdict=("gate_verdict", _verdict),
            margin=("gate_max_headroom_margin", "mean"),
            k_at_max_margin=("gate_k_at_max_margin", _k_spread),
            prediction_strength=("prediction_strength_at_chosen_k", "mean"),
            silhouette=("silhouette", "mean"),
        )
        .round(3)
        .reset_index()
    )


def _confound_table(primary):
    """Orientation beside decodability, across matched loadings."""
    rows = primary[
        primary["confound_orientation"].isin(["axis", "radial"])
        & (primary["confound_cluster_coupling"] == 0)
    ]
    return (
        rows.groupby(["confound_orientation", "confound_strength"])
        .agg(
            pcr_leading_pc_r2=("pcr_confound_leading_pc_r2", "mean"),
            pcr_variance_weighted_r2=("pcr_confound_variance_weighted_r2", "mean"),
            probe_linear_r2=("decode_confound_r2_linear", "mean"),
            probe_nonlinear_r2=("decode_confound_r2_nonlinear", "mean"),
            decoy_floor=("decode_decoy_max_r2_nonlinear", "mean"),
            ari_at_true_k=("ari_at_true_k", "mean"),
        )
        .round(3)
        .reset_index()
    )


def _rankme_table(primary):
    """RankMe against planted dimensionality, and against ambient noise."""
    by_dimension = (
        primary[
            primary["condition"].str.startswith("intrinsic-dim-")
            | (primary["condition"] == "base")
        ]
        .groupby("intrinsic_dim")
        .agg(rankme=("rankme", "mean"), error=("rankme_minus_intrinsic_dim", "mean"))
        .round(2)
        .reset_index()
        .assign(varied="intrinsic dimension")
        .rename(columns={"intrinsic_dim": "value"})
    )
    by_noise = (
        primary[
            primary["condition"].str.startswith("noise-")
            | (primary["condition"] == "base")
        ]
        .groupby("noise")
        .agg(rankme=("rankme", "mean"), error=("rankme_minus_intrinsic_dim", "mean"))
        .round(2)
        .reset_index()
        .assign(varied="ambient noise SD")
        .rename(columns={"noise": "value"})
    )
    return pd.concat([by_dimension, by_noise])[["varied", "value", "rankme", "error"]]


def _cone_table(frame):
    """Anisotropy with no structure, on both arms (the Corpas trap)."""
    rows = frame[
        frame["condition"].str.startswith("cone-only-")
        | (frame["condition"] == "separation-0.0")
    ]
    return (
        rows.groupby(["anisotropy", "scaling"])
        .agg(
            verdict=("gate_verdict", _verdict),
            margin=("gate_max_headroom_margin", "mean"),
            measured_cosine=("mean_cosine_to_centroid", "mean"),
            anisotropy_error=("anisotropy_error", "mean"),
        )
        .round(3)
        .reset_index()
    )


def _derived_verdict(frame):
    """Recompute the gate label from the recorded margin, not the stored string.

    The harness writes the label it produced, which pins ``NULL_MARGIN_THRESHOLD``
    to the value set when that run executed. Deriving it here is what makes the
    claim in ``config`` true in practice -- revising the constant reaches the
    paper's tables without a re-run, because the margin it is compared against
    was recorded beside it.
    """
    labels = [
        verdict_label(margin, beats_null=bool(beats)) if ran else None
        for ran, margin, beats in zip(
            frame["gate_ran"],
            frame["gate_max_headroom_margin"],
            frame["gate_any_k_beats_null"],
            strict=True,
        )
    ]
    return frame.assign(gate_verdict=labels)


def _verdict(series):
    """The verdict when the seeds agree, ``mixed`` when they do not."""
    unique = set(series.dropna())
    if not unique:
        return "-"
    return unique.pop() if len(unique) == 1 else "mixed"


def _k_spread(series):
    """Every seed's arg-max k, so instability is visible rather than averaged."""
    values = sorted(int(v) for v in series.dropna())
    return ", ".join(str(v) for v in values) if values else "-"


def _write(frame, path, caption, label):
    """One MyST table with a caption and a cross-reference label."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f":::{{table}} {caption}\n:name: {label}\n\n"
        f"{frame.to_markdown(index=False)}\n:::\n"
    )
