import numpy as np
import pandas as pd

from embedding_space_exploration.battery.prep import prepare_matrix
from embedding_space_exploration.splits import fit_mask


def code_coverage(timeline, *, tokenizer):
    """How much of the timeline the model's vocabulary actually recognises.

    Codes the tokenizer does not know are silently dropped (they contribute no
    token), so this is the key check before trusting any embedding: if coverage
    is low, the model barely "sees" the patient. Reported per vocabulary, both
    event-weighted (share of actual events kept) and by unique code.

    Args:
        timeline: Long event frame with a ``code`` column (vocabulary/concept).
        tokenizer: A loaded CLMBRTokenizer.

    Returns:
        One row per vocabulary (plus an ``ALL`` total) with event/code counts and
        coverage fractions, most frequent vocabulary first.
    """
    known = _known_base_codes(tokenizer)
    df = pd.DataFrame({"code": timeline["code"].to_numpy()})
    df["vocabulary"] = df["code"].str.split("/", n=1).str[0]
    df["covered"] = df["code"].isin(known)

    events = df.groupby("vocabulary")["covered"].agg(
        n_events="size", n_events_covered="sum"
    )
    codes = (
        df.drop_duplicates("code")
        .groupby("vocabulary")["covered"]
        .agg(n_codes="size", n_codes_covered="sum")
    )
    report = events.join(codes)
    report = pd.concat([report, report.sum().to_frame("ALL").T])
    report["event_coverage"] = report["n_events_covered"] / report["n_events"]
    report["code_coverage"] = report["n_codes_covered"] / report["n_codes"]

    report.index.name = "vocabulary"
    ordered = report.drop(index="ALL").sort_values("n_events", ascending=False)
    return pd.concat([ordered, report.loc[["ALL"]]]).reset_index()


def truncation_stats(token_lengths):
    """Context-window cap-hit stats from the per-patient tokenised lengths.

    Answers the question the gpt-512-vs-llama-2048 ablation exists to settle --
    *is the context window actually costing us?* All figures are over the
    **untruncated** tokenised length (``get_patient_embeddings`` records it before
    the forward pass truncates), so ``cap_hit_rate`` is the fraction of patients
    whose history the model could not fully see.

    Args:
        token_lengths: Frame from the embedding step: ``person_id``, ``n_tokens``,
            ``context_length``, ``truncated``.

    Returns:
        A single-row DataFrame of cap-hit rate and length percentiles.
    """
    n = token_lengths["n_tokens"].to_numpy()
    context_length = int(token_lengths["context_length"].iloc[0])
    return pd.DataFrame(
        [
            {
                "context_length": context_length,
                "n_patients": len(n),
                "cap_hit_rate": float((n > context_length).mean()),
                "n_tokens_median": float(np.median(n)),
                "n_tokens_p95": float(np.percentile(n, 95)),
                "n_tokens_max": int(n.max()),
            }
        ]
    )


def embedding_checks(embeddings):
    """Mechanical sanity stats on the raw embedding matrix (no clinical meaning).

    Catches the failure modes that matter before clustering: missing/garbage
    values, dead dimensions, and representation collapse (every patient mapped to
    nearly the same vector). This runs on the **raw** model output -- it states
    the anisotropy problem, it does not check whether ``prep.py``'s correction
    fixed it (see ``post_prep_anisotropy`` for that).

    Args:
        embeddings: Frame with ``person_id`` plus ``dim_0 .. dim_N`` columns.

    Returns:
        A single-row DataFrame of summary statistics.
    """
    dims = [c for c in embeddings.columns if c.startswith("dim_")]
    matrix = embeddings[dims].to_numpy(dtype="float64")
    norms = np.linalg.norm(matrix, axis=1)
    rankme = _rankme(matrix)

    stats = {
        "n_patients": len(matrix),
        "n_dims": len(dims),
        "n_nan_rows": int(np.isnan(matrix).any(axis=1).sum()),
        "n_inf_rows": int(np.isinf(matrix).any(axis=1).sum()),
        "n_constant_dims": int((matrix.std(axis=0) < _CONSTANT_TOL).sum()),
        "n_duplicate_vectors": int(len(matrix) - len(np.unique(matrix, axis=0))),
        "norm_min": float(norms.min()),
        "norm_median": float(np.median(norms)),
        "norm_max": float(norms.max()),
        # ~1.0 would mean every patient points the same way (collapse).
        "mean_cosine_to_centroid": _mean_cosine_to_centroid(matrix),
        # Label-free representation-quality proxy (Stage-1 battery): how many
        # dimensions the space *effectively* uses. See _rankme.
        "rankme": rankme,
        "rankme_ratio": rankme / len(dims),
    }
    return pd.DataFrame([stats])


def post_prep_anisotropy(embeddings, split):
    """Mean-cosine-to-centroid after ``prep.prepare_matrix``'s correction.

    ``embedding_checks`` reports anisotropy on the raw model output -- the
    problem statement. This reports the same statistic on the matrix K-means
    actually clusters on: L2-normalise -> PCA (fit on train) -> re-L2-normalise.
    A number close to the raw one means the correction did not do much; a number
    near 0 means the cone has been neutralised.

    Args:
        embeddings: Frame with ``person_id`` plus ``dim_0 .. dim_N`` columns
            (raw, pre-prep).
        split: Frame mapping ``person_id`` to train/val/test (``splits``), used to
            fit the PCA basis on ``train`` only, exactly as
            ``cluster.run_clustering`` does.

    Returns:
        A single-row DataFrame: ``mean_cosine_to_centroid_post_prep``.
    """
    _, matrix = prepare_matrix(
        embeddings, fit_mask=fit_mask(embeddings["person_id"], split)
    )
    return pd.DataFrame(
        [{"mean_cosine_to_centroid_post_prep": _mean_cosine_to_centroid(matrix)}]
    )


# ======================================================================================
# HELPER FUNCTIONS AND CONSTANTS
# ======================================================================================

# A dimension with std below this is treated as constant (dead).
_CONSTANT_TOL = 1e-8

# Smoothing constant for RankMe (Garrido et al. 2023, ICML), matching the paper.
_RANKME_EPS = 1e-7


def _rankme(matrix):
    """Effective rank of the embedding matrix (RankMe; Garrido et al. 2023).

    A one-number, label-free representation-quality proxy: the entropy of the
    (normalised) singular-value spectrum, exponentiated -- a soft count of how many
    dimensions the space *actually* uses. Ranges from 1 (rank-1 collapse: all
    variance on a single axis, the extreme of the anisotropic cone) to
    ``min(n_patients, n_dims)`` (variance spread evenly across all directions). Read
    it comparatively: gpt vs llama vs post-whitening vs (future) post-fine-tune --
    a higher number is a richer, more usable space. Caveat: vision-calibrated,
    unvalidated for EHR clustering, so a signal not a verdict (see the Stage-1
    battery notes).

    Args:
        matrix: ``(n_patients, n_dims)`` float array (raw embedding vectors).

    Returns:
        The RankMe effective rank as a float.
    """
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    p = singular_values / (singular_values.sum() + _RANKME_EPS) + _RANKME_EPS
    entropy = -(p * np.log(p)).sum()
    return float(np.exp(entropy))


def _mean_cosine_to_centroid(matrix):
    """Average cosine similarity of each row to the matrix centroid.

    ~1.0 means every patient points the same way (anisotropic collapse); ~0
    means directions are well spread.
    """
    norms = np.linalg.norm(matrix, axis=1)
    centroid = matrix.mean(axis=0)
    cos_to_centroid = matrix @ centroid / (norms * np.linalg.norm(centroid) + 1e-12)
    return float(cos_to_centroid.mean())


def _known_base_codes(tokenizer):
    """Set of base codes in the vocab (value buckets and special tokens stripped).

    Lab codes appear as value-bucketed strings ("LOINC/2236-8 || ... - 4.0"); we
    take the part before " || " so a code counts as known regardless of value.
    """
    return {
        token.split(" || ", 1)[0]
        for token in tokenizer.get_vocab()
        if not token.startswith("[")
    }
