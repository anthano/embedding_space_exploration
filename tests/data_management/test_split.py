import pandas as pd

from embedding_space_exploration.data_management.split import (
    DERIVATION,
    DEVELOPMENT,
    LOCKBOX,
    UNSEAL_LOCKBOX_ENV,
    assign_split,
    drop_lockbox,
    lockbox_is_sealed,
    visible_splits,
)

COHORT_SIZE = 100
# Defaults: hold out 40%, halve into development / lockbox -> 60 / 20 / 20.
EXPECTED = {DERIVATION: 60, DEVELOPMENT: 20, LOCKBOX: 20}
# dx_group stratified (60 scz / 40 bipolar) held out and halved.
EXPECTED_SCZ = {DERIVATION: 36, DEVELOPMENT: 12, LOCKBOX: 12}
EXPECTED_BIPOLAR = {DERIVATION: 24, DEVELOPMENT: 8, LOCKBOX: 8}
AGE_GROUP_YOUNGER = 20
AGE_GROUP_OLDER = 70
EXPECTED_PER_AGE_GROUP = {DERIVATION: 30, DEVELOPMENT: 10, LOCKBOX: 10}


def _cohort(n=100):
    return pd.DataFrame(
        {
            "person_id": range(n),
            "dx_group": ["schizophrenia_spectrum"] * 60 + ["bipolar"] * 40,
            "age_at_index": [25] * n,
        }
    )


def test_assign_split_ratio_and_columns():
    split = assign_split(_cohort(COHORT_SIZE), seed=0)
    assert list(split.columns) == ["person_id", "split"]
    assert set(split["split"]) == {DERIVATION, DEVELOPMENT, LOCKBOX}
    assert split["split"].value_counts().to_dict() == EXPECTED
    # every patient labelled exactly once, in cohort order
    assert split["person_id"].tolist() == list(range(100))


def test_assign_split_is_deterministic():
    a = assign_split(_cohort(100), seed=0)
    b = assign_split(_cohort(100), seed=0)
    pd.testing.assert_frame_equal(a, b)


def test_assign_split_stratifies_on_a_declared_column():
    cohort = _cohort(COHORT_SIZE)
    split = assign_split(cohort, seed=0, stratify_cols=("dx_group",))
    merged = cohort.merge(split, on="person_id")
    for split_name, expected in EXPECTED_SCZ.items():
        rows = merged.loc[merged["split"] == split_name, "dx_group"]
        assert (rows == "schizophrenia_spectrum").sum() == expected
        assert (rows == "bipolar").sum() == EXPECTED_BIPOLAR[split_name]


def test_assign_split_falls_back_to_age_band_without_a_declared_column():
    cohort = pd.DataFrame(
        {
            "person_id": range(COHORT_SIZE),
            "age_at_index": [AGE_GROUP_YOUNGER] * 50 + [AGE_GROUP_OLDER] * 50,
        }
    )
    merged = cohort.merge(assign_split(cohort, seed=0), on="person_id")
    for split_name, expected in EXPECTED_PER_AGE_GROUP.items():
        rows = merged.loc[merged["split"] == split_name, "age_at_index"]
        assert (rows == AGE_GROUP_YOUNGER).sum() == expected
        assert (rows == AGE_GROUP_OLDER).sum() == expected


def test_assign_split_honours_a_custom_id_column():
    cohort = pd.DataFrame({"patient_id": range(COHORT_SIZE)})
    split = assign_split(cohort, seed=0, id_col="patient_id")
    assert list(split.columns) == ["patient_id", "split"]
    assert split["split"].value_counts().to_dict() == EXPECTED


def _labels():
    return pd.DataFrame({"person_id": [1, 2, 3], "cluster": [0, 1, 0]})


def _split_frame():
    return pd.DataFrame(
        {"person_id": [1, 2, 3], "split": [DERIVATION, DEVELOPMENT, LOCKBOX]}
    )


def test_lockbox_is_sealed_by_default(monkeypatch):
    monkeypatch.delenv(UNSEAL_LOCKBOX_ENV, raising=False)
    assert lockbox_is_sealed()
    assert LOCKBOX not in visible_splits()


def test_drop_lockbox_removes_held_out_patients(monkeypatch):
    monkeypatch.delenv(UNSEAL_LOCKBOX_ENV, raising=False)
    visible = drop_lockbox(_labels(), _split_frame())
    assert visible["person_id"].tolist() == [1, 2]
    assert LOCKBOX not in visible["split"].tolist()


def test_unsealing_requires_an_explicit_opt_in(monkeypatch):
    monkeypatch.setenv(UNSEAL_LOCKBOX_ENV, "1")
    assert not lockbox_is_sealed()
    assert drop_lockbox(_labels(), _split_frame())["person_id"].tolist() == [1, 2, 3]


def test_unlabelled_patients_are_treated_as_possibly_held_out(monkeypatch):
    # A patient with no split label cannot be *shown* to be outside the lockbox, so
    # the safe reading is that they might be inside it.
    monkeypatch.delenv(UNSEAL_LOCKBOX_ENV, raising=False)
    labels = pd.DataFrame({"person_id": [1, 99], "cluster": [0, 0]})
    assert drop_lockbox(labels, _split_frame())["person_id"].tolist() == [1]
