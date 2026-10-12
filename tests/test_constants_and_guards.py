"""Pin constants and the checks that run before a fit."""

import numpy as np
import pytest

from tests.fixture_rows import save_rows
from xverifier import constants
from xverifier.config import budget_of, load_experiment
from xverifier.frames import load_frame
from xverifier.guards import groups_of
from xverifier.splitting import assign_roles

EXPECTED = {
    "CALIBRATION_FRACTION": 0.274,
    "EVALUATION_FRACTION": 0.082,
    "PROBE_C": 1.0,
    "PROBE_MAX_ITER": 2000,
    "PROBE_RANDOM_STATE": 42,
    "PCA_CAP": 256,
    "PCA_FLOOR": 4,
    "MIN_ROWS_PER_VERDICT": 40,
    "CROSS_FIT_FOLDS": 5,
    "FLIP_THRESHOLD": 0.5,
    "ERROR_LEVELS": (0.25, 0.20, 0.15, 0.10, 0.05),
    "CONFIDENCE": 0.95,
    "ERROR_BUDGET": 0.05,
    "CALIBRATION_MIN_CELL": 8,
    "GRID_LEVELS": (
        0.00, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95,
    ),
    "TIE_TOLERANCE": 1e-12,
}


def _write_rows(path, n=6, label=None, verdict=None, representation=None):
    representation = (np.zeros((n, 4), dtype=np.float32) if representation is None else representation)
    label = np.zeros(n, dtype=np.int8) if label is None else label
    verdict = np.zeros(n, dtype=np.int8) if verdict is None else verdict
    save_rows(path, representation, label, verdict, group=np.arange(n))


def _experiment(directory, extra=""):
    rows = directory / "rows.npz"
    _write_rows(rows)
    text = f"""
study: demo
seeds: [1]
frame:
  path: {rows.name}
split:
  mode: roles
estimator:
  probe:
    family: linear
    C: 1.0
    max_components: 256
    min_rows_per_verdict: 40
  calibration:
    method: venn_abers
    cross_fit_folds: 5
flip:
  threshold: 0.5
decide:
  error_levels: [0.25, 0.20, 0.15, 0.10, 0.05]
  confidence: 0.95
{extra}
"""
    path = directory / "experiment.yaml"
    path.write_text(text)
    role = np.array(["train"] * 2 + ["calibration"] * 2 + ["test"] * 2)
    save_rows(rows, np.zeros((6, 4), dtype=np.float32), np.zeros(6, dtype=np.int8),
              np.array([0, 1, 0, 1, 0, 1], dtype=np.int8), group=np.arange(6), role=role)
    return path


class TestConstantsArePinned:
    def test_every_constant(self):
        # GIVEN the fixed table
        # WHEN the public names are read
        names = {name for name in vars(constants) if name.isupper()}
        # THEN each name and value matches the pin, and no extra name is present
        assert names == set(EXPECTED)
        for name, value in EXPECTED.items():
            assert getattr(constants, name) == value

    def test_confidence_budget_is_exact(self):
        # GIVEN confidence 0.95
        # WHEN the budget is derived
        budget = budget_of(0.95)
        # THEN it is the pinned 0.05, not the leftover of a binary subtraction
        assert budget == constants.ERROR_BUDGET
        assert budget != 1 - 0.95


class TestBadRowsFailBeforeAFit:
    def test_label_outside_zero_and_one_matches_the_whole_message(self, tmp_path):
        # GIVEN a label of 2
        path = tmp_path / "rows.npz"
        _write_rows(path, label=np.array([0, 1, 2, 0], dtype=np.int8), n=4)
        # WHEN the file is loaded
        with pytest.raises(ValueError) as caught:
            load_frame(path)
        # THEN the message names the field and the failing value, and nothing else
        assert str(caught.value) == "label must be in {0, 1}; got 2."

    def test_verdict_outside_zero_and_one(self, tmp_path):
        path = tmp_path / "rows.npz"
        _write_rows(path, verdict=np.array([0, 3], dtype=np.int8), n=2)
        with pytest.raises(ValueError) as caught:
            load_frame(path)
        assert str(caught.value) == "verdict must be in {0, 1}; got 3."

    def test_nonfinite_representation(self, tmp_path):
        path = tmp_path / "rows.npz"
        representation = np.zeros((2, 4), dtype=np.float32)
        representation[0, 0] = np.nan
        _write_rows(path, n=2, representation=representation)
        with pytest.raises(ValueError) as caught:
            load_frame(path)
        assert str(caught.value) == "representation contains a non-finite value; got nan."

    def test_length_mismatch(self, tmp_path):
        path = tmp_path / "rows.npz"
        save_rows(path, np.zeros((4, 4), dtype=np.float32), np.zeros(3, dtype=np.int8),
                  np.zeros(4, dtype=np.int8))
        with pytest.raises(ValueError) as caught:
            load_frame(path)
        assert str(caught.value) == "label length must equal representation length; got 3 and 4."

    def test_unknown_key(self, tmp_path):
        path = _experiment(tmp_path, extra="extra: 1\n")
        with pytest.raises(ValueError) as caught:
            load_experiment(path)
        assert str(caught.value) == "unknown key 'extra'."

    def test_missing_section(self, tmp_path):
        path = _experiment(tmp_path)
        text = path.read_text().replace("split:\n  mode: roles\n", "")
        path.write_text(text)
        with pytest.raises(ValueError) as caught:
            load_experiment(path)
        assert str(caught.value) == "missing section 'split'."

class TestNegativeGroupsAreOwnGroups:
    def test_every_negative_id_matches_a_missing_group_column(self):
        # GIVEN the same rows with every group equal to -1, and with no group column
        n = 20
        negative = np.full(n, -1, dtype=np.int64)
        # WHEN both are drawn at the same seed
        left = assign_roles(groups_of(negative, n), mode="files", seed=7, calibration_fraction=0.274)
        right = assign_roles(groups_of(None, n), mode="files", seed=7, calibration_fraction=0.274)
        # THEN the roles match
        assert np.array_equal(left["train"], right["train"])
        assert np.array_equal(left["calibration"], right["calibration"])

    def test_each_negative_row_gets_a_fresh_id_in_row_order(self):
        # GIVEN two real groups and two ungrouped rows
        group = np.array([5, 5, -1, -2], dtype=np.int64)
        # WHEN the ids are assigned
        assigned = groups_of(group, 4)
        # THEN the real group stays together and the negative rows get 6 then 7
        assert assigned.tolist() == [5, 5, 6, 7]
