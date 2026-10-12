"""Pin the synthetic table, the label flip, and the saved-model round trip."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tests.fixture_rows import test_rows as holdout_rows
from tests.fixture_rows import training_rows
from xverifier.cli import main
from xverifier.config import load_experiment
from xverifier.flip import FlipRule
from xverifier.runner import execute, read_frames, run, score_file, train
from xverifier.store import load_model

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "examples" / "synthetic.yaml"
TRAIN_ROWS = ROOT / "examples" / "rows" / "train.npz"
TEST_ROWS = ROOT / "examples" / "rows" / "test.npz"

TABLE = "\n".join([
    "error level   cutoff     decided    errors   error rate",
    "0.25          0.571429   300/300    21       0.070",
    "0.20          0.571429   300/300    21       0.070",
    "0.15          0.571429   300/300    21       0.070",
    "0.10          0.772064   274/300     9       0.033",
    "0.05          review all    0/300",
])

COUNTS = (
    (0.25, False, 300, 21, "0.571429"),
    (0.20, False, 300, 21, "0.571429"),
    (0.15, False, 300, 21, "0.571429"),
    (0.10, False, 274, 9, "0.772064"),
    (0.05, True, 0, None, None),
)


def _settings():
    return {
        "mode": "files",
        "calibration_fraction": 0.274,
        "probe": {"C": 1.0, "max_components": 256, "min_rows_per_verdict": 40},
        "calibration": {"cross_fit_folds": 5},
        "threshold": 0.5,
        "error_levels": [0.25, 0.20, 0.15, 0.10, 0.05],
        "error_budget": 0.05,
    }


def _frame(representation, label, verdict, group):
    label = np.asarray(label, dtype=np.int8)
    verdict = np.asarray(verdict, dtype=np.int8)
    return {
        "representation": np.asarray(representation),
        "label": label,
        "verdict": verdict,
        "agreement": np.ascontiguousarray(label == verdict, dtype=np.int8),
        "group": np.asarray(group),
        "role": None,
        "sha256": "0" * 64,
        "path": "memory",
    }


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestSyntheticTable:
    def test_counts_cutoffs_and_printed_table(self, tmp_path, capsys):
        # GIVEN the shipped 900/300 synthetic files
        output = tmp_path / "runs"
        # WHEN one seed is run and printed
        results = run(EXPERIMENT, output)
        point = results[7]
        status = main(["run", str(EXPERIMENT), "--out", str(tmp_path / "printed")])
        printed = capsys.readouterr().out
        # THEN the slices, the five counts, and the printed table match
        assert status == 0
        assert point["rows"] == {"train": 654, "calibration": 246, "test": 300}
        assert point["pca_dim"] == 32
        assert printed == TABLE + "\n"
        seen = []
        for row in point["operating_points"]:
            shown = None if row["cutoff"] is None else f"{row['cutoff']:.6f}"
            seen.append((row["error_level"], row["review_all"], row["decided"], row["errors"], shown))
        assert seen == list(COUNTS)
        assert point["operating_points"][-1]["decided"] == 0
        assert point["operating_points"][-1]["rows"] == 300

    def test_records_store_the_file_digests(self, tmp_path):
        # GIVEN a finished run
        output = tmp_path / "runs"
        run(EXPERIMENT, output)
        record = json.loads((output / "seed_7.json").read_text())
        aggregate = json.loads((output / "aggregate.json").read_text())
        # THEN each record stores the YAML digest and both row-file digests
        assert record["experiment_sha256"] == _digest(EXPERIMENT)
        assert record["row_sha256"]["train"] == _digest(TRAIN_ROWS)
        assert record["row_sha256"]["test"] == _digest(TEST_ROWS)
        level = aggregate["error_levels"]["0.05"]
        assert level["coverage_mean"] == 0.0
        assert level["n_certified"] == 0
        assert level["error_rate_mean"] is None
        assert level["coverage_std"] is None
        wide = aggregate["error_levels"]["0.25"]
        assert wide["n_certified"] == 1
        assert wide["error_rate_mean"] == pytest.approx(21 / 300)
        assert wide["error_rate_std"] is None
        assert wide["coverage_std"] is None
        tight = aggregate["error_levels"]["0.10"]
        assert tight["coverage_mean"] == pytest.approx(274 / 300)
        assert tight["error_rate_mean"] == pytest.approx(9 / 274)

    def test_reloaded_model_reproduces_the_probabilities(self, tmp_path, capsys):
        # GIVEN a model written by train
        model = tmp_path / "model"
        live = train(EXPERIMENT, 7, model)["probability"]["test"]
        # WHEN the test command scores the same rows from the saved arrays
        status = main(["test", "--rows", str(TEST_ROWS), "--model", str(model)])
        scored = score_file(TEST_ROWS, load_model(model))
        # THEN the probabilities are identical and the table is unchanged
        assert status == 0
        assert capsys.readouterr().out == TABLE + "\n"
        assert np.array_equal(scored["probability"], live)

    def test_existing_output_directory_is_refused(self, tmp_path):
        # GIVEN an output directory that already exists
        output = tmp_path / "runs"
        output.mkdir()
        # WHEN a run is asked to use it
        with pytest.raises(ValueError) as caught:
            run(EXPERIMENT, output)
        # THEN the message names that directory
        assert str(caught.value) == f"output directory already exists; got {str(output)!r}."

    def test_nonfinite_rows_do_not_create_the_output_directory(self, tmp_path):
        # GIVEN a non-finite representation
        from tests.fixture_rows import save_rows
        rows = tmp_path / "rows.npz"
        representation = np.zeros((4, 4), dtype=np.float32)
        representation[0, 0] = np.inf
        save_rows(rows, representation, np.zeros(4, dtype=np.int8), np.zeros(4, dtype=np.int8))
        experiment = tmp_path / "experiment.yaml"
        experiment.write_text(f"""
study: demo
seeds: [1]
frame:
  train: {rows.name}
  test: {rows.name}
split:
  mode: files
  calibration_fraction: 0.274
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
  error_levels: [0.25]
  confidence: 0.95
""")
        output = tmp_path / "out"
        # WHEN the run is asked to fit
        with pytest.raises(ValueError) as caught:
            run(experiment, output)
        # THEN it stops before creating the output directory
        assert str(caught.value) == "representation contains a non-finite value; got inf."
        assert not output.exists()


class TestSavedModelRoundTrip:
    def test_reloaded_model_matches_the_live_fit_bit_for_bit(self, tmp_path):
        # GIVEN the live fit of the synthetic example and the model train writes for the same seed
        experiment = load_experiment(EXPERIMENT)
        live = execute(experiment, read_frames(experiment), 7)
        model = tmp_path / "model"
        train(EXPERIMENT, 7, model)
        # WHEN the saved model is loaded and scores the test rows
        loaded = load_model(model)
        scored = score_file(TEST_ROWS, loaded)
        frame = scored["frame"]
        scored["raw"] = loaded["probe"].scores(frame["representation"], frame["verdict"])
        # THEN the probabilities, confidence, raw scores and operating points equal the live ones bit for bit
        for name in ("probability", "confidence", "raw"):
            assert scored[name].dtype == live[name]["test"].dtype
            assert scored[name].shape == live[name]["test"].shape
            assert scored[name].tobytes() == live[name]["test"].tobytes()
        for reloaded, original in zip(scored["operating_points"], live["operating_points"], strict=True):
            assert reloaded == {key: original[key] for key in reloaded}


class TestLabelFlipKeepsTheOperatingPoints:
    def test_agreement_and_scores_stay_and_the_verdicts_swap(self):
        # GIVEN the synthetic rows and the same rows with both bits complemented
        representation, label, verdict, group = training_rows()
        test_representation, test_label, test_verdict, test_group = holdout_rows()
        original = {
            "train": _frame(representation, label, verdict, group),
            "test": _frame(test_representation, test_label, test_verdict, test_group),
        }
        flipped = {
            "train": _frame(representation, 1 - label, 1 - verdict, group),
            "test": _frame(test_representation, 1 - test_label, 1 - test_verdict, test_group),
        }
        # WHEN both are fit at the same seed
        left = execute(_settings(), original, 7)
        right = execute(_settings(), flipped, 7)
        # THEN the roles and agreement stay, the scores stay, and the emitted verdicts swap
        for role in ("train", "calibration", "test"):
            assert np.array_equal(left["index"][role], right["index"][role])
            assert np.array_equal(left["agreement"][role], right["agreement"][role])
            assert np.array_equal(left["raw"][role], right["raw"][role])
            assert np.array_equal(left["probability"][role], right["probability"][role])
            kept = FlipRule(0.5).corrected_verdict(left["verdict"][role], left["probability"][role])
            swapped = FlipRule(0.5).corrected_verdict(right["verdict"][role], right["probability"][role])
            assert np.array_equal(swapped, 1 - kept)
        assert left["operating_points"] == right["operating_points"]
