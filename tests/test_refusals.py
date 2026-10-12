"""Refuse bad input with one line, and leave no output directory behind."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.fixture_rows import save_rows, training_rows
from tests.fixture_rows import test_rows as holdout_rows
from xverifier import runner, store
from xverifier.calibrations import VennAbers
from xverifier.cli import main
from xverifier.estimator import Estimator
from xverifier.frames import Slice, load_frame
from xverifier.guards import groups_of
from xverifier.probes import RoutedLinearProbe
from xverifier.splitting import assign_roles

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "examples" / "synthetic.yaml"
INT64_MAX = 2**63 - 1
LEVELS = "[0.25, 0.20, 0.15, 0.10, 0.05]"


def _experiment(directory, frame, split, levels=LEVELS):
    text = f"""study: demo
seeds: [7]
frame:
{frame}
split:
{split}
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
  error_levels: {levels}
  confidence: 0.95
"""
    path = directory / "experiment.yaml"
    path.write_text(text)
    return path


def _files(directory, group, fraction=0.274, levels=LEVELS):
    representation, label, verdict, _ = training_rows()
    save_rows(directory / "train.npz", representation, label, verdict, group=group)
    representation, label, verdict, test_group = holdout_rows()
    save_rows(directory / "test.npz", representation, label, verdict, group=test_group)
    return _experiment(directory, "  train: train.npz\n  test: test.npz",
                       f"  mode: files\n  calibration_fraction: {fraction}", levels)


def _draw(directory, group, calibration, evaluation):
    representation, label, verdict, _ = training_rows()
    save_rows(directory / "rows.npz", representation, label, verdict, group=group)
    return _experiment(directory, "  path: rows.npz",
                       f"  mode: draw\n  calibration_fraction: {calibration}\n"
                       f"  evaluation_fraction: {evaluation}")


def _roles(directory, group):
    representation, label, verdict, _ = training_rows()
    role = np.array(["train"] * 600 + ["calibration"] * 200 + ["test"] * 100)
    save_rows(directory / "rows.npz", representation, label, verdict, group=group, role=role)
    return _experiment(directory, "  path: rows.npz", "  mode: roles")


def _cli(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH")) if part)
    return subprocess.run([sys.executable, "-m", "xverifier", *args],
                          capture_output=True, text=True, env=env, check=False)


def _outcome(experiment, command="run", output=None):
    directory = experiment.parent
    before = {path.name for path in directory.iterdir()}
    output = directory / "out" if output is None else output
    if command == "train":
        result = _cli("train", str(experiment), "--seed", "7", "--out", str(output))
    else:
        result = _cli("run", str(experiment), "--out", str(output))
    added = sorted(path.name for path in directory.iterdir() if path.name not in before)
    return (result.returncode, result.stderr.splitlines(), "Traceback" in result.stderr, added)


def _refused(message):
    return (1, [message], False, [])


class TestSplitsWithoutEnoughGroups:
    @pytest.mark.parametrize("command", ["run", "train"])
    def test_one_group_train_file_has_no_calibration_group(self, tmp_path, command):
        # GIVEN a files-mode training file whose 900 rows share one group
        experiment = _files(tmp_path, np.zeros(900, dtype=np.int64))
        # WHEN the command is run
        outcome = _outcome(experiment, command)
        # THEN it exits 1 with one line and no output directory, where today PCA fails after
        # the directory exists
        assert outcome == _refused(
            "split.calibration_fraction leaves no calibration group; got 0.274 and 1 group.")

    def test_draw_with_no_calibration_group(self, tmp_path):
        # GIVEN three groups, so floor(0.274 * 3) calibration groups is zero
        experiment = _draw(tmp_path, np.arange(900) % 3, 0.274, 0.082)
        # WHEN the run starts
        outcome = _outcome(experiment)
        # THEN it is refused before any output directory exists
        assert outcome == _refused(
            "split.calibration_fraction leaves no calibration group; got 0.274 and 3 groups.")

    def test_files_with_one_training_group(self, tmp_path):
        # GIVEN two groups and a calibration fraction of one half, so one group trains
        experiment = _files(tmp_path, np.arange(900) % 2, fraction=0.5)
        # WHEN the run starts
        outcome = _outcome(experiment)
        # THEN it is refused, where today the run completes
        assert outcome == _refused("training slice must span at least two groups; got 1.")

    def test_draw_with_one_training_group(self, tmp_path):
        # GIVEN four groups: one calibrates, two test, and one is left to train
        experiment = _draw(tmp_path, np.arange(900) % 4, 0.274, 0.5)
        # WHEN the run starts
        outcome = _outcome(experiment)
        # THEN it is refused, where today the run completes
        assert outcome == _refused("training slice must span at least two groups; got 1.")

    def test_draw_with_no_test_group(self, tmp_path):
        # GIVEN ten groups, so two calibrate and floor(0.082 * 10) test groups is zero
        experiment = _draw(tmp_path, np.arange(900) % 10, 0.274, 0.082)
        # WHEN the run starts
        outcome = _outcome(experiment)
        # THEN it is refused before any output directory exists, where today PCA fails on no rows
        assert outcome == _refused(
            "split.evaluation_fraction leaves no test group; got 0.082 and 10 groups.")

    @pytest.mark.parametrize("command", ["run", "train"])
    def test_roles_with_one_training_group(self, tmp_path, command):
        # GIVEN a roles file whose 600 training rows share one group
        group = np.concatenate([np.zeros(600, dtype=np.int64), np.arange(1, 301)])
        experiment = _roles(tmp_path, group)
        # WHEN the command is run
        outcome = _outcome(experiment, command)
        # THEN it is refused, where today the fit runs and keeps the raw scores
        assert outcome == _refused("training slice must span at least two groups; got 1.")

    def test_group_in_two_roles(self, tmp_path):
        # GIVEN a roles file whose last test row shares group 0 with the first training row
        group = np.arange(900)
        group[899] = 0
        experiment = _roles(tmp_path, group)
        # WHEN the run starts
        outcome = _outcome(experiment)
        # THEN it is refused, where today the run completes
        assert outcome == _refused("group 0 must have one role; got 'train' and 'test'.")

    def test_ungrouped_rows_in_every_role_are_kept(self, tmp_path):
        # GIVEN a roles file in which every row has the ungrouped id -1
        _roles(tmp_path, np.full(900, -1, dtype=np.int64))
        frame = load_frame(tmp_path / "rows.npz")
        # WHEN the roles are assigned
        found = assign_roles(frame["group"], mode="roles", seed=7, calibration_fraction=None,
                             role=frame["role"])
        # THEN each row is its own group, so neither the role nor the training check refuses it
        assert [len(found[name]) for name in ("train", "calibration", "test")] == [600, 200, 100]


class TestExperimentFileRefusals:
    def test_levels_that_share_a_two_decimal_key(self, tmp_path):
        # GIVEN 0.12 and 0.125, which both print as 0.12
        experiment = _files(tmp_path, np.arange(900), levels="[0.12, 0.125]")
        # WHEN the run starts
        outcome = _outcome(experiment)
        # THEN it is refused, where today one level's aggregate overwrites the other's
        assert outcome == _refused(
            "decide.error_levels must differ at two decimals; got 0.12 and 0.125.")

    @pytest.mark.parametrize(("original", "repeated", "name"), [
        ("study: demo\n", "study: demo\nstudy: other\n", "study"),
        ("  mode: files\n", "  mode: files\n  mode: draw\n", "split.mode"),
        ("    C: 1.0\n", "    C: 1.0\n    C: 2.0\n", "estimator.probe.C"),
    ])
    def test_repeated_key_at_any_depth(self, tmp_path, original, repeated, name):
        # GIVEN an experiment with one key written twice
        experiment = _files(tmp_path, np.arange(900))
        text = experiment.read_text().replace(original, repeated, 1)
        experiment.write_text(text)
        line = text[:text.index(repeated) + len(original)].count("\n") + 1
        # WHEN the run starts
        outcome = _outcome(experiment)
        # THEN it is refused, where today the last value wins in silence
        assert outcome == _refused(f"repeated key {name!r}; got it again on line {line}.")


class TestGroupIdsOutsideInt64:
    def test_unsigned_id_at_two_to_the_63(self, tmp_path):
        # GIVEN two rows that share the unsigned group id 2**63, which the int64 cast turns negative
        group = np.arange(900, dtype=np.uint64)
        group[0:2] = 2**63
        experiment = _files(tmp_path, group)
        # WHEN the run starts
        outcome = _outcome(experiment)
        # THEN it is refused, where today the two rows silently become two groups
        assert outcome == _refused(f"group must be at most {INT64_MAX}; got {2**63}.")

    def test_generated_ids_past_the_int64_maximum(self, tmp_path):
        # GIVEN a real id at the int64 maximum and two rows with no group
        group = np.arange(900, dtype=np.int64)
        group[0] = INT64_MAX
        group[1:3] = -1
        experiment = _files(tmp_path, group)
        # WHEN the run starts
        outcome = _outcome(experiment)
        # THEN it is refused with one line, where today numpy's overflow prints a traceback
        assert outcome == _refused(
            f"group must leave room above its largest id for 2 negative-id rows; got {INT64_MAX}.")

    def test_ids_at_the_edge_are_kept(self):
        # GIVEN ids that reach the int64 maximum but do not pass it
        unsigned = np.array([INT64_MAX, 0], dtype=np.uint64)
        tight = np.array([INT64_MAX - 2, -1, -1], dtype=np.int64)
        # THEN they are kept, and the two negative rows take the last two ids
        assert groups_of(unsigned, 2).tolist() == [INT64_MAX, 0]
        assert groups_of(tight, 3).tolist() == [INT64_MAX - 2, INT64_MAX - 1, INT64_MAX]


def _small_rows():
    rng = np.random.RandomState(2)
    representation = rng.normal(size=(65, 6))
    verdict = np.array([0] * 5 + [1] * 60)
    agreement = np.array([0, 1, 0, 1, 0] + [0, 1] * 30)
    return representation, verdict, agreement


class TestApiGuards:
    @pytest.mark.parametrize(("verdict", "shown"), [([0, 1, 0, 2], "2"), ([0.0, 1.0, 0.5, 1.0], "0.5")])
    def test_probe_refuses_a_verdict_outside_zero_and_one(self, verdict, shown):
        # GIVEN a fitted probe and four rows, one of whose verdicts is neither 0 nor 1
        representation, fit_verdict, agreement = _small_rows()
        probe = RoutedLinearProbe(C=1.0, max_components=256, min_rows=40, seed=2).fit(
            representation, fit_verdict, agreement)
        # WHEN the rows are scored
        with pytest.raises(ValueError) as caught:
            probe.scores(representation[:4], np.array(verdict))
        # THEN the verdict is refused, where today that row keeps uninitialized memory
        assert str(caught.value) == f"verdict must be in {{0, 1}}; got {shown}."

    @pytest.mark.parametrize(("verdict", "shown"), [([0, 1, 0, 2], "2"), ([0.0, 1.0, 0.5, 1.0], "0.5")])
    def test_calibration_refuses_a_verdict_outside_zero_and_one(self, verdict, shown):
        # GIVEN a calibration fit on both verdicts and four scores, one with a verdict of neither 0 nor 1
        scores = np.linspace(0.05, 0.95, 40)
        calibration = VennAbers(min_cell=8).fit(scores, np.tile([0, 1], 20), np.repeat([0, 1], 20))
        # WHEN the scores are calibrated
        with pytest.raises(ValueError) as caught:
            calibration.probability(np.array([0.2, 0.4, 0.6, 0.8]), np.array(verdict))
        # THEN the verdict is refused, where today that row keeps uninitialized memory
        assert str(caught.value) == f"verdict must be in {{0, 1}}; got {shown}."

    def test_estimator_refuses_training_rows_left_unscored(self):
        # GIVEN 65 training rows that share one group, so the only cross-fitting fold has no rows to fit
        representation, verdict, agreement = _small_rows()
        estimator = Estimator({"C": 1.0, "max_components": 256, "min_rows_per_verdict": 40},
                              {"cross_fit_folds": 5}, seed=2)
        # WHEN the estimator is fit through the API, without the runner's split check
        with pytest.raises(ValueError) as caught:
            estimator.fit(Slice(representation, verdict, agreement), np.zeros(65, dtype=np.int64))
        # THEN it is refused, where today the calibration fits on no rows and returns the raw scores
        assert str(caught.value) == "out-of-fold pass must score every training row; got 65 of 65 unscored."


def _single_class(directory):
    experiment = _files(directory, np.arange(900))
    representation, label, _, group = training_rows()
    save_rows(directory / "train.npz", representation, label, label, group=group)
    return experiment


def _failing_write(real, name):
    def write_json(path, payload):
        if Path(path).name == name:
            raise OSError("disk full")
        real(path, payload)
    return write_json


class TestNoOutputAfterTheFilesAreAccepted:
    @pytest.mark.parametrize("command", ["run", "train"])
    def test_single_agreement_class(self, tmp_path, command):
        # GIVEN a training file whose verdicts all agree with the label
        experiment = _single_class(tmp_path)
        # WHEN the command fits it
        outcome = _outcome(experiment, command)
        # THEN it exits 1 with one line and leaves nothing, where today the directory stays behind
        assert outcome == _refused("agreement on the training rows has a single class; got [1].")

    @pytest.mark.parametrize(("written", "shown"), [(".nan", "nan"), (".inf", "inf")])
    def test_nonfinite_c(self, tmp_path, written, shown):
        # GIVEN a probe strength that YAML reads as a non-finite float
        experiment = _files(tmp_path, np.arange(900))
        experiment.write_text(experiment.read_text().replace("C: 1.0", f"C: {written}"))
        # WHEN the run starts
        outcome = _outcome(experiment)
        # THEN it is refused, where today nan fails inside scikit-learn and inf runs to the end
        assert outcome == _refused(f"estimator.probe.C must be finite; got {shown}.")

    @pytest.mark.parametrize("command", ["run", "train"])
    def test_test_file_with_fewer_columns(self, tmp_path, command):
        # GIVEN a 32-column training file and a 16-column test file
        experiment = _files(tmp_path, np.arange(900))
        representation, label, verdict, group = holdout_rows()
        save_rows(tmp_path / "test.npz", representation[:, :16], label, verdict, group=group)
        # WHEN the command scores the test rows
        outcome = _outcome(experiment, command)
        # THEN the package names the mismatch, where today the scikit-learn text is printed
        assert outcome == _refused("representation must have 32 columns like the training rows; got 16.")

    def test_a_later_seed_fails(self, tmp_path):
        # GIVEN a roles file and a second seed outside the range a random state accepts
        experiment = _roles(tmp_path, np.arange(900))
        experiment.write_text(experiment.read_text().replace("seeds: [7]", "seeds: [7, 4294967296]"))
        # WHEN seed 7 writes its record and the second seed fails
        outcome = _outcome(experiment)
        # THEN nothing is left, and the package names the seed in place of the scikit-learn text
        assert outcome == _refused("seed must be a valid random seed; got 4294967296.")

    def test_missing_parents_are_removed_after_a_failure(self, tmp_path):
        # GIVEN an --out path two folders below directories that do not exist yet
        experiment = _single_class(tmp_path)
        output = tmp_path / "runs" / "deep" / "out"
        # WHEN the run fails after the files are accepted
        outcome = _outcome(experiment, output=output)
        # THEN the folders made for it are gone too
        assert outcome == _refused("agreement on the training rows has a single class; got [1].")

    @pytest.mark.parametrize(("command", "module", "name"), [
        ("run", runner, "aggregate.json"),
        ("train", store, "meta.json"),
    ])
    def test_a_failed_write_leaves_nothing(self, tmp_path, monkeypatch, capsys, command, module, name):
        # GIVEN a disk that fails on the last file, after the earlier files are written
        monkeypatch.setattr(module, "write_json", _failing_write(module.write_json, name))
        output = tmp_path / "out"
        arguments = [command, str(EXPERIMENT), "--out", str(output)]
        if command == "train":
            arguments[2:2] = ["--seed", "7"]
        # WHEN the command runs
        status = main(arguments)
        # THEN it exits 1 with one line, and neither the directory nor a temporary one is left
        assert status == 1
        assert capsys.readouterr().err.splitlines() == [
            f"output directory {str(output)!r} was not written; got OSError: disk full."]
        assert list(tmp_path.iterdir()) == []

    def test_a_finished_run_leaves_only_its_directory(self, tmp_path):
        # GIVEN an --out path below directories that do not exist yet
        output = tmp_path / "runs" / "deep" / "out"
        # WHEN the run finishes
        runner.run(EXPERIMENT, output)
        # THEN the parents hold only --out, and --out holds only the records
        assert [path.name for path in output.parent.iterdir()] == ["out"]
        assert sorted(path.name for path in output.iterdir()) == ["aggregate.json", "seed_7.json"]


class TestOlderModelFile:
    def test_model_saved_without_the_pca_variance(self, tmp_path):
        # GIVEN a model whose model.npz was saved before the PCA variance was stored
        model = tmp_path / "model"
        runner.train(EXPERIMENT, 7, model)
        with np.load(model / "model.npz") as stored:
            arrays = {name: stored[name] for name in stored.files if name != "pca_explained_variance"}
        (model / "model.npz").unlink()
        np.savez(model / "model.npz", **arrays)
        # WHEN the test command loads it
        result = _cli("test", "--rows", str(ROOT / "examples" / "rows" / "test.npz"), "--model", str(model))
        # THEN it exits 1 with one line that asks for a retrain, where today a KeyError traceback prints
        assert (result.returncode, result.stderr.splitlines(), "Traceback" in result.stderr) == (
            1,
            [f"model must be retrained; got {str(model / 'model.npz')!r} without pca_explained_variance."],
            False,
        )
