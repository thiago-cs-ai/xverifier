"""The behaviors the README states match the code."""

import hashlib
import json
import re
import shutil
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from xverifier.cli import main
from xverifier.config import load_experiment
from xverifier.constants import ERROR_BUDGET, GRID_LEVELS, PCA_FLOOR, TIE_TOLERANCE
from xverifier.decide import _minimum_clean, allocate_budget, count_decisions, select_cutoffs, upper_bound
from xverifier.probes import RoutedLinearProbe, _pca_width
from xverifier.runner import read_frames, run
from xverifier.splitting import assign_roles

ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "examples" / "rows"
LEVELS = (0.25, 0.20, 0.15, 0.10, 0.05)


def _readme():
    return (ROOT / "README.md").read_text(encoding="utf-8")


def _minimum_table():
    rows = re.findall(r"^\| (0\.\d+) \| (\d+) \|$", _readme(), flags=re.MULTILINE)
    return {float(level): int(count) for level, count in rows}


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _synthetic(directory, study="synthetic", seeds="[7]", sha256=""):
    for name in ("train.npz", "test.npz"):
        shutil.copyfile(ROWS / name, directory / name)
    path = directory / "experiment.yaml"
    path.write_text(f"""study: {study}
seeds: {seeds}
frame:
  train: train.npz
  test: test.npz
{sha256}
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
  error_levels: [0.25, 0.20, 0.15, 0.10, 0.05]
  confidence: 0.95
""")
    return path


class TestMinimumRowsTable:
    def test_the_table_is_the_minimum_with_every_candidate_distinct(self):
        # GIVEN the README's table and the candidate count it names
        table = _minimum_table()
        named = re.search(r"(\d+) candidate cutoffs", _readme())
        # WHEN fifteen distinct training quantiles split the 5% budget and no calibration row is clean enough
        shares = allocate_budget(len(GRID_LEVELS), ERROR_BUDGET)
        train = np.linspace(0.5, 1.0, 1000)
        chosen, band = select_cutoffs(train, np.full(10, 0.99), np.zeros(10, dtype=int), list(LEVELS),
                                      ERROR_BUDGET)
        # THEN the README names the code's count, and each row is the minimum the code reports
        assert named is not None
        assert int(named.group(1)) == len(GRID_LEVELS) == len(band) == 15
        assert sorted(table) == sorted(LEVELS)
        for level, point in zip(LEVELS, chosen, strict=True):
            assert table[level] == _minimum_clean(level, ERROR_BUDGET / 15)
            assert table[level] == min(_minimum_clean(level, share) for share in shares)
            assert point["review_all"] is True
            assert point["minimum_clean"] == table[level]

    def test_fewer_distinct_candidates_never_raise_a_minimum(self):
        # GIVEN the README's table
        table = _minimum_table()
        for level, count in table.items():
            for candidates in range(1, len(GRID_LEVELS)):
                # WHEN fewer distinct candidates share the budget
                shares = allocate_budget(candidates, ERROR_BUDGET)
                # THEN the minimum is the same or lower
                assert min(_minimum_clean(level, share) for share in shares) <= count


class TestDecidedColumn:
    def test_the_window_counts_calibration_and_test_rows_alike(self):
        # GIVEN one candidate cutoff and rows at the edges of the window below it
        cutoff = 0.8
        edge = cutoff - TIE_TOLERANCE
        scores = np.concatenate([np.full(60, cutoff), [edge, np.nextafter(edge, 0.0), cutoff - 1e-6]])
        errors = np.zeros(len(scores), dtype=int)
        # WHEN the cutoff is chosen on these calibration rows and counted on the same rows as a test file
        chosen, band = select_cutoffs(np.full(100, cutoff), scores, errors, [0.25], ERROR_BUDGET)
        counted = count_decisions(chosen, scores, errors)
        # THEN the README names the window, and the same 61 rows count in calibration and in test
        assert f"score >= cutoff - {TIE_TOLERANCE:g}" in _readme()
        assert band[0]["decided"] == 61
        assert chosen[0]["cutoff"] == cutoff
        assert counted[0]["decided"] == 61

    def test_the_upper_bound_meets_the_level_with_no_window(self):
        # GIVEN one candidate with 3 errors among 60 calibration rows
        train = np.full(100, 0.8)
        calibration = np.full(60, 0.8)
        errors = np.zeros(60, dtype=int)
        errors[:3] = 1
        bound = upper_bound(3, 60, ERROR_BUDGET)
        # WHEN the level equals that bound, and when it is the next float below it
        at, _ = select_cutoffs(train, calibration, errors, [bound], ERROR_BUDGET)
        below, _ = select_cutoffs(train, calibration, errors, [float(np.nextafter(bound, 0.0))],
                                  ERROR_BUDGET)
        # THEN the cutoff is chosen at the bound and refused one float below it
        assert at[0]["review_all"] is False
        assert at[0]["upper_bound"] == bound
        assert below[0]["review_all"] is True


class TestOneRunCoversAllItsLevels:
    def test_a_level_gets_the_same_cutoff_alone_or_with_the_others(self):
        # GIVEN one draw of training and calibration scores
        rng = np.random.RandomState(0)
        train = rng.uniform(0.5, 1.0, 500)
        calibration = rng.uniform(0.5, 1.0, 400)
        errors = (rng.uniform(size=400) > calibration).astype(int)
        # WHEN every level is asked together, and each level alone
        together, band = select_cutoffs(train, calibration, errors, list(LEVELS), ERROR_BUDGET)
        # THEN each level's choice does not depend on the other levels, and one 5% covers every candidate
        for level, point in zip(LEVELS, together, strict=True):
            alone, _ = select_cutoffs(train, calibration, errors, [level], ERROR_BUDGET)
            assert alone[0] == point
        assert sum(Fraction(row["budget"]) for row in band) <= Fraction(ERROR_BUDGET)


class TestOutputFiles:
    def test_one_file_per_seed_and_the_aggregate_without_the_study_name(self, tmp_path):
        # GIVEN the synthetic rows, two seeds and a study name
        experiment = _synthetic(tmp_path, study="first-try", seeds="[7, 8]")
        output = tmp_path / "out"
        # WHEN they run
        run(experiment, output)
        # THEN the README's file names are the ones written, and the study is inside each file
        names = sorted(path.name for path in output.iterdir())
        assert names == ["aggregate.json", "seed_7.json", "seed_8.json"]
        assert "`seed_42.json`" in _readme()
        assert "`aggregate.json`" in _readme()
        for name in names:
            assert json.loads((output / name).read_text())["study"] == "first-try"


class TestSha256:
    @pytest.mark.parametrize("block", [
        "",
        "  sha256: null",
        "  sha256:\n    train: null\n    test: null",
        "  sha256:\n    test: null",
    ], ids=["omitted", "null", "both-null", "one-null-one-omitted"])
    def test_an_omitted_or_null_hash_is_not_checked(self, tmp_path, block):
        # GIVEN an experiment whose hash is left out or set to null
        experiment = load_experiment(_synthetic(tmp_path, sha256=block))
        # WHEN the row files are read
        frames = read_frames(experiment)
        # THEN no expected hash is held, and both files are read
        assert experiment["frame"]["sha256"] == {"train": None, "test": None}
        assert frames["train"]["sha256"] == _digest(ROWS / "train.npz")
        assert frames["test"]["sha256"] == _digest(ROWS / "test.npz")

    def test_a_given_hash_is_still_checked_beside_a_null_one(self, tmp_path):
        # GIVEN a wrong hash for the train file and null for the test file
        wrong = "ab" * 32
        experiment = load_experiment(
            _synthetic(tmp_path, sha256=f"  sha256:\n    train: {wrong}\n    test: null"))
        # WHEN the row files are read
        with pytest.raises(ValueError) as caught:
            read_frames(experiment)
        # THEN the train file is refused by name
        assert str(caught.value) == (
            f"frame.sha256.train must be {wrong}; got {_digest(ROWS / 'train.npz')}.")

    def test_a_null_hash_runs_from_the_command_line(self, tmp_path, capsys):
        # GIVEN sha256 set to null
        experiment = _synthetic(tmp_path, sha256="  sha256: null")
        # WHEN the experiment runs
        status = main(["run", str(experiment), "--out", str(tmp_path / "out")])
        # THEN it finishes and writes its records
        assert status == 0
        assert capsys.readouterr().err == ""
        assert (tmp_path / "out" / "seed_7.json").is_file()


class TestPcaWidth:
    def test_the_readme_rule_is_the_code_rule(self):
        # GIVEN the README's width rule
        assert "max(4, min(n_train // 10, max_components, columns))" in _readme()
        assert PCA_FLOOR == 4
        # WHEN the code's width is computed over a grid of sizes
        # THEN it equals the README's rule
        for n_train in (0, 5, 39, 40, 80, 654, 5000):
            for columns in (4, 7, 32, 300):
                for max_components in (1, 2, 4, 8, 256):
                    assert _pca_width(n_train, columns, max_components) == max(
                        4, min(n_train // 10, max_components, columns))

    def test_a_requested_width_below_four_fits_four(self):
        # GIVEN 80 rows of 30 columns and a requested width of 2
        representation = np.random.RandomState(0).normal(size=(80, 30))
        verdict = np.array([0] * 40 + [1] * 40)
        agreement = np.tile([0, 1], 40)
        # WHEN the probe is fit
        probe = RoutedLinearProbe(C=1.0, max_components=2, min_rows=40, seed=0).fit(
            representation, verdict, agreement)
        # THEN the fitted width is 4
        assert probe.n_components == 4


class TestCalibrationIsDrawnAgainForEachSeed:
    @pytest.mark.parametrize("mode", ["files", "draw"])
    def test_each_seed_draws_its_own_calibration_cases(self, mode):
        # GIVEN 900 groups split by the default fractions
        group = np.arange(900)

        def calibration(seed):
            return assign_roles(group, mode=mode, seed=seed, calibration_fraction=0.274,
                                evaluation_fraction=0.082)["calibration"]

        # WHEN the calibration cases are drawn for two seeds, and again for the first
        # THEN the README says so, one seed repeats its draw, and another seed draws other cases
        assert "drawn from your rows again for each seed" in " ".join(_readme().split())
        assert np.array_equal(calibration(7), calibration(7))
        assert not np.array_equal(calibration(7), calibration(8))
