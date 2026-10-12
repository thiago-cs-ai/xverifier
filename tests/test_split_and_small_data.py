"""Pin the group draw, the probe width, the pooled probe, and the raw-score cell."""

from fractions import Fraction

import numpy as np

from xverifier.calibrations import VennAbers
from xverifier.constants import ERROR_BUDGET, TIE_TOLERANCE
from xverifier.decide import accepts, allocate_budget
from xverifier.estimator import Estimator
from xverifier.frames import Slice
from xverifier.probes import RoutedLinearProbe
from xverifier.splitting import assign_roles, cross_fit_folds


def _probe_spec():
    return {"C": 1.0, "max_components": 256, "min_rows_per_verdict": 40}


class TestDrawUsesIntegerGroupIds:
    def test_counts_are_floors_of_all_groups(self):
        # GIVEN 100 groups
        group = np.arange(100)
        # WHEN the file is split three ways
        roles = assign_roles(group, mode="draw", seed=7, calibration_fraction=0.274,
                             evaluation_fraction=0.082)
        # THEN calibration and test are floors of all groups, and train is the rest
        assert len(roles["calibration"]) == int(np.floor(0.274 * 100))
        assert len(roles["test"]) == int(np.floor(0.082 * 100))
        assert len(roles["train"]) == 100 - len(roles["calibration"]) - len(roles["test"])
        assert len(set(roles["train"]) & set(roles["calibration"])) == 0
        assert len(set(roles["train"]) & set(roles["test"])) == 0

    def test_permutation_follows_the_integer_sort(self):
        # GIVEN group ids whose integer order is not their string order
        group = np.array([10, 2, 1, 10, 2, 1])
        roles = assign_roles(group, mode="files", seed=3, calibration_fraction=0.5)
        unique = np.unique(group)
        drawn = unique[np.random.RandomState(3).permutation(len(unique))]
        n_calibration = int(np.floor(0.5 * len(unique)))
        expected = set(np.flatnonzero(np.isin(group, drawn[:n_calibration])).tolist())
        # THEN the calibration rows are that integer draw
        assert set(roles["calibration"].tolist()) == expected
        string_pool = np.array(sorted(str(int(value)) for value in unique))
        string_drawn = string_pool[np.random.RandomState(3).permutation(len(string_pool))]
        assert list(string_drawn) != [str(int(value)) for value in drawn]

    def test_a_group_stays_inside_one_fold(self):
        # GIVEN eight rows in four groups, three of which have more than one row
        group = np.array([5, 5, 5, 8, 8, 9, 9, 4])
        # WHEN the rows are cross-fit in five folds
        pairs = cross_fit_folds(group, 5, 7)
        # THEN a group in a held-out fold has all of its rows there and none in that fold's fitting rows,
        # and the held-out folds together cover every row once
        held_rows = []
        for fit, held in pairs:
            for value in set(group[held].tolist()):
                rows = set(np.flatnonzero(group == value).tolist())
                assert rows <= set(held.tolist())
                assert rows.isdisjoint(fit.tolist())
            held_rows.extend(held.tolist())
        assert sorted(held_rows) == list(range(len(group)))


class TestPcaWidthFollowsTheTrainingRowCount:
    def test_n_train_over_ten_is_the_fitted_width(self):
        # GIVEN 80 training rows and 30 columns, so 8 is below both 256 and the column count
        representation = np.random.RandomState(0).normal(size=(80, 30))
        verdict = np.array([0] * 40 + [1] * 40)
        agreement = np.tile([0, 1], 40)
        # WHEN the probe is fit
        probe = RoutedLinearProbe(C=1.0, max_components=256, min_rows=40, seed=0).fit(
            representation, verdict, agreement)
        # THEN the fitted width is 80 // 10
        assert probe.n_components == 8
        assert probe.pca.components_.shape == (8, 30)


class TestThinVerdictUsesThePooledProbe:
    def test_fewer_than_forty_rows_or_a_single_class(self):
        # GIVEN a verdict with 10 rows, and another that is a single class
        rng = np.random.RandomState(1)
        thin_x = rng.normal(size=(100, 8))
        thin_verdict = np.array([0] * 10 + [1] * 90)
        thin_agreement = np.tile([0, 1], 50)
        single_x = rng.normal(size=(100, 8))
        single_verdict = np.array([0] * 50 + [1] * 50)
        single_agreement = np.array([1] * 50 + [0, 1] * 25)
        # WHEN each is fit with the 40-row floor and again with pooling forced
        for representation, verdict, agreement, thin_regime in (
            (thin_x, thin_verdict, thin_agreement, 0),
            (single_x, single_verdict, single_agreement, 0),
        ):
            routed = RoutedLinearProbe(C=1.0, max_components=256, min_rows=40, seed=1).fit(
                representation, verdict, agreement)
            forced = RoutedLinearProbe(C=1.0, max_components=256, min_rows=10**9, seed=1).fit(
                representation, verdict, agreement)
            rows = verdict == thin_regime
            # THEN that verdict has no cell of its own and receives the pooled scores
            assert routed.cells[thin_regime] is None
            assert np.allclose(routed.scores(representation, verdict)[rows],
                               forced.scores(representation, verdict)[rows])
        assert RoutedLinearProbe(C=1.0, max_components=256, min_rows=40, seed=1).fit(
            thin_x, thin_verdict, thin_agreement).cells[1] is not None


class TestThinCalibrationCellKeepsTheRawScore:
    def test_fewer_than_eight_rows_or_a_single_class(self):
        # GIVEN a seven-row cell of both classes, and a single-class cell
        few_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9, 0.95])
        few_labels = np.array([0, 0, 0, 1, 1, 1, 1])
        query = np.array([0.05, 0.5, 0.99])
        few = VennAbers(min_cell=8).fit(few_scores, few_labels, np.zeros(7, dtype=int))
        # WHEN the cell is applied
        few_out = few.probability(query, np.zeros(3, dtype=int))
        # THEN the raw query is returned unchanged
        assert np.array_equal(few_out, query)
        single_scores = np.linspace(0.1, 0.9, 20)
        single = VennAbers(min_cell=8).fit(
            single_scores, np.ones(20, dtype=int), np.zeros(20, dtype=int))
        assert np.array_equal(single.probability(query, np.zeros(3, dtype=int)), query)
        healthy_scores = np.array([0.1] * 10 + [0.9] * 10)
        healthy_labels = np.array([0] * 10 + [1] * 10)
        healthy = VennAbers(min_cell=8).fit(healthy_scores, healthy_labels, np.zeros(20, dtype=int))
        assert not np.array_equal(healthy.probability(query, np.zeros(3, dtype=int)), query)

        # GIVEN a verdict with five training rows
        representation = np.random.RandomState(2).normal(size=(65, 6))
        verdict = np.array([0] * 5 + [1] * 60)
        agreement = np.array([0, 1, 0, 1, 0] + [0, 1] * 30)
        estimator = Estimator(_probe_spec(), {"cross_fit_folds": 5}, seed=2).fit(
            Slice(representation, verdict, agreement), np.arange(65))
        raw = estimator.probe_scores(representation, verdict)
        probability = estimator.probability_of_agreement(representation, verdict)
        # THEN those rows keep the deployed raw score
        assert np.array_equal(probability[verdict == 0], raw[verdict == 0])


class TestBudgetDoesNotOverspend:
    def test_exact_sum_stays_within_the_budget(self):
        # GIVEN fifteen candidate shares of 0.05
        shares = allocate_budget(15, ERROR_BUDGET)
        naive = ERROR_BUDGET / 15
        # WHEN the binary sum is formed
        exact = sum((Fraction(float(share)) for share in shares), Fraction(0))
        # THEN it does not exceed 0.05, and the first share absorbed the leftover
        assert exact <= Fraction(ERROR_BUDGET)
        assert shares[0] < naive
        assert shares[1:] == [naive] * 14

    def test_acceptance_window_is_one_e_minus_12(self):
        cutoff = 0.8
        scores = np.array([cutoff - 1e-6, cutoff - TIE_TOLERANCE, cutoff])
        assert accepts(scores, cutoff).tolist() == [False, True, True]
