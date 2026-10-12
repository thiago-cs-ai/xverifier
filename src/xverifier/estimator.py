"""Fit the probe and calibrate it on out-of-fold training scores."""

import numpy as np

from xverifier.calibrations import VennAbers
from xverifier.constants import CALIBRATION_MIN_CELL
from xverifier.frames import Slice
from xverifier.guards import pin_threads
from xverifier.probes import RoutedLinearProbe
from xverifier.splitting import cross_fit_folds


class Estimator:
    """Routed linear probe with Venn-Abers fit on out-of-fold scores."""

    def __init__(self, probe, calibration, seed):
        self._probe_spec = dict(probe)
        self._folds = int(calibration["cross_fit_folds"])
        self._min_cell = int(calibration.get("min_cell", CALIBRATION_MIN_CELL))
        self._seed = int(seed)
        self.probe = None
        self.calibration = None
        self.unscored_out_of_fold = 0

    def fit(self, training, group):
        """Fit the probe on these rows and calibrate it on out-of-fold scores."""
        pin_threads()
        if not isinstance(training, Slice):
            training = Slice(*training)
        group = np.asarray(group)
        if len(group) != len(training.verdict):
            raise ValueError(
                f"group length must equal representation length; got {len(group)} and "
                f"{len(training.verdict)}.")
        self.probe = _new_probe(self._probe_spec, self._seed).fit(
            training.representation, training.verdict, training.agreement)
        out_of_fold = np.full(len(group), np.nan, dtype=np.float64)
        for fit_index, score_index in cross_fit_folds(group, self._folds, self._seed):
            if len(fit_index) == 0:
                continue
            fold = _new_probe(self._probe_spec, self._seed).fit(
                training.representation[fit_index],
                training.verdict[fit_index],
                training.agreement[fit_index],
            )
            out_of_fold[score_index] = fold.scores(
                training.representation[score_index], training.verdict[score_index])
        scored = ~np.isnan(out_of_fold)
        self.unscored_out_of_fold = int((~scored).sum())
        if self.unscored_out_of_fold:
            raise ValueError(
                "out-of-fold pass must score every training row; "
                f"got {self.unscored_out_of_fold} of {len(group)} unscored.")
        self.calibration = VennAbers(self._min_cell).fit(
            out_of_fold[scored], training.agreement[scored], training.verdict[scored])
        return self

    def probe_scores(self, representation, verdict):
        """Return the raw probability that each verdict agrees with the label."""
        return self.probe.scores(representation, verdict)

    def probability_of_agreement(self, representation, verdict):
        """Return the calibrated probability that each verdict agrees with the label."""
        raw = self.probe_scores(representation, verdict)
        return self.calibration.probability(raw, verdict)


def _new_probe(spec, seed):
    return RoutedLinearProbe(
        C=spec["C"],
        max_components=spec["max_components"],
        min_rows=spec["min_rows_per_verdict"],
        seed=seed,
    )
