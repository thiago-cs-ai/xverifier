"""Map raw agreement scores to probabilities with Venn-Abers."""

import numpy as np
from sklearn.isotonic import IsotonicRegression

from xverifier.constants import CALIBRATION_MIN_CELL
from xverifier.guards import assert_verdict


class VennAbers:
    """One inductive Venn-Abers map per verdict, fit on out-of-fold scores."""

    def __init__(self, min_cell=CALIBRATION_MIN_CELL):
        self.min_cell = int(min_cell)
        self.cells = {}

    def fit(self, scores, agreement, verdict):
        """Store one calibration cell per verdict; return self."""
        scores = np.asarray(scores, dtype=np.float64)
        agreement = np.asarray(agreement)
        verdict = np.asarray(verdict)
        for regime in (0, 1):
            mask = verdict == regime
            self.cells[regime] = (scores[mask].copy(), np.asarray(agreement[mask], dtype=int).copy())
        return self

    def probability(self, scores, verdict):
        """Return calibrated agreement probabilities, or the raw score for a thin cell."""
        scores = np.asarray(scores, dtype=np.float64)
        verdict = assert_verdict(verdict)
        probabilities = np.empty(len(scores), dtype=np.float64)
        for regime in (0, 1):
            mask = verdict == regime
            if not np.any(mask):
                continue
            cell_scores, cell_agreement = self.cells.get(regime, (_empty(), _empty_labels()))
            probabilities[mask] = _cell_probability(
                cell_scores, cell_agreement, scores[mask], self.min_cell)
        return probabilities

    def arrays(self):
        """Return the calibration cells."""
        payload = {"min_cell": np.array(self.min_cell, dtype=np.int64)}
        for regime in (0, 1):
            scores, agreement = self.cells.get(regime, (_empty(), _empty_labels()))
            payload[f"cal{regime}_scores"] = np.asarray(scores, dtype=np.float64)
            payload[f"cal{regime}_agreement"] = np.asarray(agreement, dtype=np.int64)
        return payload

    @classmethod
    def from_arrays(cls, stored):
        """Rebuild a calibration from arrays saved without pickle."""
        calibration = cls(min_cell=int(stored["min_cell"]))
        for regime in (0, 1):
            calibration.cells[regime] = (
                np.asarray(stored[f"cal{regime}_scores"], dtype=np.float64),
                np.asarray(stored[f"cal{regime}_agreement"], dtype=int),
            )
        return calibration


def _empty():
    return np.empty(0, dtype=np.float64)


def _empty_labels():
    return np.empty(0, dtype=int)


def _cell_probability(cell_scores, cell_agreement, query, min_cell):
    query = np.asarray(query, dtype=np.float64)
    cell_scores = np.asarray(cell_scores, dtype=np.float64)
    cell_agreement = np.asarray(cell_agreement, dtype=int)
    if len(cell_scores) >= min_cell and len(set(cell_agreement.tolist())) > 1:
        intervals = _intervals(cell_scores, cell_agreement, query)
        return _point(intervals[:, 0], intervals[:, 1])
    return query.copy()


def _intervals(cell_scores, cell_agreement, query):
    intervals = np.zeros((len(query), 2), dtype=np.float64)
    for index, score in enumerate(query):
        low, high = _one_interval(cell_scores, cell_agreement, float(score))
        intervals[index, 0] = low
        intervals[index, 1] = high
    return intervals


def _one_interval(cell_scores, cell_agreement, score):
    low = _isotonic_at(cell_scores, cell_agreement, score, 0)
    high = _isotonic_at(cell_scores, cell_agreement, score, 1)
    if low > high:
        low, high = high, low
    return low, high


def _isotonic_at(cell_scores, cell_agreement, score, added_label):
    scores = np.append(cell_scores, score)
    labels = np.append(cell_agreement, added_label)
    regressor = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    regressor.fit(scores, labels)
    return float(regressor.predict([score])[0])


def _point(low, high):
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    return high / np.maximum(1e-9, 1.0 - low + high)
