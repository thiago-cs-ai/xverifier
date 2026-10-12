"""Keep a verdict at or above the threshold, and flip it otherwise."""

import numpy as np


class FlipRule:
    """Turn an agreement probability into a keep-or-flip confidence score."""

    def __init__(self, threshold):
        if not 0.0 < float(threshold) < 1.0:
            raise ValueError(f"flip.threshold must be in (0, 1); got {threshold}.")
        self.threshold = float(threshold)

    def apply(self, probability, agreement):
        """Return confidence, error, and whether each row flips."""
        probability = np.asarray(probability, dtype=float)
        agreement = np.asarray(agreement, dtype=int)
        flipped = probability < self.threshold
        confidence = np.where(flipped, 1.0 - probability, probability)
        error = np.where(flipped, agreement, 1 - agreement).astype(int)
        return confidence, error, flipped

    def corrected_verdict(self, verdict, probability):
        """Return the kept or flipped verdict for each row."""
        verdict = np.asarray(verdict)
        probability = np.asarray(probability, dtype=float)
        return np.where(probability < self.threshold, 1 - verdict, verdict)
