"""Choose confidence cutoffs and count the rows that clear them."""

import math
from fractions import Fraction

import numpy as np
from scipy.stats import beta

from xverifier.constants import ERROR_BUDGET, GRID_LEVELS, TIE_TOLERANCE


def candidate_grid(train_confidence, levels=GRID_LEVELS):
    """Return ascending unique quantiles of the training confidence scores."""
    scores = np.asarray(train_confidence, dtype=float)
    if scores.size == 0:
        return np.array([], dtype=float)
    if not np.isfinite(scores).all():
        bad = scores[~np.isfinite(scores)].reshape(-1)[0]
        shown = bad.item() if hasattr(bad, "item") else bad
        raise ValueError(f"training confidence contains a non-finite value; got {shown}.")
    quantiles = np.quantile(scores, list(levels))
    if not np.isfinite(quantiles).all():
        bad = quantiles[~np.isfinite(quantiles)].reshape(-1)[0]
        shown = bad.item() if hasattr(bad, "item") else bad
        raise ValueError(f"candidate cutoff is not finite; got {shown}.")
    first = {}
    for cutoff in quantiles:
        cutoff = float(cutoff)
        if cutoff not in first:
            first[cutoff] = cutoff
    return np.array(sorted(first), dtype=float)


def allocate_budget(n_candidates, budget=ERROR_BUDGET):
    """Split the budget across candidates without letting the exact sum exceed it."""
    n_candidates = int(n_candidates)
    budget = float(budget)
    if n_candidates < 1:
        raise ValueError(f"candidate count must be at least 1; got {n_candidates}.")
    if not math.isfinite(budget) or budget <= 0.0:
        raise ValueError(f"error budget must be finite and positive; got {budget}.")
    shares = [budget / n_candidates] * n_candidates
    return _keep_within_budget(shares, budget)


def select_cutoffs(train_confidence, calibration_confidence, calibration_error, error_levels, budget):
    """Pick the most permissive cutoff whose upper bound is at most each error level."""
    calibration_confidence = np.asarray(calibration_confidence, dtype=float)
    calibration_error = np.asarray(calibration_error, dtype=int)
    n_rows = int(calibration_confidence.size)
    n_errors = int(calibration_error.sum()) if n_rows else 0
    n_clean = n_rows - n_errors
    if n_rows == 0:
        return [_review(level, "empty_cell") for level in error_levels], []
    cutoffs = candidate_grid(train_confidence)
    if cutoffs.size == 0:
        return [_review(level, "empty_train_grid") for level in error_levels], []
    shares = allocate_budget(int(cutoffs.size), budget)
    band = []
    for index, (cutoff, share) in enumerate(zip(cutoffs, shares)):
        accepted = accepts(calibration_confidence, cutoff)
        decided = int(accepted.sum())
        errors = int(calibration_error[accepted].sum()) if decided else 0
        band.append({
            "scan_index": index,
            "cutoff": float(cutoff),
            "budget": float(share),
            "decided": decided,
            "errors": errors,
            "upper_bound": upper_bound(errors, decided, share),
        })
    chosen = []
    for level in error_levels:
        level = float(level)
        minimum = min(_minimum_clean(level, share) for share in shares)
        hit = next((row for row in band if row["upper_bound"] <= level), None)
        if hit is None:
            reason = "SR" if n_rows < minimum else ("AR" if n_clean < minimum else "R")
            chosen.append(_review(level, reason, minimum_clean=minimum))
        else:
            chosen.append({
                "error_level": level,
                "review_all": False,
                "cutoff": hit["cutoff"],
                "upper_bound": hit["upper_bound"],
                "reason": None,
                "calibration_decided": hit["decided"],
                "calibration_errors": hit["errors"],
                "calibration_rows": n_rows,
                "budget": hit["budget"],
                "scan_index": hit["scan_index"],
                "minimum_clean": minimum,
            })
    return chosen, band


def count_decisions(cutoffs, confidence, error):
    """Count decided rows and errors at each cutoff."""
    confidence = np.asarray(confidence, dtype=float)
    error = np.asarray(error, dtype=int)
    n_rows = int(confidence.size)
    counted = []
    for cutoff in cutoffs:
        point = dict(cutoff)
        point["rows"] = n_rows
        if cutoff["review_all"] or cutoff["cutoff"] is None:
            point["decided"] = 0
            point["errors"] = None
            point["error_rate"] = None
        else:
            accepted = accepts(confidence, cutoff["cutoff"])
            decided = int(accepted.sum())
            errors = int(error[accepted].sum()) if decided else 0
            point["decided"] = decided
            point["errors"] = errors
            point["error_rate"] = None if decided == 0 else errors / decided
        counted.append(point)
    return counted


def accepts(scores, cutoff):
    """Return rows whose score is at least the cutoff minus the tie tolerance."""
    return np.asarray(scores, dtype=float) >= float(cutoff) - TIE_TOLERANCE


def upper_bound(errors, decided, share):
    """Return the one-sided Clopper-Pearson bound, or 1 for an empty or all-error prefix."""
    decided = int(decided)
    errors = int(errors)
    if decided <= 0 or errors == decided:
        return 1.0
    if errors < 0 or errors > decided:
        raise ValueError(f"errors must be in [0, {decided}]; got {errors}.")
    return float(beta.ppf(1.0 - float(share), errors + 1, decided - errors))


def _minimum_clean(level, share):
    if not 0.0 < float(level) < 1.0:
        raise ValueError(f"error level must be in (0, 1); got {level}.")
    return math.ceil(math.log(1.0 / float(share)) / math.log(1.0 / (1.0 - float(level))))


def _review(level, reason, minimum_clean=None):
    return {
        "error_level": float(level),
        "review_all": True,
        "cutoff": None,
        "upper_bound": None,
        "reason": reason,
        "calibration_decided": None,
        "calibration_errors": None,
        "calibration_rows": None,
        "budget": None,
        "scan_index": None,
        "minimum_clean": minimum_clean,
    }


def _keep_within_budget(shares, budget):
    # The first candidate gives up the leftover so the exact binary sum stays within the budget.
    shares = [float(share) for share in shares]
    if not shares:
        return shares
    cap = Fraction(float(budget))
    if _exact_sum(shares) > cap:
        shares[0] = shares[0] - float(_exact_sum(shares) - cap)
        guard = 0
        while _exact_sum(shares) > cap:
            shares[0] = math.nextafter(shares[0], 0.0)
            guard += 1
            if guard > 128:
                raise ValueError(
                    f"error budget cannot cover the candidate shares; got {budget}.")
    if not all(math.isfinite(share) and share > 0.0 for share in shares):
        raise ValueError(f"error budget produced a nonpositive share; got {budget}.")
    return shares


def _exact_sum(shares):
    return sum((Fraction(float(share)) for share in shares), Fraction(0))
