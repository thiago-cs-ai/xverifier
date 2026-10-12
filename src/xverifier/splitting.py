"""Group-blocked roles and group-blocked cross-fitting folds."""

import numpy as np

ROLES = ("train", "calibration", "test")


def assign_roles(group, *, mode, seed, calibration_fraction, evaluation_fraction=None, role=None):
    """Return row indices for train, calibration, and test."""
    group = np.asarray(group)
    if mode == "roles":
        found = _roles_from_column(role, len(group))
        _require_one_role_per_group(group, found)
        _require_training_groups(len(np.unique(group[found["train"]])))
        return found
    if mode not in ("files", "draw"):
        raise ValueError(f"split.mode must be 'files', 'draw', or 'roles'; got {mode!r}.")
    drawn = _permute_groups(group, seed)
    n_groups = int(len(drawn))
    n_calibration = int(np.floor(float(calibration_fraction) * n_groups))
    if mode == "files":
        if n_calibration >= n_groups:
            raise ValueError(
                "split.calibration_fraction leaves no training group; "
                f"got {calibration_fraction}.")
        _require_group(n_calibration, "calibration", calibration_fraction, n_groups)
        calibration_ids = drawn[:n_calibration]
        train_ids = drawn[n_calibration:]
        _require_training_groups(len(train_ids))
        return {
            "train": _rows_of(group, train_ids),
            "calibration": _rows_of(group, calibration_ids),
            "test": None,
        }
    if evaluation_fraction is None:
        raise ValueError("split.evaluation_fraction is missing; got None.")
    n_test = int(np.floor(float(evaluation_fraction) * n_groups))
    if n_calibration + n_test >= n_groups:
        raise ValueError(
            "split.calibration_fraction and split.evaluation_fraction leave no training group; "
            f"got {calibration_fraction} and {evaluation_fraction}.")
    _require_group(n_calibration, "calibration", calibration_fraction, n_groups)
    _require_group(n_test, "test", evaluation_fraction, n_groups)
    calibration_ids = drawn[:n_calibration]
    test_ids = drawn[n_calibration:n_calibration + n_test]
    train_ids = drawn[n_calibration + n_test:]
    _require_training_groups(len(train_ids))
    return {
        "train": _rows_of(group, train_ids),
        "calibration": _rows_of(group, calibration_ids),
        "test": _rows_of(group, test_ids),
    }


def cross_fit_folds(group, folds, seed):
    """Put each whole group in one fold and return fit and held-out indices."""
    group = np.asarray(group)
    folds = int(folds)
    if folds < 2:
        raise ValueError(f"cross_fit_folds must be at least 2; got {folds}.")
    unique = np.unique(group)
    try:
        permutation = np.random.RandomState(int(seed)).permutation(len(unique))
    except ValueError as exc:
        raise ValueError(f"seed must be a valid random seed; got {seed}.") from exc
    fold_of = {int(unique[permutation[position]]): position % folds
               for position in range(len(unique))}
    assigned = np.array([fold_of[int(one)] for one in group], dtype=np.int64)
    pairs = []
    for fold in range(folds):
        held_out = np.flatnonzero(assigned == fold)
        if len(held_out) == 0:
            continue
        fitting = np.flatnonzero(assigned != fold)
        pairs.append((fitting, held_out))
    return pairs


def _permute_groups(group, seed):
    unique = np.unique(group)
    try:
        order = np.random.RandomState(int(seed)).permutation(len(unique))
    except ValueError as exc:
        raise ValueError(f"seed must be a valid random seed; got {seed}.") from exc
    return unique[order]


def _rows_of(group, chosen):
    return np.flatnonzero(np.isin(group, chosen))


def _require_group(count, role, fraction, n_groups):
    if count == 0:
        field = "split.calibration_fraction" if role == "calibration" else "split.evaluation_fraction"
        noun = "group" if n_groups == 1 else "groups"
        raise ValueError(f"{field} leaves no {role} group; got {fraction} and {n_groups} {noun}.")


def _require_training_groups(n_groups):
    if n_groups < 2:
        raise ValueError(f"training slice must span at least two groups; got {n_groups}.")


def _require_one_role_per_group(group, found):
    for position, first in enumerate(ROLES):
        for second in ROLES[position + 1:]:
            shared = np.intersect1d(group[found[first]], group[found[second]])
            if len(shared):
                raise ValueError(
                    f"group {shared[0].item()} must have one role; got {first!r} and {second!r}.")


def _roles_from_column(role, n_rows):
    if role is None:
        raise ValueError("role is missing; got None.")
    role = np.asarray(role).astype(str)
    if role.ndim != 1 or len(role) != n_rows:
        got = len(role) if role.ndim == 1 else role.shape
        raise ValueError(f"role length must equal representation length; got {got} and {n_rows}.")
    bad = [value for value in role if value not in ROLES]
    if bad:
        raise ValueError(f"role must be 'train', 'calibration', or 'test'; got {bad[0]!r}.")
    found = {value: np.flatnonzero(role == value) for value in ROLES}
    for name in ROLES:
        if len(found[name]) == 0:
            raise ValueError(f"role must include {name!r}; got no {name} rows.")
    return found
