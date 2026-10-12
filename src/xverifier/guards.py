"""Checks that run before a fit."""

import os

import numpy as np

_PINNED = False
_INT64_MAX = int(np.iinfo(np.int64).max)


def pin_threads():
    """Pin BLAS threads to one so a fit does not depend on the machine."""
    global _PINNED
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    if _PINNED:
        return
    _PINNED = True
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1).__enter__()
    except Exception:
        return


def assert_representation(representation):
    """Require a finite two-dimensional representation."""
    representation = np.asarray(representation)
    if representation.ndim != 2:
        raise ValueError(f"representation must be a 2-d array; got {representation.ndim}.")
    if representation.shape[0] == 0 or representation.shape[1] == 0:
        raise ValueError(f"representation must contain rows and columns; got {representation.shape}.")
    finite = np.isfinite(representation)
    if not finite.all():
        bad = representation[~finite].reshape(-1)[0]
        shown = bad.item() if hasattr(bad, "item") else bad
        raise ValueError(f"representation contains a non-finite value; got {shown}.")
    return representation


def assert_binary(name, values, n_rows):
    """Require a 0/1 column with one entry per representation row."""
    values = np.asarray(values)
    if values.ndim != 1 or len(values) != n_rows:
        got = len(values) if values.ndim == 1 else values.shape
        raise ValueError(f"{name} length must equal representation length; got {got} and {n_rows}.")
    _require_zero_or_one(name, values)
    return np.ascontiguousarray(values, dtype=np.int8)


def assert_verdict(verdict):
    """Require every verdict to be 0 or 1, so each row goes to a probe or a calibration cell."""
    verdict = np.asarray(verdict)
    _require_zero_or_one("verdict", verdict)
    return verdict


def groups_of(group, n_rows):
    """Give each negative group id its own id above the largest real id, in row order."""
    if group is None:
        return np.arange(n_rows, dtype=np.int64)
    group = np.asarray(group)
    if group.ndim != 1 or len(group) != n_rows:
        got = len(group) if group.ndim == 1 else group.shape
        raise ValueError(f"group length must equal representation length; got {got} and {n_rows}.")
    if group.dtype == np.bool_ or not np.issubdtype(group.dtype, np.integer):
        raise ValueError(f"group must be an integer array; got {group.dtype}.")
    if group.dtype.kind == "u" and len(group) and int(group.max()) > _INT64_MAX:
        raise ValueError(f"group must be at most {_INT64_MAX}; got {int(group.max())}.")
    owned = np.ascontiguousarray(group, dtype=np.int64)
    if np.any(owned < 0):
        real = owned >= 0
        if not np.any(real):
            return np.arange(n_rows, dtype=np.int64)
        negative = np.flatnonzero(~real)
        largest = int(owned[real].max())
        if largest + len(negative) > _INT64_MAX:
            rows = "row" if len(negative) == 1 else "rows"
            raise ValueError(
                f"group must leave room above its largest id for {len(negative)} negative-id {rows}; "
                f"got {largest}.")
        start = largest + 1
        owned = owned.copy()
        owned[negative] = np.arange(start, start + len(negative), dtype=np.int64)
    return owned


def _require_zero_or_one(name, values):
    valid = np.isin(values, (0, 1))
    if not valid.all():
        bad = values[~valid].reshape(-1)[0]
        shown = bad.item() if hasattr(bad, "item") else bad
        raise ValueError(f"{name} must be in {{0, 1}}; got {shown}.")
