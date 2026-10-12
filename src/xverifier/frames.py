"""Load row files and slice them for the estimator."""

import hashlib
from typing import NamedTuple

import numpy as np

from xverifier.guards import assert_binary, assert_representation, groups_of

ARRAY_FIELDS = ("representation", "label", "verdict", "agreement", "group")


class Slice(NamedTuple):
    """Estimator inputs: representation, verdict, and agreement arrays."""
    representation: np.ndarray
    verdict: np.ndarray
    agreement: np.ndarray


def sha256_of(path):
    """Return the SHA-256 hex digest of a file."""
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_frame(path, *, expected_sha256=None, sha_field="frame"):
    """Load one row file, check it, and replace negative group ids."""
    digest = sha256_of(path)
    if expected_sha256 is not None and digest != str(expected_sha256).lower():
        raise ValueError(f"{sha_field} must be {expected_sha256}; got {digest}.")
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    if "representation" not in arrays:
        raise ValueError("representation is missing; got no representation array.")
    representation = assert_representation(arrays["representation"])
    n_rows = int(representation.shape[0])
    for name in ("label", "verdict"):
        if name not in arrays:
            raise ValueError(f"{name} is missing; got no {name} array.")
    label = assert_binary("label", arrays["label"], n_rows)
    verdict = assert_binary("verdict", arrays["verdict"], n_rows)
    group = groups_of(arrays.get("group"), n_rows)
    role = arrays.get("role")
    if role is not None:
        role = np.asarray(role)
        if role.ndim != 1 or len(role) != n_rows:
            got = len(role) if role.ndim == 1 else role.shape
            raise ValueError(f"role length must equal representation length; got {got} and {n_rows}.")
        role = np.array([str(value) for value in role])
    agreement = np.ascontiguousarray(label == verdict, dtype=np.int8)
    return {
        "representation": representation,
        "label": label,
        "verdict": verdict,
        "agreement": agreement,
        "group": group,
        "role": role,
        "sha256": digest,
        "path": str(path),
    }


def take(frame, index):
    """Return the rows selected by index."""
    chosen = dict(frame)
    for name in ARRAY_FIELDS:
        chosen[name] = frame[name][index]
    if frame["role"] is not None:
        chosen["role"] = frame["role"][index]
    return chosen


def slice_of(frame):
    """Return estimator inputs for a frame."""
    return Slice(frame["representation"], frame["verdict"], frame["agreement"])
