"""Generate the synthetic rows used by the example and the tests."""

import numpy as np


def training_rows():
    """Return the 900-row training draw."""
    rng = np.random.RandomState(7)
    representation = rng.normal(size=(900, 32)).astype(np.float32)
    label = (rng.rand(900) < 0.46).astype(np.int8)
    agree = (representation[:, 0] + 0.15 * rng.normal(size=900) > 0).astype(np.int8)
    verdict = np.where(agree == 1, label, 1 - label).astype(np.int8)
    group = np.arange(900)
    return representation, label, verdict, group


def test_rows():
    """Return the 300-row test draw."""
    rng = np.random.RandomState(11)
    representation = rng.normal(size=(300, 32)).astype(np.float32)
    label = (rng.rand(300) < 0.46).astype(np.int8)
    agree = (representation[:, 0] + 0.15 * rng.normal(size=300) > 0).astype(np.int8)
    verdict = np.where(agree == 1, label, 1 - label).astype(np.int8)
    group = np.arange(300)
    return representation, label, verdict, group


def save_rows(path, representation, label, verdict, group=None, role=None):
    """Write one row file."""
    payload = {"representation": representation, "label": label, "verdict": verdict}
    if group is not None:
        payload["group"] = np.asarray(group)
    if role is not None:
        payload["role"] = np.asarray(role)
    np.savez(path, **payload)
