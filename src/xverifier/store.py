"""Write a model and records once, as arrays and canonical JSON."""

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import sklearn

from xverifier.calibrations import VennAbers
from xverifier.probes import RoutedLinearProbe


def write_json(path, payload):
    """Write canonical JSON without replacing an existing file."""
    path = Path(path)
    if path.exists():
        raise ValueError(f"output file already exists; got {str(path)!r}.")
    path.write_text(json.dumps(as_jsonable(payload), sort_keys=True, indent=2,
                               separators=(",", ": ")) + "\n")


def read_json(path):
    """Read one JSON record."""
    return json.loads(Path(path).read_text())


def save_model(directory, probe, calibration, cutoffs, meta):
    """Write model.npz, cutoffs.json, and meta.json."""
    directory = Path(directory)
    payload = {}
    payload.update(probe.arrays())
    payload.update(calibration.arrays())
    path = directory / "model.npz"
    if path.exists():
        raise ValueError(f"output file already exists; got {str(path)!r}.")
    np.savez(path, **payload)
    write_json(directory / "cutoffs.json", {"cutoffs": cutoffs})
    write_json(directory / "meta.json", meta)


def load_model(directory):
    """Load a model saved as arrays."""
    directory = Path(directory)
    model_path = directory / "model.npz"
    if not model_path.is_file():
        raise ValueError(f"model file must exist; got {str(model_path)!r}.")
    with np.load(model_path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    meta = read_json(directory / "meta.json")
    cutoffs = read_json(directory / "cutoffs.json")["cutoffs"]
    try:
        probe = RoutedLinearProbe.from_arrays(arrays)
        calibration = VennAbers.from_arrays(arrays)
    except KeyError as exc:
        raise ValueError(
            f"model must be retrained; got {str(model_path)!r} without {exc.args[0]}.") from exc
    return {"probe": probe, "calibration": calibration, "cutoffs": cutoffs, "meta": meta}


def code_sha256():
    """Digest the package sources in name order."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for source in sorted(root.glob("*.py")):
        digest.update(source.name.encode("utf-8"))
        digest.update(source.read_bytes())
    return digest.hexdigest()


def environment():
    """Return the numerical-library versions used by a run."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }


def as_jsonable(value):
    """Convert NumPy values into JSON values."""
    if isinstance(value, dict):
        return {str(key): as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return as_jsonable(value.tolist())
    if isinstance(value, np.floating):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
