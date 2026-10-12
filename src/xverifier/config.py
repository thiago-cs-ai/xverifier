"""Read an experiment file and reject an unknown key or a missing section."""

import math
from decimal import Decimal
from pathlib import Path

import yaml

MODES = ("files", "draw", "roles")
TOP_LEVEL = ("study", "seeds", "frame", "split", "estimator", "flip", "decide")


class _UniqueKeyLoader(yaml.SafeLoader):
    """A safe loader that refuses a key written twice in one mapping, at any depth."""

    def __init__(self, stream):
        super().__init__(stream)
        self._names = {}

    def construct_mapping(self, node, deep=False):
        prefix = self._names.get(id(node), "")
        seen = set()
        for key_node, value_node in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=True)
            try:
                repeated = key in seen
            except TypeError:
                continue
            name = f"{prefix}.{key}" if prefix else str(key)
            if repeated:
                raise ValueError(
                    f"repeated key {name!r}; got it again on line {key_node.start_mark.line + 1}.")
            seen.add(key)
            self._name(value_node, name)
        return super().construct_mapping(node, deep=deep)

    def _name(self, node, name):
        if id(node) in self._names:
            return
        self._names[id(node)] = name
        if isinstance(node, yaml.SequenceNode):
            for position, item in enumerate(node.value):
                self._name(item, f"{name}[{position}]")


def load_experiment(path):
    """Load one experiment file and return the resolved settings."""
    path = Path(path)
    raw = path.read_bytes()
    document = yaml.load(raw, Loader=_UniqueKeyLoader)
    if not isinstance(document, dict):
        raise ValueError(f"experiment must be a mapping; got {type(document).__name__}.")
    _keys(document, TOP_LEVEL, (), "")
    mode = _mode(document["split"])
    frame = _frame(document["frame"], mode, path.parent)
    probe = _probe(document["estimator"])
    calibration = _calibration(document["estimator"])
    threshold = _number_in_open_unit("flip.threshold", _only(document["flip"], ("threshold",), "flip")["threshold"])
    levels = _levels(document["decide"])
    confidence = _number_in_open_unit(
        "decide.confidence", _only(document["decide"], ("error_levels", "confidence"), "decide")["confidence"])
    return {
        "study": _study(document["study"]),
        "seeds": _seeds(document["seeds"]),
        "experiment_sha256": _sha256_bytes(raw),
        "experiment_path": str(path),
        "directory": path.parent,
        "mode": mode,
        "frame": frame,
        "calibration_fraction": _fraction(document["split"]),
        "evaluation_fraction": _evaluation(document["split"], mode),
        "probe": probe,
        "calibration": calibration,
        "threshold": threshold,
        "error_levels": levels,
        "confidence": confidence,
        "error_budget": budget_of(confidence),
    }


def budget_of(confidence):
    """Return 1 minus confidence without the binary leftover of a float subtraction."""
    return float(Decimal(1) - Decimal(str(confidence)))


def _sha256_bytes(raw):
    import hashlib
    return hashlib.sha256(raw).hexdigest()


def _keys(mapping, required, optional, prefix):
    if not isinstance(mapping, dict):
        name = prefix or "experiment"
        raise ValueError(f"{name} must be a mapping; got {type(mapping).__name__}.")
    allowed = set(required) | set(optional)
    for key in sorted(mapping):
        if key not in allowed:
            name = f"{prefix}.{key}" if prefix else str(key)
            raise ValueError(f"unknown key {name!r}.")
    for key in required:
        if key not in mapping:
            name = f"{prefix}.{key}" if prefix else str(key)
            if prefix == "":
                raise ValueError(f"missing section {name!r}.")
            raise ValueError(f"missing key {name!r}.")


def _only(mapping, required, prefix):
    _keys(mapping, required, (), prefix)
    return mapping


def _study(value):
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"study must be a non-empty string; got {value!r}.")
    return value


def _seeds(value):
    if not isinstance(value, list) or not value:
        raise ValueError(f"seeds must be a non-empty list of whole numbers; got {value!r}.")
    seeds = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"seeds must be a list of whole numbers; got {item!r}.")
        seeds.append(int(item))
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be unique; got {seeds}.")
    return seeds


def _mode(split):
    _keys(split, ("mode",), ("calibration_fraction", "evaluation_fraction"), "split")
    mode = split["mode"]
    if mode not in MODES:
        raise ValueError(f"split.mode must be 'files', 'draw', or 'roles'; got {mode!r}.")
    if mode == "files":
        _keys(split, ("mode", "calibration_fraction"), (), "split")
    elif mode == "draw":
        _keys(split, ("mode", "calibration_fraction", "evaluation_fraction"), (), "split")
    else:
        _keys(split, ("mode",), (), "split")
    return mode


def _fraction(split):
    if split["mode"] == "roles":
        return None
    return _number_in_open_unit("split.calibration_fraction", split["calibration_fraction"])


def _evaluation(split, mode):
    if mode != "draw":
        return None
    return _number_in_open_unit("split.evaluation_fraction", split["evaluation_fraction"])


def _frame(frame, mode, directory):
    if mode == "files":
        _keys(frame, ("train", "test"), ("sha256",), "frame")
        files = ("train", "test")
    else:
        _keys(frame, ("path",), ("sha256",), "frame")
        files = ("path",)
    resolved = {}
    for name in files:
        resolved[name] = _path(directory, frame[name], f"frame.{name}")
    expected = frame.get("sha256")
    if expected is not None:
        _keys(expected, (), files, "frame.sha256")
    resolved["sha256"] = {name: None if expected is None else expected.get(name) for name in files}
    return resolved


def _path(directory, value, field):
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{field} must be a path; got {value!r}.")
    path = Path(value)
    if not path.is_absolute():
        path = directory / path
    return path


def _probe(estimator):
    _keys(estimator, ("probe", "calibration"), (), "estimator")
    probe = estimator["probe"]
    _keys(probe, ("family", "C", "max_components", "min_rows_per_verdict"), (), "estimator.probe")
    if probe["family"] != "linear":
        raise ValueError(f"estimator.probe.family must be 'linear'; got {probe['family']!r}.")
    strength = probe["C"]
    if isinstance(strength, bool) or not isinstance(strength, (int, float)) or float(strength) <= 0:
        raise ValueError(f"estimator.probe.C must be positive; got {strength!r}.")
    if not math.isfinite(float(strength)):
        raise ValueError(f"estimator.probe.C must be finite; got {strength!r}.")
    return {
        "family": "linear",
        "C": float(strength),
        "max_components": _positive_int("estimator.probe.max_components", probe["max_components"]),
        "min_rows_per_verdict": _positive_int(
            "estimator.probe.min_rows_per_verdict", probe["min_rows_per_verdict"]),
    }


def _calibration(estimator):
    calibration = estimator["calibration"]
    _keys(calibration, ("method", "cross_fit_folds"), (), "estimator.calibration")
    if calibration["method"] != "venn_abers":
        raise ValueError(
            f"estimator.calibration.method must be 'venn_abers'; got {calibration['method']!r}.")
    folds = calibration["cross_fit_folds"]
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError(
            f"estimator.calibration.cross_fit_folds must be an integer of at least 2; got {folds!r}.")
    return {"method": "venn_abers", "cross_fit_folds": int(folds)}


def _levels(decide):
    _keys(decide, ("error_levels", "confidence"), (), "decide")
    levels = decide["error_levels"]
    if not isinstance(levels, list) or not levels:
        raise ValueError(f"decide.error_levels must be a non-empty list; got {levels!r}.")
    parsed = []
    for level in levels:
        parsed.append(_number_in_open_unit("decide.error_levels", level))
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"decide.error_levels must be unique; got {parsed}.")
    keys = {}
    for level in parsed:
        key = f"{level:.2f}"
        if key in keys:
            raise ValueError(
                f"decide.error_levels must differ at two decimals; got {keys[key]} and {level}.")
        keys[key] = level
    return parsed


def _number_in_open_unit(field, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be in (0, 1); got {value!r}.")
    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{field} must be in (0, 1); got {value}.")
    return value


def _positive_int(field, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer; got {value!r}.")
    return int(value)
