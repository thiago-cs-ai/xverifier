"""Fit each seed, write its record, and then write the aggregate."""

import os
import secrets
import shutil
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from xverifier.config import load_experiment
from xverifier.decide import count_decisions, select_cutoffs
from xverifier.estimator import Estimator
from xverifier.flip import FlipRule
from xverifier.frames import load_frame, slice_of, take
from xverifier.splitting import assign_roles
from xverifier.store import code_sha256, environment, save_model, write_json

ROLES = ("train", "calibration", "test")


def run(experiment_path, output_dir):
    """Run every seed, write one record each, and return the per-row probabilities."""
    experiment = load_experiment(experiment_path)
    frames = read_frames(experiment)
    _check_roles(experiment, frames, experiment["seeds"])
    results = {}
    records = []
    with _staged_directory(output_dir) as output:
        for seed in experiment["seeds"]:
            result = execute(experiment, frames, seed)
            record = record_of(experiment, frames, seed, result)
            write_json(output / f"seed_{seed}.json", record)
            records.append(record)
            results[seed] = result
        write_json(output / "aggregate.json", aggregate_of(experiment, records))
    return results


def train(experiment_path, seed, output_dir):
    """Fit one seed and write the model."""
    experiment = load_experiment(experiment_path)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be a whole number; got {seed!r}.")
    frames = read_frames(experiment)
    _check_roles(experiment, frames, [int(seed)])
    with _staged_directory(output_dir) as output:
        result = execute(experiment, frames, int(seed))
        save_model(
            output,
            result["estimator"].probe,
            result["estimator"].calibration,
            [_cutoff_record(point) for point in result["cutoffs"]],
            {
                "study": experiment["study"],
                "seed": int(seed),
                "threshold": experiment["threshold"],
                "pca_dim": result["pca_dim"],
                "probe": experiment["probe"],
                "calibration": experiment["calibration"],
                "error_levels": experiment["error_levels"],
                "confidence": experiment["confidence"],
                "error_budget": experiment["error_budget"],
                "experiment_sha256": experiment["experiment_sha256"],
                "row_sha256": {name: frame["sha256"] for name, frame in frames.items()},
                "code_sha256": code_sha256(),
                "environment": environment(),
            },
        )
    return result


def score_file(rows_path, model):
    """Apply a saved model to a row file and return the operating points."""
    frame = load_frame(rows_path)
    rule = FlipRule(model["meta"]["threshold"])
    probability = model["calibration"].probability(
        model["probe"].scores(frame["representation"], frame["verdict"]), frame["verdict"])
    confidence, error, flipped = rule.apply(probability, frame["agreement"])
    points = count_decisions(model["cutoffs"], confidence, error)
    return {
        "operating_points": points,
        "probability": probability,
        "confidence": confidence,
        "error": error,
        "flipped": flipped,
        "frame": frame,
    }


def execute(experiment, frames, seed):
    """Fit one seed on frames that are already loaded."""
    roles = _roles(experiment, frames, seed)
    training = take(roles["source"]["train"], roles["index"]["train"])
    estimator = Estimator(experiment["probe"], experiment["calibration"], seed).fit(
        slice_of(training), training["group"])
    rule = FlipRule(experiment["threshold"])
    scored = {}
    for role in ROLES:
        frame = take(roles["source"][role], roles["index"][role])
        raw = estimator.probe_scores(frame["representation"], frame["verdict"])
        probability = estimator.calibration.probability(raw, frame["verdict"])
        confidence, error, flipped = rule.apply(probability, frame["agreement"])
        scored[role] = {
            "raw": raw,
            "probability": probability,
            "confidence": confidence,
            "error": error,
            "flipped": flipped,
            "verdict": frame["verdict"],
            "agreement": frame["agreement"],
            "fallback_rows": int(estimator.probe.pooled_mask(frame["verdict"]).sum()),
        }
    cutoffs, candidates = select_cutoffs(
        scored["train"]["confidence"],
        scored["calibration"]["confidence"],
        scored["calibration"]["error"],
        experiment["error_levels"],
        experiment["error_budget"],
    )
    points = count_decisions(cutoffs, scored["test"]["confidence"], scored["test"]["error"])
    return {
        "seed": int(seed),
        "index": roles["index"],
        "source_name": roles["source_name"],
        "rows": {role: int(len(roles["index"][role])) for role in ROLES},
        "probability": {role: scored[role]["probability"] for role in ROLES},
        "raw": {role: scored[role]["raw"] for role in ROLES},
        "confidence": {role: scored[role]["confidence"] for role in ROLES},
        "error": {role: scored[role]["error"] for role in ROLES},
        "flipped": {role: scored[role]["flipped"] for role in ROLES},
        "verdict": {role: scored[role]["verdict"] for role in ROLES},
        "agreement": {role: scored[role]["agreement"] for role in ROLES},
        "fallback_rows": {role: scored[role]["fallback_rows"] for role in ROLES},
        "cutoffs": cutoffs,
        "candidates": candidates,
        "operating_points": points,
        "pca_dim": estimator.probe.n_components,
        "unscored_out_of_fold": estimator.unscored_out_of_fold,
        "estimator": estimator,
    }


def read_frames(experiment):
    """Load the row files named by an experiment."""
    loaded = {}
    for name, path in experiment["frame"].items():
        if name == "sha256":
            continue
        if not Path(path).is_file():
            raise ValueError(f"frame.{name} must be an existing file; got {str(path)!r}.")
        loaded[name] = load_frame(
            path,
            expected_sha256=experiment["frame"]["sha256"].get(name),
            sha_field=f"frame.sha256.{name}",
        )
    return loaded


def record_of(experiment, frames, seed, result):
    """Build the JSON record for one seed."""
    return {
        "study": experiment["study"],
        "seed": int(seed),
        "experiment_sha256": experiment["experiment_sha256"],
        "row_sha256": {name: frame["sha256"] for name, frame in frames.items()},
        "pca_dim": result["pca_dim"],
        "rows": result["rows"],
        "fallback_rows": result["fallback_rows"],
        "unscored_out_of_fold": result["unscored_out_of_fold"],
        "operating_points": result["operating_points"],
        "candidates": result["candidates"],
        "settings": {
            "mode": experiment["mode"],
            "calibration_fraction": experiment["calibration_fraction"],
            "evaluation_fraction": experiment["evaluation_fraction"],
            "probe": experiment["probe"],
            "calibration": experiment["calibration"],
            "threshold": experiment["threshold"],
            "error_levels": experiment["error_levels"],
            "confidence": experiment["confidence"],
            "error_budget": experiment["error_budget"],
        },
        "code_sha256": code_sha256(),
        "environment": environment(),
    }


def aggregate_of(experiment, records):
    """Average coverage over seeds, and the error rate over seeds that decided rows."""
    levels = {}
    for level in experiment["error_levels"]:
        key = f"{float(level):.2f}"
        points = [_point(record, level) for record in records]
        certified = [point for point in points if not point["review_all"]]
        coverage = [0.0 if point["review_all"] or point["rows"] == 0
                    else point["decided"] / point["rows"] for point in points]
        error_rates = [point["error_rate"] for point in certified if point["error_rate"] is not None]
        coverage_mean, coverage_std = _mean_std(coverage)
        error_mean, error_std = _mean_std(error_rates)
        levels[key] = {
            "error_level": float(level),
            "n_seeds": len(records),
            "n_certified": len(certified),
            "coverage_mean": coverage_mean,
            "coverage_std": coverage_std,
            "error_rate_mean": error_mean,
            "error_rate_std": error_std,
        }
    return {
        "study": experiment["study"],
        "seeds": list(experiment["seeds"]),
        "experiment_sha256": experiment["experiment_sha256"],
        "row_sha256": records[0]["row_sha256"] if records else {},
        "error_levels": levels,
        "code_sha256": code_sha256(),
        "environment": environment(),
    }


def _roles(experiment, frames, seed):
    mode = experiment["mode"]
    if mode == "files":
        assigned = assign_roles(
            frames["train"]["group"],
            mode="files",
            seed=seed,
            calibration_fraction=experiment["calibration_fraction"],
        )
        return {
            "index": {
                "train": assigned["train"],
                "calibration": assigned["calibration"],
                "test": np.arange(len(frames["test"]["representation"])),
            },
            "source": {
                "train": frames["train"],
                "calibration": frames["train"],
                "test": frames["test"],
            },
            "source_name": {"train": "train", "calibration": "train", "test": "test"},
        }
    frame = frames["path"]
    assigned = assign_roles(
        frame["group"],
        mode=mode,
        seed=seed,
        calibration_fraction=experiment["calibration_fraction"],
        evaluation_fraction=experiment["evaluation_fraction"],
        role=frame["role"],
    )
    return {
        "index": assigned,
        "source": {role: frame for role in ROLES},
        "source_name": {role: "path" for role in ROLES},
    }


def _check_roles(experiment, frames, seeds):
    """Assign the roles of every seed, so a bad split fails before any output directory exists."""
    for seed in seeds:
        _roles(experiment, frames, seed)


@contextmanager
def _staged_directory(path):
    """Yield a temporary sibling of path, and rename it to path only after the last file is written.

    On any failure the temporary directory, and any parent made for it, is removed. A failure that is
    not a ValueError is reported as one, so the command line prints one line.
    """
    path = Path(path)
    _refuse_existing(path)
    made = _missing_parents(path.parent)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.parent / f".{path.name[:64]}.{secrets.token_hex(8)}.partial"
        staging.mkdir()
    except OSError as exc:
        _remove_empty(made)
        raise _not_written(path, exc) from exc
    try:
        yield staging
        _refuse_existing(path)
        staging.rename(path)
    except BaseException as exc:
        shutil.rmtree(staging, ignore_errors=True)
        _remove_empty(made)
        if isinstance(exc, Exception) and not isinstance(exc, ValueError):
            raise _not_written(path, exc) from exc
        raise


def _refuse_existing(path):
    if os.path.lexists(path):
        raise ValueError(f"output directory already exists; got {str(path)!r}.")


def _missing_parents(directory):
    missing = []
    while not os.path.lexists(directory) and directory != directory.parent:
        missing.append(directory)
        directory = directory.parent
    return missing


def _remove_empty(directories):
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            return


def _not_written(path, exc):
    detail = " ".join(str(exc).split()).rstrip(".")
    cause = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
    return ValueError(f"output directory {str(path)!r} was not written; got {cause}.")


def _cutoff_record(point):
    return {
        "error_level": point["error_level"],
        "review_all": point["review_all"],
        "cutoff": point["cutoff"],
    }


def _point(record, level):
    for point in record["operating_points"]:
        if float(point["error_level"]) == float(level):
            return point
    raise ValueError(f"decide.error_levels is missing {level}; got {record['operating_points']}.")


def _mean_std(values):
    if not values:
        return None, None
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if len(array) < 2:
        return mean, None
    return mean, float(array.std(ddof=1))
