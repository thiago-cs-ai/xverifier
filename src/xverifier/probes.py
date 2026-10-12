"""A shared PCA and one linear probe per verdict."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xverifier.constants import PCA_FLOOR, PROBE_MAX_ITER, PROBE_RANDOM_STATE
from xverifier.guards import assert_verdict, pin_threads

_SEED_MAX = 2**32 - 1


class RoutedLinearProbe:
    """One linear probe per verdict behind a PCA fit on the training rows."""

    def __init__(self, C=1.0, max_components=256, min_rows=40, seed=0):
        self.C = float(C)
        self.max_components = int(max_components)
        self.min_rows = int(min_rows)
        self.seed = int(seed)
        if not 0 <= self.seed <= _SEED_MAX:
            raise ValueError(f"seed must be a valid random seed; got {seed}.")
        self.pca = None
        self.pooled = None
        self.cells = {0: None, 1: None}

    def fit(self, representation, verdict, agreement):
        """Fit the shared PCA and the per-verdict probes; return self."""
        pin_threads()
        representation = np.asarray(representation)
        verdict = np.asarray(verdict)
        agreement = np.asarray(agreement)
        width = _pca_width(len(representation), representation.shape[1], self.max_components)
        if width > representation.shape[1] or width > len(representation):
            raise ValueError(
                f"representation cannot support {width} components; got shape {representation.shape}.")
        self.pca = PCA(n_components=width, random_state=self.seed).fit(representation)
        transformed = self.pca.transform(representation)
        self.pooled = _fit_cell(transformed, agreement, self.C)
        if self.pooled is None:
            classes = sorted(set(agreement.tolist()))
            raise ValueError(f"agreement on the training rows has a single class; got {classes}.")
        for regime in (0, 1):
            mask = verdict == regime
            if int(mask.sum()) >= self.min_rows and _both_classes(agreement[mask]):
                self.cells[regime] = _fit_cell(transformed[mask], agreement[mask], self.C)
            else:
                self.cells[regime] = None
        return self

    @property
    def n_components(self):
        """Return the fitted PCA width."""
        return int(self.pca.n_components_)

    def scores(self, representation, verdict):
        """Return the raw probability of agreement for each row."""
        verdict = assert_verdict(verdict)
        representation = np.asarray(representation)
        expected = int(self.pca.n_features_in_)
        if representation.ndim == 2 and representation.shape[1] != expected:
            raise ValueError(
                f"representation must have {expected} columns like the training rows; "
                f"got {representation.shape[1]}.")
        transformed = self.pca.transform(representation)
        scores = np.empty(len(verdict), dtype=np.float64)
        for regime in (0, 1):
            mask = verdict == regime
            if not np.any(mask):
                continue
            model = self.cells[regime] if self.cells[regime] is not None else self.pooled
            scores[mask] = model.predict_proba(transformed[mask])[:, 1]
        return scores

    def pooled_mask(self, verdict):
        """Return which rows are scored by the pooled probe."""
        verdict = np.asarray(verdict)
        mask = np.zeros(len(verdict), dtype=bool)
        for regime in (0, 1):
            if self.cells[regime] is None:
                mask |= verdict == regime
        return mask

    def arrays(self):
        """Return the fitted PCA and probe arrays."""
        payload = {
            "pca_mean": np.array(self.pca.mean_),
            "pca_components": np.array(self.pca.components_),
            "pca_explained_variance": np.array(self.pca.explained_variance_),
            "probe_C": np.array(self.C, dtype=np.float64),
            "probe_seed": np.array(self.seed, dtype=np.int64),
        }
        payload.update(_pipeline_arrays("pooled", self.pooled))
        for regime in (0, 1):
            active = self.cells[regime] is not None
            payload[f"cell{regime}_active"] = np.array(active)
            if active:
                payload.update(_pipeline_arrays(f"cell{regime}", self.cells[regime]))
        return payload

    @classmethod
    def from_arrays(cls, stored):
        """Rebuild a probe from arrays saved without pickle."""
        probe = cls(C=float(stored["probe_C"]), seed=int(stored["probe_seed"]))
        components = np.array(stored["pca_components"])
        mean = np.array(stored["pca_mean"])
        explained_variance = np.array(stored["pca_explained_variance"])
        probe.pca = _restore_pca(mean, components, explained_variance, int(stored["probe_seed"]))
        probe.pooled = _restore_pipeline("pooled", stored, probe.C)
        probe.cells = {}
        for regime in (0, 1):
            if bool(stored[f"cell{regime}_active"]):
                probe.cells[regime] = _restore_pipeline(f"cell{regime}", stored, probe.C)
            else:
                probe.cells[regime] = None
        return probe


def _pca_width(n_train, columns, max_components):
    return max(PCA_FLOOR, min(n_train // 10, int(max_components), int(columns)))


def _both_classes(agreement):
    return np.unique(agreement).size > 1


def _fit_cell(transformed, agreement, C):
    if not _both_classes(agreement):
        return None
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=float(C), max_iter=PROBE_MAX_ITER,
                                   random_state=PROBE_RANDOM_STATE)),
    ])
    return pipeline.fit(transformed, agreement)


def _pipeline_arrays(prefix, pipeline):
    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["clf"]
    return {
        f"{prefix}_mean": np.array(scaler.mean_),
        f"{prefix}_scale": np.array(scaler.scale_),
        f"{prefix}_coef": np.array(classifier.coef_),
        f"{prefix}_intercept": np.array(classifier.intercept_),
    }


def _restore_pca(mean, components, explained_variance, seed):
    pca = PCA(n_components=int(components.shape[0]), random_state=int(seed))
    pca.mean_ = mean
    pca.components_ = components
    pca.explained_variance_ = explained_variance
    pca.n_components_ = int(components.shape[0])
    pca.n_features_in_ = int(mean.shape[0])
    pca.n_samples_seen_ = np.int64(1)
    return pca


def _restore_pipeline(prefix, stored, C):
    mean = np.array(stored[f"{prefix}_mean"])
    scale = np.array(stored[f"{prefix}_scale"])
    coef = np.array(stored[f"{prefix}_coef"])
    intercept = np.array(stored[f"{prefix}_intercept"])
    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = scale
    scaler.var_ = scale ** 2
    scaler.n_features_in_ = int(mean.shape[0])
    scaler.n_samples_seen_ = np.int64(1)
    classifier = LogisticRegression(C=float(C), max_iter=PROBE_MAX_ITER,
                                   random_state=PROBE_RANDOM_STATE)
    classifier.coef_ = coef.reshape(1, -1)
    classifier.intercept_ = intercept.reshape(-1)
    classifier.classes_ = np.array([0, 1])
    classifier.n_features_in_ = int(coef.shape[-1])
    return Pipeline([("scaler", scaler), ("clf", classifier)])
