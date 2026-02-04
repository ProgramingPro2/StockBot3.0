"""
Scorer model for universe autodiscovery: maps feature vector to a single score (higher = prefer to add/keep).
Trainable (GradientBoostingClassifier) or rule-based fallback.
"""
import json
from pathlib import Path

import numpy as np

from ..config import ENV_OBS_ROLLING_FEATURES, UNIVERSE_MODEL_DIR
from ..utils.logger import get_logger
from .features import feature_dict_to_vector

logger = get_logger()

SCORER_MODEL_PATH = UNIVERSE_MODEL_DIR / "scorer_model.pkl"
SCORER_META_PATH = UNIVERSE_MODEL_DIR / "scorer_meta.json"


def _rule_score(features: dict) -> float:
    """
    Simple rule when no trained model: score = w1*sentiment + w2*(quant-0.5) + w3*fund - w4*regime.
    Normalized to [0, 1] via (score + 1) / 2 clipped.
    """
    s = float(features.get("sentiment", 0.0))
    q = float(features.get("quant_prob", 0.5))
    f = float(features.get("fund_score", 0.0))
    r = float(features.get("regime_flag", 0.5))
    raw = 0.25 * s + 0.25 * (q - 0.5) * 2 + 0.25 * f - 0.25 * r
    return float(np.clip((raw + 1) / 2, 0, 1))


class ScorerModel:
    """
    Score symbols for autodiscovery. Loads trained model from UNIVERSE_MODEL_DIR or uses rule fallback.
    """

    def __init__(self, load: bool = True):
        self._model = None
        self._n_features: int | None = None
        if load:
            self._load()

    def _load(self) -> None:
        if not SCORER_MODEL_PATH.exists():
            return
        try:
            import joblib

            self._model = joblib.load(SCORER_MODEL_PATH)
            if SCORER_META_PATH.exists():
                meta = json.loads(SCORER_META_PATH.read_text())
                self._n_features = meta.get("n_features")
            logger.info("Loaded universe scorer from %s", SCORER_MODEL_PATH)
        except Exception as e:
            logger.warning("Could not load universe scorer: %s; using rule", e)
            self._model = None

    def score_features(self, features: dict) -> float:
        """Score a single symbol's feature dict. Returns value in [0, 1]."""
        if self._model is None:
            return _rule_score(features)
        vec = feature_dict_to_vector(features)
        vec = vec.reshape(1, -1)
        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(vec)[0]
            if proba.shape[0] == 2:
                return float(proba[1])
            return float(proba[0])
        pred = self._model.predict(vec)[0]
        return float(pred)

    def score_vectors(self, X: np.ndarray) -> np.ndarray:
        """Score a matrix of feature vectors. Returns 1d array of scores."""
        if self._model is None:
            raise RuntimeError("No trained model; use score_features with dict for rule fallback")
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)[:, 1]
        return self._model.predict(X).astype(np.float64)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
    ) -> None:
        """
        Train GradientBoostingClassifier. y should be binary (1 = good / top quartile).
        Saves to UNIVERSE_MODEL_DIR.
        """
        from sklearn.ensemble import GradientBoostingClassifier

        self._model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
        )
        self._model.fit(X, y)
        self._n_features = X.shape[1]
        UNIVERSE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        import joblib

        joblib.dump(self._model, SCORER_MODEL_PATH)
        SCORER_META_PATH.write_text(json.dumps({"n_features": self._n_features}))
        logger.info("Saved universe scorer to %s (n_features=%d)", SCORER_MODEL_PATH, self._n_features)
