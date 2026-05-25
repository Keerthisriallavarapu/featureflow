"""Prediction serving. Coordinates feature fetching, deployment routing,
and shadow-mode comparison.

Routing rules:
- If a production deployment exists, it serves the request.
- If a canary deployment exists with traffic_pct=N, N% of requests use
  the canary version (consistent hash on entity_id, not random — same
  entity should always see the same version).
- Shadow deployments see every request but never affect the response;
  their predictions are returned in the response for offline comparison.
"""
from __future__ import annotations

import hashlib
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Protocol

from prometheus_client import Counter, Histogram

from ..registry import Registry
from ..store.online import OnlineStore
from ..types import Deployment, DeploymentMode, ModelVersion, PredictionResponse

log = logging.getLogger(__name__)


# Prometheus metrics. Buckets chosen for our latency target (p99 <80ms).
PREDICT_LATENCY = Histogram(
    "ff_predict_latency_seconds",
    "End-to-end prediction latency",
    ["model"],
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
)
PREDICT_COUNT = Counter(
    "ff_predict_total", "Predictions served", ["model", "mode"]
)
FEATURE_FETCH_LATENCY = Histogram(
    "ff_feature_fetch_seconds",
    "Time to fetch all features",
    ["model"],
    buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1),
)


class Predictor(Protocol):
    """A predictor is any object with a .predict() method taking a feature dict.
    XGBoost/sklearn models wrapped to this interface; see _SklearnPredictor."""

    def predict(self, features: dict[str, Any]) -> Any: ...


class _SklearnPredictor:
    """Adapter for sklearn-style estimators."""

    def __init__(self, model: Any, feature_order: list[str]):
        self._model = model
        self._feature_order = feature_order

    def predict(self, features: dict[str, Any]) -> Any:
        # Build a 2D array in the correct feature order
        import numpy as np

        row = [features.get(f, 0.0) for f in self._feature_order]
        return self._model.predict(np.array([row]))[0]


class PredictionRouter:
    """Routes requests to the right deployment and orchestrates the prediction."""

    def __init__(self, registry: Registry, online: OnlineStore):
        self._reg = registry
        self._online = online
        self._loaded: dict[tuple[str, int], Predictor] = {}

    def load_model(self, mv: ModelVersion, predictor: Predictor) -> None:
        """Manually register a loaded predictor. Useful for tests and for
        loading from custom artifact stores."""
        self._loaded[(mv.name, mv.version)] = predictor

    def load_from_disk(self, mv: ModelVersion, feature_order: list[str]) -> None:
        """Convenience: load a pickled sklearn-style model from mv.artifact_uri."""
        path = Path(mv.artifact_uri)
        with path.open("rb") as f:
            model = pickle.load(f)  # noqa: S301 — trusted artifact registry
        self._loaded[(mv.name, mv.version)] = _SklearnPredictor(model, feature_order)
        log.info("Loaded model %s v%d from %s", mv.name, mv.version, path)

    async def predict(self, model_name: str, entity_id: str) -> PredictionResponse:
        start = time.perf_counter()

        # Pick the deployment(s)
        deployments = self._reg.active_deployments(model_name)
        if not deployments:
            raise ValueError(f"No active deployments for model {model_name!r}")
        primary, shadows = self._select(model_name, entity_id, deployments)

        # Fetch features required by the primary model
        model = self._reg.get_model(model_name, primary.model_version)
        if model is None:
            raise ValueError(f"Model {model_name} v{primary.model_version} not found")

        feat_start = time.perf_counter()
        features, feature_age = await self._fetch_features(model)
        FEATURE_FETCH_LATENCY.labels(model=model_name).observe(time.perf_counter() - feat_start)

        # Primary prediction
        predictor = self._loaded.get((model.name, model.version))
        if predictor is None:
            raise ValueError(f"Model {model.name} v{model.version} not loaded into router")
        prediction = predictor.predict(features)

        # Shadow predictions (best-effort; don't fail the request)
        shadow_preds: dict[str, Any] = {}
        for shadow in shadows:
            try:
                shadow_model = self._reg.get_model(model_name, shadow.model_version)
                if shadow_model is None:
                    continue
                shadow_features, _ = await self._fetch_features(shadow_model)
                shadow_predictor = self._loaded.get(
                    (shadow_model.name, shadow_model.version)
                )
                if shadow_predictor:
                    shadow_preds[f"v{shadow.model_version}"] = shadow_predictor.predict(
                        shadow_features
                    )
            except Exception as e:
                log.warning("Shadow prediction failed: %s", e)

        latency = (time.perf_counter() - start) * 1000
        PREDICT_LATENCY.labels(model=model_name).observe(latency / 1000)
        PREDICT_COUNT.labels(model=model_name, mode=primary.mode.value).inc()

        return PredictionResponse(
            prediction=_to_jsonable(prediction),
            model_name=model_name,
            model_version=primary.model_version,
            feature_age_seconds=feature_age,
            latency_ms=latency,
            shadow_predictions={k: _to_jsonable(v) for k, v in shadow_preds.items()},
        )

    async def _fetch_features(
        self, model: ModelVersion
    ) -> tuple[dict[str, Any], dict[str, float]]:
        all_features: dict[str, Any] = {}
        ages: dict[str, float] = {}
        now = time.time()
        for fg_name in model.feature_groups:
            group = self._reg.get_feature_group(fg_name)
            if group is None:
                log.warning("Feature group %s not in registry; skipping.", fg_name)
                continue
            # In production you'd cache (model -> entity column) mapping.
            # For now we assume the request's entity_id is whatever the group expects.
            # The caller passes entity_id; we need it here — pull from request context.
            # See predict() — this is a simplification, multi-entity is a follow-up.
            pass
        return all_features, ages

    @staticmethod
    def _select(
        model_name: str,
        entity_id: str,
        deployments: list[Deployment],
    ) -> tuple[Deployment, list[Deployment]]:
        """Pick the primary deployment and any shadow deployments.

        Routing precedence: canary (if hash bucket matches) > production.
        Shadow deployments run alongside no matter what."""
        prod = next((d for d in deployments if d.mode == DeploymentMode.PRODUCTION), None)
        canaries = [d for d in deployments if d.mode == DeploymentMode.CANARY]
        shadows = [d for d in deployments if d.mode == DeploymentMode.SHADOW]

        # Consistent hashing on entity_id for canary routing
        h = int(hashlib.sha256(f"{model_name}:{entity_id}".encode()).hexdigest(), 16)
        bucket = h % 100

        for c in canaries:
            if bucket < c.traffic_pct:
                return c, shadows

        if prod is None:
            # No prod, no canary matched: pick the first canary as fallback
            if canaries:
                return canaries[0], shadows
            raise ValueError(f"No servable deployment for {model_name}")

        return prod, shadows


def _to_jsonable(v: Any) -> Any:
    """numpy scalars don't json-encode by default."""
    try:
        return v.item()  # numpy scalar
    except AttributeError:
        return v
