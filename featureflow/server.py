"""FastAPI HTTP API for feature reads, model serving, and admin ops."""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from .registry import Registry
from .serving import PredictionRouter
from .settings import get_settings
from .store.online import OnlineStore
from .types import (
    Deployment,
    DeploymentMode,
    FeatureGroup,
    ModelVersion,
)

log = logging.getLogger(__name__)


class ReadRequest(BaseModel):
    group: str
    entity_id: str
    features: list[str] | None = None


class WriteRequest(BaseModel):
    group: str
    entity_id: str
    values: dict
    event_ts: float | None = None


class DeployRequest(BaseModel):
    model_name: str
    model_version: int
    mode: DeploymentMode
    traffic_pct: int = 100


def create_app(
    registry: Registry | None = None,
    online: OnlineStore | None = None,
    router: PredictionRouter | None = None,
) -> FastAPI:
    app = FastAPI(title="FeatureFlow", version="0.1.0")

    _registry = registry or Registry()
    _online = online or OnlineStore()
    _router = router or PredictionRouter(_registry, _online)

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # ---- Feature groups ----
    @app.post("/feature-groups")
    async def register_group(group: FeatureGroup):
        _registry.register_feature_group(group)
        return {"ok": True, "name": group.name, "version": group.version}

    @app.get("/feature-groups")
    async def list_groups():
        return [g.model_dump() for g in _registry.list_feature_groups()]

    @app.get("/feature-groups/{name}")
    async def get_group(name: str):
        g = _registry.get_feature_group(name)
        if not g:
            raise HTTPException(404, f"Feature group {name} not found")
        return g.model_dump()

    # ---- Online store ----
    @app.post("/features/read")
    async def read_features(req: ReadRequest):
        group = _registry.get_feature_group(req.group)
        if not group:
            raise HTTPException(404, f"Group {req.group} not found")
        return await _online.read(group, req.entity_id, req.features)

    @app.post("/features/write")
    async def write_features(req: WriteRequest):
        group = _registry.get_feature_group(req.group)
        if not group:
            raise HTTPException(404, f"Group {req.group} not found")
        await _online.write(group, req.entity_id, req.values, req.event_ts)
        return {"ok": True}

    # ---- Models ----
    @app.post("/models")
    async def register_model(model: ModelVersion):
        _registry.register_model(model)
        return {"ok": True}

    @app.post("/deploy")
    async def deploy(req: DeployRequest):
        deployment = Deployment(
            model_name=req.model_name,
            model_version=req.model_version,
            mode=req.mode,
            traffic_pct=req.traffic_pct,
        )
        _registry.deploy(deployment)
        return {"ok": True}

    # ---- Predict ----
    @app.post("/predict/{model_name}")
    async def predict(model_name: str, body: dict):
        entity_id = body.get("entity_id")
        if not entity_id:
            raise HTTPException(400, "entity_id required")
        try:
            resp = await _router.predict(model_name, entity_id)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return resp.model_dump()

    return app


# Convenience for `uvicorn featureflow.server:app`
app = create_app()
