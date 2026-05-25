"""Core types. Feature groups, features, models, deployments.

A FeatureGroup is a logical unit of features that share an entity key and
update cadence (e.g. user_activity_5m). Features within a group are read
together from the online store in one Redis pipeline call.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,62}$")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class FeatureType(str, Enum):
    INT64 = "int64"
    FLOAT64 = "float64"
    STRING = "string"
    BOOL = "bool"
    # Vector / list types live as separate keys in Redis (json-encoded)
    FLOAT_LIST = "float_list"


class Feature(BaseModel):
    name: str
    dtype: FeatureType
    description: str = ""
    default_value: Any = None

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        if not NAME_PATTERN.match(v):
            raise ValueError(f"Invalid feature name: {v!r}")
        return v


class FeatureGroup(BaseModel):
    """A logical group of features for an entity, materialized together.

    Versioning: feature group changes that affect schema bump the version.
    Old versions stay readable so models trained against them keep working
    during rollouts. See docs/DECISIONS.md D-002.
    """

    name: str
    version: int = 1
    entity: str  # the join key, e.g. "user_id"
    features: list[Feature]
    ttl_seconds: int = 86400  # online store TTL; offline is permanent
    source: str = ""  # free-form source descriptor (kafka topic, table name)
    owner: str = ""
    created_at: datetime = Field(default_factory=_utcnow)
    description: str = ""

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        if not NAME_PATTERN.match(v):
            raise ValueError(f"Invalid feature group name: {v!r}")
        return v

    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    def online_key(self, entity_id: str) -> str:
        """Redis key. Hash-tagged so multi-feature lookups in the same group
        land on the same Redis slot (matters for Redis Cluster)."""
        return f"ff:{{{self.name}:{entity_id}}}:v{self.version}"


class ModelVersion(BaseModel):
    """A trained model artifact registered for serving."""

    name: str
    version: int
    feature_groups: list[str]  # feature group names this model requires
    artifact_uri: str  # where the model file lives
    created_at: datetime = Field(default_factory=_utcnow)
    metrics: dict[str, float] = Field(default_factory=dict)
    description: str = ""


class DeploymentMode(str, Enum):
    PRODUCTION = "production"  # serves traffic
    CANARY = "canary"  # serves a percentage
    SHADOW = "shadow"  # sees all traffic, doesn't affect responses


class Deployment(BaseModel):
    model_name: str
    model_version: int
    mode: DeploymentMode
    traffic_pct: int = 100  # only meaningful for CANARY
    created_at: datetime = Field(default_factory=_utcnow)


class PredictionRequest(BaseModel):
    entity_id: str
    features_override: dict[str, Any] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    prediction: Any
    model_name: str
    model_version: int
    feature_age_seconds: dict[str, float] = Field(default_factory=dict)
    latency_ms: float = 0.0
    shadow_predictions: dict[str, Any] = Field(default_factory=dict)
