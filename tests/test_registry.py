"""Tests for the registry."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from featureflow.registry import Registry
from featureflow.types import (
    Deployment,
    DeploymentMode,
    Feature,
    FeatureGroup,
    FeatureType,
    ModelVersion,
)


@pytest.fixture
def registry():
    with tempfile.TemporaryDirectory() as tmp:
        r = Registry(Path(tmp) / "test.duckdb")
        yield r
        r.close()


def _fg(name="user_activity", version=1, features=None) -> FeatureGroup:
    return FeatureGroup(
        name=name,
        version=version,
        entity="user_id",
        features=features or [
            Feature(name="login_count", dtype=FeatureType.INT64),
        ],
    )


def test_register_and_get(registry):
    g = _fg()
    registry.register_feature_group(g)
    fetched = registry.get_feature_group("user_activity")
    assert fetched is not None
    assert fetched.name == "user_activity"
    assert fetched.version == 1


def test_get_specific_version(registry):
    registry.register_feature_group(_fg(version=1))
    registry.register_feature_group(_fg(version=2))
    v1 = registry.get_feature_group("user_activity", version=1)
    v2 = registry.get_feature_group("user_activity", version=2)
    assert v1.version == 1
    assert v2.version == 2


def test_get_latest_version(registry):
    registry.register_feature_group(_fg(version=1))
    registry.register_feature_group(_fg(version=3))
    registry.register_feature_group(_fg(version=2))
    latest = registry.get_feature_group("user_activity")
    assert latest.version == 3


def test_schema_evolution_blocks_feature_removal(registry):
    """Removing a feature without bumping the version should fail."""
    registry.register_feature_group(_fg(features=[
        Feature(name="a", dtype=FeatureType.INT64),
        Feature(name="b", dtype=FeatureType.INT64),
    ]))
    # Re-register v1 with feature 'b' removed — should raise
    with pytest.raises(ValueError, match="Cannot remove features"):
        registry.register_feature_group(_fg(features=[
            Feature(name="a", dtype=FeatureType.INT64),
        ]))


def test_deployment_replaces_previous_production(registry):
    registry.register_model(ModelVersion(
        name="churn", version=1, feature_groups=["user_activity"], artifact_uri="/x"
    ))
    registry.register_model(ModelVersion(
        name="churn", version=2, feature_groups=["user_activity"], artifact_uri="/y"
    ))
    registry.deploy(Deployment(
        model_name="churn", model_version=1, mode=DeploymentMode.PRODUCTION
    ))
    registry.deploy(Deployment(
        model_name="churn", model_version=2, mode=DeploymentMode.PRODUCTION
    ))
    active = registry.active_deployments("churn")
    # Only v2 should be active
    assert len(active) == 1
    assert active[0].model_version == 2


def test_canary_coexists_with_production(registry):
    registry.register_model(ModelVersion(
        name="churn", version=1, feature_groups=["user_activity"], artifact_uri="/x"
    ))
    registry.register_model(ModelVersion(
        name="churn", version=2, feature_groups=["user_activity"], artifact_uri="/y"
    ))
    registry.deploy(Deployment(
        model_name="churn", model_version=1, mode=DeploymentMode.PRODUCTION
    ))
    registry.deploy(Deployment(
        model_name="churn", model_version=2, mode=DeploymentMode.CANARY, traffic_pct=10
    ))
    active = registry.active_deployments("churn")
    assert len(active) == 2
    modes = {d.mode for d in active}
    assert DeploymentMode.PRODUCTION in modes
    assert DeploymentMode.CANARY in modes
