"""Tests for the online store using fakeredis."""
from __future__ import annotations

import fakeredis.aioredis
import pytest

from featureflow.store.online import OnlineStore
from featureflow.types import Feature, FeatureGroup, FeatureType


@pytest.fixture
def group():
    return FeatureGroup(
        name="user_activity",
        entity="user_id",
        features=[
            Feature(name="login_count_7d", dtype=FeatureType.INT64, default_value=0),
            Feature(name="avg_session_seconds", dtype=FeatureType.FLOAT64, default_value=0.0),
            Feature(name="is_premium", dtype=FeatureType.BOOL, default_value=False),
            Feature(name="last_country", dtype=FeatureType.STRING, default_value="unknown"),
        ],
    )


@pytest.fixture
async def store():
    redis = fakeredis.aioredis.FakeRedis(decode_responses=False)
    s = OnlineStore(redis=redis)
    yield s
    await s.close()


async def test_write_then_read_roundtrip(store, group):
    await store.write(
        group,
        "user_42",
        {
            "login_count_7d": 5,
            "avg_session_seconds": 312.5,
            "is_premium": True,
            "last_country": "US",
        },
        event_ts=1700000000.0,
    )
    result = await store.read(group, "user_42")
    assert result["login_count_7d"] == 5
    assert result["avg_session_seconds"] == 312.5
    assert result["is_premium"] is True
    assert result["last_country"] == "US"
    assert result["_event_ts"] == 1700000000.0


async def test_missing_entity_returns_defaults(store, group):
    result = await store.read(group, "nonexistent")
    assert result["login_count_7d"] == 0
    assert result["avg_session_seconds"] == 0.0
    assert result["is_premium"] is False
    assert result["last_country"] == "unknown"


async def test_partial_read(store, group):
    await store.write(
        group,
        "user_1",
        {"login_count_7d": 3, "is_premium": True},
        event_ts=1700000000.0,
    )
    result = await store.read(group, "user_1", ["login_count_7d"])
    assert result["login_count_7d"] == 3
    assert "is_premium" not in result


async def test_batch_write_and_read_many(store, group):
    rows = [
        (f"user_{i}", {"login_count_7d": i, "is_premium": i % 2 == 0}, 1700000000.0 + i)
        for i in range(10)
    ]
    n = await store.write_batch(group, rows)
    assert n == 10

    results = await store.read_many(group, [f"user_{i}" for i in range(10)])
    assert len(results) == 10
    assert results["user_5"]["login_count_7d"] == 5
    assert results["user_4"]["is_premium"] is True
    assert results["user_5"]["is_premium"] is False
