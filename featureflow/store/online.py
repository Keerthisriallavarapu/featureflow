"""Online feature store. Redis is the backend.

Key design decision: hash-tagged keys ({group:entity}) so all features for an
entity in one group land in the same Redis slot. This means a single MGET
pipeline call retrieves them all — even in Redis Cluster. Without hash tags,
cluster-mode would scatter the keys and force multiple round-trips.

See docs/DECISIONS.md D-003.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from redis.asyncio import Redis
from redis.asyncio.client import Pipeline

from ..settings import get_settings
from ..types import FeatureGroup, FeatureType

log = logging.getLogger(__name__)


class OnlineStore:
    def __init__(self, redis: Redis | None = None):
        if redis is None:
            s = get_settings()
            redis = Redis.from_url(
                s.redis_url,
                max_connections=s.redis_max_connections,
                decode_responses=False,  # we handle bytes ourselves for perf
            )
        self._r = redis

    async def close(self) -> None:
        await self._r.aclose()

    async def write(
        self,
        group: FeatureGroup,
        entity_id: str,
        values: dict[str, Any],
        event_ts: float | None = None,
    ) -> None:
        """Write feature values for one entity. event_ts (if provided) is the
        time the underlying event happened — used later for feature age."""
        key = group.online_key(entity_id)
        payload = self._encode(group, values, event_ts or time.time())
        await self._r.hset(key, mapping=payload)
        if group.ttl_seconds > 0:
            await self._r.expire(key, group.ttl_seconds)

    async def write_batch(
        self,
        group: FeatureGroup,
        rows: list[tuple[str, dict[str, Any], float]],
    ) -> int:
        """Bulk write. (entity_id, values, event_ts) tuples. Returns count."""
        if not rows:
            return 0
        pipe = self._r.pipeline(transaction=False)
        for entity_id, values, ts in rows:
            key = group.online_key(entity_id)
            payload = self._encode(group, values, ts)
            pipe.hset(key, mapping=payload)
            if group.ttl_seconds > 0:
                pipe.expire(key, group.ttl_seconds)
        await pipe.execute()
        return len(rows)

    async def read(
        self,
        group: FeatureGroup,
        entity_id: str,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Read features for one entity. If feature_names is None, reads all.

        Returns dict of feature_name -> value. Missing features get the
        feature's default_value from the spec.
        """
        names = feature_names or group.feature_names()
        # Always fetch _event_ts so callers can compute feature age
        key = group.online_key(entity_id)
        fields = [n.encode() for n in names] + [b"_event_ts"]
        raw = await self._r.hmget(key, fields)
        return self._decode(group, names, raw)

    async def read_many(
        self,
        group: FeatureGroup,
        entity_ids: list[str],
        feature_names: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Fan-out read. Uses a single pipeline so all reads land together."""
        names = feature_names or group.feature_names()
        fields = [n.encode() for n in names] + [b"_event_ts"]
        pipe = self._r.pipeline(transaction=False)
        for eid in entity_ids:
            pipe.hmget(group.online_key(eid), fields)
        results = await pipe.execute()
        return {
            eid: self._decode(group, names, raw)
            for eid, raw in zip(entity_ids, results, strict=True)
        }

    # ---- Encoding ---------------------------------------------------------

    def _encode(
        self,
        group: FeatureGroup,
        values: dict[str, Any],
        event_ts: float,
    ) -> dict[bytes, bytes]:
        out: dict[bytes, bytes] = {b"_event_ts": str(event_ts).encode()}
        for f in group.features:
            if f.name not in values:
                continue
            v = values[f.name]
            out[f.name.encode()] = _encode_value(v, f.dtype)
        return out

    def _decode(
        self,
        group: FeatureGroup,
        feature_names: list[str],
        raw: list[bytes | None],
    ) -> dict[str, Any]:
        spec_by_name = {f.name: f for f in group.features}
        out: dict[str, Any] = {}
        # raw is feature_names + [_event_ts]
        for i, name in enumerate(feature_names):
            val_bytes = raw[i] if i < len(raw) else None
            spec = spec_by_name.get(name)
            if val_bytes is None:
                out[name] = spec.default_value if spec else None
            else:
                out[name] = _decode_value(val_bytes, spec.dtype if spec else FeatureType.STRING)
        # event_ts
        ts_bytes = raw[-1] if raw else None
        out["_event_ts"] = float(ts_bytes) if ts_bytes else None
        return out


def _encode_value(v: Any, dtype: FeatureType) -> bytes:
    if dtype == FeatureType.FLOAT_LIST:
        return json.dumps(list(v)).encode()
    if dtype == FeatureType.BOOL:
        return b"1" if v else b"0"
    return str(v).encode()


def _decode_value(b: bytes, dtype: FeatureType) -> Any:
    if dtype == FeatureType.INT64:
        return int(b)
    if dtype == FeatureType.FLOAT64:
        return float(b)
    if dtype == FeatureType.BOOL:
        return b == b"1"
    if dtype == FeatureType.FLOAT_LIST:
        return json.loads(b.decode())
    return b.decode()
