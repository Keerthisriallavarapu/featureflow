"""Pre-populate the online store with synthetic features for load testing."""
from __future__ import annotations

import argparse
import asyncio
import time

import numpy as np

from featureflow import OnlineStore, Registry
from featureflow.types import Feature, FeatureGroup, FeatureType


async def main(n_entities: int) -> None:
    group = FeatureGroup(
        name="user_activity_30d",
        entity="user_id",
        ttl_seconds=86400 * 7,
        features=[
            Feature(name="login_count", dtype=FeatureType.INT64, default_value=0),
            Feature(name="avg_session_seconds", dtype=FeatureType.FLOAT64, default_value=0.0),
            Feature(name="days_since_last_login", dtype=FeatureType.INT64, default_value=999),
            Feature(name="num_features_used", dtype=FeatureType.INT64, default_value=0),
        ],
    )
    Registry().register_feature_group(group)

    rng = np.random.default_rng(42)
    online = OnlineStore()

    batch_size = 1000
    written = 0
    start = time.time()

    for batch_start in range(0, n_entities, batch_size):
        rows = []
        for i in range(batch_start, min(batch_start + batch_size, n_entities)):
            rows.append((
                f"u_{i}",
                {
                    "login_count": int(rng.poisson(3)),
                    "avg_session_seconds": float(rng.exponential(120)),
                    "days_since_last_login": int(rng.integers(0, 90)),
                    "num_features_used": int(rng.poisson(2)),
                },
                time.time() - rng.integers(0, 86400),
            ))
        await online.write_batch(group, rows)
        written += len(rows)
        if written % 10_000 == 0:
            elapsed = time.time() - start
            print(f"  wrote {written}/{n_entities} in {elapsed:.1f}s ({written/elapsed:.0f} rows/s)")

    await online.close()
    print(f"Done. Wrote {written} entities in {time.time() - start:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=int, default=100_000)
    args = parser.parse_args()
    asyncio.run(main(args.entities))
