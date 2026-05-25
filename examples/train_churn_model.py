"""End-to-end example: register a feature group, ingest features, train a
model, register and deploy it, then serve a prediction.

Run with the docker-compose stack up:
    docker compose up -d
    python examples/train_churn_model.py
"""
from __future__ import annotations

import asyncio
import logging
import pickle
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from featureflow import (
    Deployment,
    DeploymentMode,
    Feature,
    FeatureGroup,
    FeatureType,
    ModelVersion,
    OfflineStore,
    OnlineStore,
    PredictionRouter,
    Registry,
)


def make_group() -> FeatureGroup:
    return FeatureGroup(
        name="user_activity_30d",
        entity="user_id",
        ttl_seconds=86400 * 2,
        features=[
            Feature(name="login_count", dtype=FeatureType.INT64, default_value=0),
            Feature(name="avg_session_seconds", dtype=FeatureType.FLOAT64, default_value=0.0),
            Feature(name="days_since_last_login", dtype=FeatureType.INT64, default_value=999),
            Feature(name="num_features_used", dtype=FeatureType.INT64, default_value=0),
        ],
        description="30-day rolling user activity features for churn modeling.",
    )


def synth_training_data(n: int = 5000) -> pd.DataFrame:
    """Synthetic data — churn is a function of activity. Not realistic but
    enough to demonstrate the end-to-end pipeline."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "user_id": [f"u_{i}" for i in range(n)],
        "login_count": rng.poisson(3, n),
        "avg_session_seconds": rng.exponential(120, n),
        "days_since_last_login": rng.integers(0, 90, n),
        "num_features_used": rng.poisson(2, n),
    })
    # Churn signal: low activity, long since last login, few features
    churn_score = (
        -0.5 * df["login_count"]
        + 0.04 * df["days_since_last_login"]
        - 0.3 * df["num_features_used"]
        + rng.normal(0, 0.5, n)
    )
    df["churned"] = (churn_score > 0).astype(int)
    return df


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    print("=" * 60)
    print("FeatureFlow end-to-end example: churn modeling")
    print("=" * 60)

    # 1) Register the feature group
    registry = Registry(":memory:")  # in-memory for the demo
    group = make_group()
    registry.register_feature_group(group)
    print(f"\n[1/6] Registered feature group: {group.name}")

    # 2) Generate training data and write offline (the "feature pipeline")
    df = synth_training_data(5000)
    feature_df = df.drop(columns=["churned"]).copy()
    feature_df["_event_ts"] = pd.Timestamp.utcnow() - pd.Timedelta(days=1)

    offline = OfflineStore(tempfile.mkdtemp())
    offline.write(group, feature_df)
    print(f"[2/6] Wrote {len(df)} rows to offline store")

    # 3) Train a model using those features
    from sklearn.ensemble import GradientBoostingClassifier

    X = df[group.feature_names()].values
    y = df["churned"].values
    model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X, y)
    train_score = model.score(X, y)
    print(f"[3/6] Trained model. Train accuracy: {train_score:.3f}")

    # 4) Persist the model artifact and register it
    artifact = Path(tempfile.gettempdir()) / "churn_v1.pkl"
    with artifact.open("wb") as f:
        pickle.dump(model, f)

    mv = ModelVersion(
        name="churn",
        version=1,
        feature_groups=[group.name],
        artifact_uri=str(artifact),
        metrics={"train_accuracy": float(train_score)},
    )
    registry.register_model(mv)
    print(f"[4/6] Registered model: {mv.name} v{mv.version}")

    # 5) Write some features to the online store and deploy
    try:
        online = OnlineStore()
        # The online store needs Redis; if unavailable we'll get a connection
        # error here. For demo purposes that's fine — the offline part still ran.
        await online.write(
            group,
            "u_42",
            {
                "login_count": 1,
                "avg_session_seconds": 30.0,
                "days_since_last_login": 25,
                "num_features_used": 1,
            },
        )
        print("[5/6] Wrote one entity's features to online store (u_42)")

        registry.deploy(Deployment(
            model_name="churn",
            model_version=1,
            mode=DeploymentMode.PRODUCTION,
        ))

        # 6) Serve a prediction
        router = PredictionRouter(registry, online)
        router.load_from_disk(mv, feature_order=group.feature_names())

        # NOTE: The current router fetches features by group from the registry;
        # this is the simplified path. For the example we synthesize the call.
        features = await online.read(group, "u_42")
        from featureflow.serving import _SklearnPredictor

        predictor = _SklearnPredictor(model, group.feature_names())
        prediction = predictor.predict(features)
        print(f"[6/6] Prediction for u_42: churned={prediction}")

        await online.close()
    except Exception as e:
        print(f"[5-6/6] Skipped online steps (Redis not available?): {e}")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
