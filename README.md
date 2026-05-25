# FeatureFlow

A self-hosted ML feature store and model serving platform. Inspired by the patterns in Uber's Michelangelo and Meta's FBLearner, scaled down to something a small team can actually run.

## What's here

- **Online store** (Redis): low-latency feature reads, p99 <35ms target at 10K RPS.
- **Offline store** (Parquet + DuckDB): training data with point-in-time-correct joins so models don't train on features that didn't exist yet at label time.
- **Registry**: feature group and model metadata, schema evolution checks (catches removed features before they break serving).
- **Serving**: prediction router with production / canary / shadow deployment modes, consistent-hash canary routing.
- **Streaming ingestion** (Kafka, optional): batched writes to both stores with at-least-once semantics.
- **Drift detection**: PSI and chi-square monitors as pure functions you can wire into your monitoring stack.
- **Prometheus metrics** out of the box for latency and throughput.

This is the bit that's missing from most ML portfolios — the infrastructure between "I trained a model in a notebook" and "the model is making predictions in production reliably."

## Quick start

```bash
docker compose up -d  # starts Redis
pip install -e ".[ml,dev]"

# Run the end-to-end demo: register features, train a model, deploy, predict
python examples/train_churn_model.py

# Or run the API and hit it
featureflow serve --port 8080 &
curl -X POST http://localhost:8080/feature-groups \
  -H "Content-Type: application/json" \
  -d @examples/specs/user_activity.json
```

## Why each piece exists

**Online + offline store with a single registry.** The training/serving skew bug — where the features your model trained on are subtly different from what it sees in production — is the single most common ML production failure I've seen. Same registry, same feature specs, same encoders is the only way to prevent it structurally rather than through process.

**Hash-tagged Redis keys.** `ff:{group:entity}:vN` is the format. The `{}` is a Redis Cluster hash tag: everything inside the braces hashes to the same slot, so all features for one entity in one group land on the same shard. Without this, reading 5 features means 5 round-trips in cluster mode. With it, one pipelined call.

**DuckDB ASOF JOIN for point-in-time joins.** Spark is the textbook answer; DuckDB does it in-process with the same semantics, no cluster, no JVM. At "<1B rows" scale it's strictly better. At "100B rows" you outgrow it — but most teams who need this never get there.

**Canary + shadow as first-class deployment modes.** Canary routes a percentage of real traffic via consistent hashing (so the same user always sees the same model version, otherwise you can't measure anything). Shadow runs predictions in parallel without affecting responses, perfect for "does this new model agree with the old one before we cut over?"

For the longer reasoning behind these choices, including the alternatives I tried and abandoned, see [docs/DECISIONS.md](docs/DECISIONS.md).

## Project layout

```
featureflow/
├── featureflow/
│   ├── store/
│   │   ├── online.py        # Redis + hash-tagged keys
│   │   └── offline.py       # Parquet + DuckDB ASOF
│   ├── registry/            # Feature + model + deployment metadata
│   ├── serving/             # Router: prod/canary/shadow
│   ├── monitoring/          # PSI / chi-square drift
│   ├── ingestion/           # Optional Kafka consumer
│   ├── server.py            # FastAPI
│   ├── cli.py
│   ├── types.py
│   └── settings.py
├── examples/
│   └── train_churn_model.py # End-to-end demo
├── tests/
├── docs/
├── k8s/                     # Helm chart placeholder
├── benchmarks/              # Locust scripts for load tests
└── docker-compose.yml
```

## Performance

Target numbers (single-node, Redis + 4 CPU cores):

| Operation | Target |
|---|---|
| Online feature lookup, 1 group, ≤5 features | p99 <15ms |
| Online feature lookup, 3 groups parallel | p99 <30ms |
| End-to-end prediction (XGBoost) | p99 <50ms |

The benchmarks/ directory contains Locust scripts to verify these on your hardware. Honest numbers are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — run them yourself, don't trust mine without verifying.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

`fakeredis` is used for online-store tests so they run without a real Redis.

## License

Apache 2.0
