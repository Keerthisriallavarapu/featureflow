# Benchmarks

How to run the included load tests and what to look for. **Don't trust numbers without running them on your own hardware.**

## Setup

```bash
docker compose up -d
pip install -e ".[dev]"
pip install locust
```

Pre-load the online store with synthetic features:

```bash
python benchmarks/load_data.py --entities 100000
```

## Online store read latency

```bash
locust -f benchmarks/locust_read.py --host http://localhost:8080 \
  --users 100 --spawn-rate 50 --run-time 2m
```

Watch for:

- **p99 latency**: should be flat under 50ms up to ~5K RPS on a single-node Redis. Spikes mean GC or Redis blocking commands.
- **Error rate**: should be 0. Any error usually means connection pool exhaustion — bump `FEATUREFLOW_REDIS_MAX_CONNECTIONS`.

## End-to-end prediction latency

```bash
# Make sure a model is loaded and deployed
python examples/train_churn_model.py

locust -f benchmarks/locust_predict.py --host http://localhost:8080 \
  --users 50 --spawn-rate 25 --run-time 2m
```

The prediction path adds ~5-15ms on top of feature reads (model inference + serialization). XGBoost predictions are essentially free; sklearn ensembles are similar; PyTorch with model loading is slower.

## What to do when you hit a wall

| Symptom | Likely cause | Fix |
|---|---|---|
| p99 spikes correlate with feature group size | Reading too many features in one call | Split into multiple groups |
| Throughput plateaus at low RPS | Single Redis instance | Move to Redis Cluster; verify hash tags route correctly |
| Latency degrades over time | Hot keys, slow Redis commands | Run `redis-cli --latency-history`, check `SLOWLOG` |
| Memory grows unbounded | TTLs not set on feature groups | Set non-zero `ttl_seconds` on the group spec |

## Reproducing the numbers in the README

The README quotes target numbers, not measured ones. You'll get something close on a modern laptop with Docker Desktop. To get production-quality numbers:

1. Run Redis on a dedicated machine with `appendonly no` (no AOF) and `save ""` (no RDB) — you don't need persistence for the online store.
2. Pin the FeatureFlow process to CPU cores away from Redis.
3. Pre-warm the connection pool before measuring.
4. Use `--users` that's 2-3x your target concurrency to find the knee in the latency curve.

If your numbers are significantly worse than the targets, the README's target is wrong, not your hardware. Open an issue.
