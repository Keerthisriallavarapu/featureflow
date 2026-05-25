# Engineering Decisions

## D-001: One process exposing both feature reads and predictions

**Status:** Accepted

**Context.** Should feature reads and model predictions live in separate services? Most ML platform writeups suggest yes — separation of concerns, independent scaling.

**Decision.** Same FastAPI process for now. Split only when load actually demands it.

**Why.** The hot path for a prediction is "fetch features, run model". When those live in two services, you add a network hop, an extra serialization round-trip, and a failure mode where features are stale because the feature service is up but reads are slow.

For 99% of teams shipping their first feature store, putting them together is the right call. The interfaces (`OnlineStore`, `PredictionRouter`) are designed to be split later: the router talks to the store via a clean interface, so swapping the in-process call for an HTTP/gRPC call is a one-file change.

---

## D-002: Schema evolution: features can be added, never removed without a version bump

**Status:** Accepted

**Context.** When the team changes a feature group definition, what's safe?

**Decision.** Adding features is fine in-place. Removing features requires a new version of the group (`v2`, `v3`, ...). The registry enforces this.

**Why.** This caught a real bug in early testing. A teammate removed an unused feature from the group definition. Tests passed, deployment succeeded, but a model trained against the old schema was still in production and started getting empty feature values for the removed column — silently degrading predictions because the default was 0.

The registry now blocks the removal at registration time with a clear error. Old versions stay readable, so models trained against v1 keep working until they're retrained.

**Why not full semantic versioning?** Considered it. The simple "additive in-place, breaking changes need a bump" rule covers ~95% of cases without the overhead of explaining semver to people who just want to add a feature.

---

## D-003: Hash tags on Redis keys

**Status:** Accepted

**Context.** Reading 5 features for one user from one group should be one Redis call, not five.

**Decision.** Key format `ff:{group:entity}:vN`. The `{group:entity}` is a Redis hash tag — Redis Cluster hashes only the tagged portion, so all features for one entity in one group land on the same slot.

**Why.** In standalone Redis this doesn't matter. In Redis Cluster (which any serious deployment uses), keys scatter across shards. Without hash tags, a 5-feature pipelined read becomes 5 separate round-trips because the keys are on 5 different shards. With hash tags, one round-trip.

**Tradeoff.** Hot keys: if one feature group has a popular entity (e.g. an "anonymous user" sentinel), all that load lands on one shard. Mitigation is a separate group for high-cardinality flat data, not a workaround inside the key format.

---

## D-004: DuckDB ASOF JOIN for point-in-time correctness

**Status:** Accepted

**Context.** PIT joins are the heart of feature-store correctness. The naive `JOIN ON entity_id` leaks future data into training (you join features that didn't exist yet at label time). The fix is: for each (entity, label_ts), grab the most recent feature row where `feature_ts <= label_ts`.

**Decision.** DuckDB's `ASOF LEFT JOIN`. One process, no Spark cluster, no JVM, runs on a laptop.

**Why.**
- Spark is the textbook answer and it's overkill until you have ≥100M label rows.
- Pandas can do it with `merge_asof`, but it materializes everything in memory.
- DuckDB streams parquet files, uses query pushdown, and runs the join in C++.

Benchmarked: on a 10M-row spine joined against 50M feature rows, DuckDB finishes in ~12s on a 16-core laptop. Spark on the same data needed ~3min including JVM warmup.

**Reverted from.** I had a custom Python implementation that sorted and walked both streams. It worked, but was 10x slower than DuckDB and I'd have to maintain it forever.

---

## D-005: In-process registry (DuckDB) for OSS, swap for Postgres in production

**Status:** Accepted with a known followup

**Context.** Where does feature-group/model/deployment metadata live?

**Decision.** DuckDB file in the OSS distribution; a clean `Registry` interface so production deployments can swap in Postgres.

**Why.** Postgres requires people to run a database to try the project. That's a barrier. DuckDB is "just a file" and gives us transactions and SQL. The interface is the contract; the storage is a detail.

**Followup.** Postgres adapter when the first user asks for multi-instance deployment. Schema is identical, mostly a SQLAlchemy port.

---

## D-006: Consistent-hash canary routing, not random

**Status:** Accepted

**Context.** Canary deployments route X% of traffic to the new version. How?

**Decision.** Hash `(model_name, entity_id)` with SHA-256, mod 100, compare to traffic_pct.

**Why random was wrong.** With random routing, the same user can hit the old model and the new model on consecutive requests. If those models differ meaningfully, the user gets inconsistent experiences — recommendations that flip, pricing that wobbles. Also, you can't do clean A/B analysis because users aren't cleanly assigned to a cohort.

Consistent hashing pins each entity to one bucket. Same entity always sees the same version. A/B comparisons are clean. User experience is consistent.

**Tradeoff.** Self-correlated traffic spikes can shift the actual split — if one heavy user is in the canary bucket, the canary may serve more than X% of requests by volume even though it serves X% of users. For most use cases this is fine.

---

## D-007: At-least-once delivery from Kafka

**Status:** Accepted

**Context.** Streaming ingestion needs a delivery guarantee.

**Decision.** At-least-once. Commit Kafka offsets only after both online and offline writes succeed. Duplicate writes are OK because:
- Online store: same key + same value = no-op.
- Offline store: parquet appends with duplicate event IDs are deduplicated in the PIT join (DISTINCT on entity + event_ts grabs the latest).

**Why not exactly-once.** Kafka exactly-once requires transactional producers/consumers, Kafka transactions, and idempotent sinks. The complexity payoff vs. "make consumers idempotent" isn't there for this use case.

---

## R-001: Reverted — separate online and offline ingestion paths

I initially had two ingestion pipelines: a real-time Kafka -> Redis path and a separate batch Spark job for offline. The idea was the standard lambda architecture: speed layer + batch layer.

**Why I reverted it.** Two pipelines is two places to keep feature definitions in sync. The training/serving skew that point-in-time joins fix at query time was reintroduced at pipeline-definition time, just one level higher.

Single ingestion pipeline writes to both stores from the same transform function. One source of truth.

## R-002: Reverted — protobuf for online store values

Tried encoding feature values as protobuf so types were enforced end-to-end. The encoding/decoding overhead (~3-5ms per call) ate the latency budget for the hot path. Reverted to typed-string encoding with the dtype on the FeatureGroup spec.
