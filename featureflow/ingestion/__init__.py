"""Streaming ingestion from Kafka into the online and offline stores.

This module is optional — install with `pip install -e .[streaming]`.
For low-volume use cases the batch writes via the SDK are sufficient.

Design notes:
- We commit Kafka offsets only after successful writes to *both* online and
  offline. At-least-once delivery is the contract; downstream consumers need
  to be idempotent. Exactly-once would require Kafka transactions + a
  transactional sink, which is overkill for a feature store.
- Writes are batched. Online uses a Redis pipeline; offline accumulates rows
  and flushes to parquet every BATCH_SECONDS or BATCH_SIZE.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import pandas as pd

from ..settings import get_settings
from ..store.offline import OfflineStore
from ..store.online import OnlineStore
from ..types import FeatureGroup

log = logging.getLogger(__name__)


BATCH_SECONDS = 5.0
BATCH_SIZE = 1000


@dataclass
class IngestionStats:
    messages_consumed: int = 0
    rows_written_online: int = 0
    rows_written_offline: int = 0
    errors: int = 0
    last_offset: dict[str, int] = field(default_factory=dict)


TransformFn = Callable[[dict], Awaitable[tuple[str, dict, float]] | tuple[str, dict, float]]


class KafkaIngestor:
    """Consumes a Kafka topic, applies a transform, writes to both stores.

    transform: async or sync function (msg_dict) -> (entity_id, feature_values, event_ts).
        Return None to skip the message (e.g. filtered out).
    """

    def __init__(
        self,
        group: FeatureGroup,
        topic: str,
        transform: TransformFn,
        online: OnlineStore | None = None,
        offline: OfflineStore | None = None,
    ):
        self._group = group
        self._topic = topic
        self._transform = transform
        self._online = online or OnlineStore()
        self._offline = offline or OfflineStore()
        self.stats = IngestionStats()

    async def run(self, max_messages: int | None = None) -> IngestionStats:
        try:
            from aiokafka import AIOKafkaConsumer  # local import
        except ImportError as e:
            raise RuntimeError("Kafka ingestion requires `pip install featureflow[streaming]`") from e

        s = get_settings()
        consumer = AIOKafkaConsumer(
            self._topic,
            bootstrap_servers=s.kafka_bootstrap_servers,
            group_id=s.kafka_consumer_group,
            enable_auto_commit=False,
            value_deserializer=lambda b: json.loads(b.decode()),
        )
        await consumer.start()
        try:
            batch_rows: list[tuple[str, dict, float]] = []
            last_flush = asyncio.get_event_loop().time()

            async for msg in consumer:
                self.stats.messages_consumed += 1
                try:
                    result = self._transform(msg.value)
                    if asyncio.iscoroutine(result):
                        result = await result
                except Exception as e:
                    log.exception("Transform error: %s", e)
                    self.stats.errors += 1
                    continue

                if result is None:
                    continue

                entity_id, values, event_ts = result
                batch_rows.append((entity_id, values, event_ts))

                now = asyncio.get_event_loop().time()
                if len(batch_rows) >= BATCH_SIZE or now - last_flush >= BATCH_SECONDS:
                    await self._flush(batch_rows)
                    batch_rows = []
                    last_flush = now
                    await consumer.commit()

                if max_messages is not None and self.stats.messages_consumed >= max_messages:
                    break

            if batch_rows:
                await self._flush(batch_rows)
                await consumer.commit()

            return self.stats
        finally:
            await consumer.stop()

    async def _flush(self, rows: list[tuple[str, dict, float]]) -> None:
        # Online: pipelined writes
        n_online = await self._online.write_batch(self._group, rows)
        self.stats.rows_written_online += n_online

        # Offline: build a DataFrame and append
        df_rows = []
        for entity_id, values, ts in rows:
            row = dict(values)
            row[self._group.entity] = entity_id
            row["_event_ts"] = pd.to_datetime(ts, unit="s")
            df_rows.append(row)
        df = pd.DataFrame(df_rows)
        self._offline.write(self._group, df)
        self.stats.rows_written_offline += len(rows)

        log.info("Flushed batch: %d rows online, %d offline", n_online, len(rows))
