"""Offline feature store. Parquet files partitioned by feature_group and date.

DuckDB does the heavy lifting for point-in-time joins. The PIT join is the
single most important piece of feature-store correctness: when training,
each label row must see only features that existed at or before the label's
event time. Naive joins leak future information; PIT joins prevent that.

DuckDB has native ASOF JOIN support, which is exactly what we need here.
"""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..settings import get_settings
from ..types import FeatureGroup

log = logging.getLogger(__name__)


class OfflineStore:
    def __init__(self, root: str | Path | None = None):
        self._root = Path(root or get_settings().offline_store_path)
        self._root.mkdir(parents=True, exist_ok=True)

    def _group_path(self, group: FeatureGroup) -> Path:
        return self._root / f"{group.name}_v{group.version}"

    def write(self, group: FeatureGroup, df: pd.DataFrame) -> Path:
        """Append a batch of feature values to the offline store.

        df must contain: entity column (named after group.entity),
        '_event_ts' column (datetime64), and all feature columns.
        """
        self._validate_dataframe(group, df)
        path = self._group_path(group)
        path.mkdir(parents=True, exist_ok=True)

        # Partition by date for query pruning
        if "_event_ts" not in df.columns:
            raise ValueError("DataFrame missing required '_event_ts' column")
        df = df.copy()
        df["_event_date"] = pd.to_datetime(df["_event_ts"]).dt.date

        for date_val, chunk in df.groupby("_event_date"):
            partition_path = path / f"date={date_val}"
            partition_path.mkdir(exist_ok=True)
            fname = partition_path / f"part-{pd.Timestamp.utcnow().value}.parquet"
            table = pa.Table.from_pandas(chunk.drop(columns=["_event_date"]))
            pq.write_table(table, fname, compression="zstd")
            log.info("Wrote %d rows to %s", len(chunk), fname)

        return path

    def point_in_time_join(
        self,
        groups: list[FeatureGroup],
        spine: pd.DataFrame,
        entity_col: str,
        event_ts_col: str = "event_ts",
    ) -> pd.DataFrame:
        """ASOF join: for each row in spine, attach the latest feature values
        that existed at or before event_ts.

        spine: DataFrame of (entity_id, event_ts, ...labels). The training
        labels. Each row gets feature values frozen at event_ts.
        """
        con = duckdb.connect(":memory:")
        try:
            con.register("spine", spine)
            result = spine.copy()
            result["_join_ts"] = pd.to_datetime(spine[event_ts_col])

            for group in groups:
                feature_path = self._group_path(group)
                if not feature_path.exists():
                    log.warning("No data for group %s; skipping.", group.name)
                    continue

                glob = str(feature_path / "**" / "*.parquet")
                con.execute(f"""
                    CREATE OR REPLACE TEMP VIEW fg AS
                    SELECT * FROM read_parquet('{glob}')
                """)

                # Determine columns dynamically (the parquet may have extra cols)
                feat_cols = ", ".join(f'fg."{f.name}"' for f in group.features)

                # ASOF join: matching entity, fg._event_ts <= spine.event_ts,
                # take the most recent row.
                query = f"""
                    SELECT
                        spine.*,
                        {feat_cols},
                        fg._event_ts AS _ts_{group.name}
                    FROM spine
                    ASOF LEFT JOIN fg
                      ON spine.{entity_col} = fg.{group.entity}
                     AND spine.{event_ts_col} >= fg._event_ts
                """
                joined = con.execute(query).fetch_df()
                result = joined  # last assignment wins; in practice chain joins

            return result
        finally:
            con.close()

    @staticmethod
    def _validate_dataframe(group: FeatureGroup, df: pd.DataFrame) -> None:
        required = {group.entity, "_event_ts"} | set(group.feature_names())
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {sorted(missing)}")
