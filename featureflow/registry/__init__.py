"""Metadata registry for feature groups, models, and deployments.

DuckDB is overkill for this use case (a few hundred rows), but it gives us
a transactional file-backed store with zero ops. SQLite would also work;
DuckDB is just nicer for joins when querying lineage.

Production deployments should swap this for Postgres — the interface is
intentionally thin so the swap is a single class.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import duckdb

from ..settings import get_settings
from ..types import Deployment, DeploymentMode, FeatureGroup, ModelVersion

log = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS feature_groups (
    name VARCHAR NOT NULL,
    version INTEGER NOT NULL,
    payload JSON NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (name, version)
);

CREATE TABLE IF NOT EXISTS models (
    name VARCHAR NOT NULL,
    version INTEGER NOT NULL,
    payload JSON NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (name, version)
);

CREATE TABLE IF NOT EXISTS deployments (
    id INTEGER PRIMARY KEY,
    model_name VARCHAR NOT NULL,
    model_version INTEGER NOT NULL,
    mode VARCHAR NOT NULL,
    traffic_pct INTEGER NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE SEQUENCE IF NOT EXISTS deployments_id_seq START 1;
"""


class Registry:
    def __init__(self, db_path: str | Path | None = None):
        path = Path(db_path or get_settings().registry_db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(path))
        for stmt in SCHEMA.strip().split(";"):
            if stmt.strip():
                self._con.execute(stmt)

    # ---- Feature groups ---------------------------------------------------

    def register_feature_group(self, group: FeatureGroup) -> None:
        existing = self.get_feature_group(group.name, group.version)
        if existing is not None:
            # Schema evolution check: features can only be added, never removed
            # in a non-bumped version. See DECISIONS.md D-002.
            if set(group.feature_names()) < set(existing.feature_names()):
                removed = set(existing.feature_names()) - set(group.feature_names())
                raise ValueError(
                    f"Cannot remove features {removed} from {group.name} v{group.version} "
                    f"without bumping version. Existing consumers may break."
                )
            log.info("Updating feature group %s v%d", group.name, group.version)
        self._con.execute(
            "INSERT OR REPLACE INTO feature_groups (name, version, payload) VALUES (?, ?, ?)",
            (group.name, group.version, group.model_dump_json()),
        )

    def get_feature_group(self, name: str, version: int | None = None) -> FeatureGroup | None:
        if version is None:
            row = self._con.execute(
                "SELECT payload FROM feature_groups WHERE name=? ORDER BY version DESC LIMIT 1",
                (name,),
            ).fetchone()
        else:
            row = self._con.execute(
                "SELECT payload FROM feature_groups WHERE name=? AND version=?",
                (name, version),
            ).fetchone()
        if not row:
            return None
        return FeatureGroup.model_validate(json.loads(row[0]))

    def list_feature_groups(self) -> list[FeatureGroup]:
        rows = self._con.execute(
            "SELECT payload FROM feature_groups ORDER BY name, version"
        ).fetchall()
        return [FeatureGroup.model_validate(json.loads(r[0])) for r in rows]

    # ---- Models -----------------------------------------------------------

    def register_model(self, model: ModelVersion) -> None:
        self._con.execute(
            "INSERT OR REPLACE INTO models (name, version, payload) VALUES (?, ?, ?)",
            (model.name, model.version, model.model_dump_json()),
        )

    def get_model(self, name: str, version: int | None = None) -> ModelVersion | None:
        if version is None:
            row = self._con.execute(
                "SELECT payload FROM models WHERE name=? ORDER BY version DESC LIMIT 1",
                (name,),
            ).fetchone()
        else:
            row = self._con.execute(
                "SELECT payload FROM models WHERE name=? AND version=?",
                (name, version),
            ).fetchone()
        if not row:
            return None
        return ModelVersion.model_validate(json.loads(row[0]))

    # ---- Deployments ------------------------------------------------------

    def deploy(self, deployment: Deployment) -> None:
        """Register a deployment. Production deployments deactivate previous
        production deployments of the same model name (so there's always at
        most one active prod per model)."""
        if deployment.mode == DeploymentMode.PRODUCTION:
            self._con.execute(
                "UPDATE deployments SET active=FALSE "
                "WHERE model_name=? AND mode=? AND active=TRUE",
                (deployment.model_name, DeploymentMode.PRODUCTION.value),
            )
        self._con.execute(
            "INSERT INTO deployments (id, model_name, model_version, mode, traffic_pct) "
            "VALUES (nextval('deployments_id_seq'), ?, ?, ?, ?)",
            (
                deployment.model_name,
                deployment.model_version,
                deployment.mode.value,
                deployment.traffic_pct,
            ),
        )

    def active_deployments(self, model_name: str) -> list[Deployment]:
        rows = self._con.execute(
            "SELECT model_name, model_version, mode, traffic_pct, created_at "
            "FROM deployments WHERE model_name=? AND active=TRUE",
            (model_name,),
        ).fetchall()
        return [
            Deployment(
                model_name=r[0],
                model_version=r[1],
                mode=DeploymentMode(r[2]),
                traffic_pct=r[3],
                created_at=r[4],
            )
            for r in rows
        ]

    def close(self) -> None:
        self._con.close()
