"""Command-line interface for FeatureFlow."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .registry import Registry
from .types import Deployment, DeploymentMode, FeatureGroup

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def register(spec: Path = typer.Argument(..., help="Path to a JSON feature-group spec.")):
    """Register a feature group from a JSON file."""
    data = json.loads(spec.read_text())
    group = FeatureGroup.model_validate(data)
    Registry().register_feature_group(group)
    console.print(f"[green]Registered[/green] {group.name} v{group.version}")


@app.command()
def list_groups():
    """List all registered feature groups."""
    groups = Registry().list_feature_groups()
    table = Table(title="Feature groups")
    table.add_column("Name")
    table.add_column("Version", justify="right")
    table.add_column("Entity")
    table.add_column("# features", justify="right")
    table.add_column("TTL (s)", justify="right")
    for g in groups:
        table.add_row(g.name, str(g.version), g.entity, str(len(g.features)), str(g.ttl_seconds))
    console.print(table)


@app.command()
def deploy(
    model_name: str,
    model_version: int,
    mode: str = typer.Option("production", help="production | canary | shadow"),
    traffic_pct: int = typer.Option(100, help="For canary, percent of traffic."),
):
    """Deploy a model version."""
    d = Deployment(
        model_name=model_name,
        model_version=model_version,
        mode=DeploymentMode(mode),
        traffic_pct=traffic_pct,
    )
    Registry().deploy(d)
    console.print(f"[green]Deployed[/green] {model_name} v{model_version} as {mode}")


@app.command()
def serve(
    host: str = "0.0.0.0",
    port: int = 8080,
):
    """Start the HTTP API server."""
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("featureflow.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
