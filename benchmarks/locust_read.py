"""Load test for the feature read endpoint."""
from __future__ import annotations

import random

from locust import HttpUser, between, task


class ReadUser(HttpUser):
    wait_time = between(0.05, 0.2)

    @task
    def read_features(self):
        entity_id = f"u_{random.randint(0, 99_999)}"
        with self.client.post(
            "/features/read",
            json={"group": "user_activity_30d", "entity_id": entity_id},
            catch_response=True,
            name="/features/read",
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"got {resp.status_code}")
