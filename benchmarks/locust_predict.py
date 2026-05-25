"""Locust load test for the prediction endpoint.

Run with:
    locust -f benchmarks/locust_predict.py --host http://localhost:8080
"""
from __future__ import annotations

import random

from locust import HttpUser, between, task


class PredictUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def predict_churn(self):
        entity_id = f"u_{random.randint(0, 99_999)}"
        with self.client.post(
            "/predict/churn",
            json={"entity_id": entity_id},
            catch_response=True,
            name="/predict/churn",
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"got {resp.status_code}: {resp.text[:200]}")
                return
            try:
                body = resp.json()
                if "prediction" not in body:
                    resp.failure("response missing prediction field")
            except Exception as e:
                resp.failure(f"bad json: {e}")
