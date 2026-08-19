"""
monitoring.py
-------------
A small, dependency-free monitoring layer for the API.

I keep this deliberately simple (in-memory counters, no external metrics
database) because the brief asks for *basic* monitoring for a *local*
service -- not a production Prometheus/Grafana setup. Everything here
resets when the process restarts, which is fine for this assignment's
scope; I call that out explicitly in observations.md rather than pretend
it's more durable than it is.

What it tracks, matching Part F of the assignment:
  - total prediction requests
  - successful vs failed requests
  - recent + average prediction latency
  - prediction-class distribution (how many 0s vs 1s predicted)
  - missing/invalid input counts (validation failures)

What it deliberately does NOT do:
  - log raw patient input values (age, glucose level, etc.) -- only
    aggregated, non-identifying counters and the prediction/probability/
    latency of each call are recorded, so nothing sensitive ends up in
    logs or in the /monitoring endpoint.
"""

import logging
import time
from collections import Counter, deque
from threading import Lock

logger = logging.getLogger("stroke_api.monitoring")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

RECENT_WINDOW = 100  # how many recent latencies to keep for a "recent average"


class MonitoringStats:
    """Thread-safe in-memory counters for the running API process."""

    def __init__(self):
        self._lock = Lock()
        self.start_time = time.time()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.invalid_input_requests = 0
        self.prediction_class_counts = Counter()
        self.recent_latencies_ms = deque(maxlen=RECENT_WINDOW)
        self.all_time_latency_sum_ms = 0.0
        self.all_time_latency_count = 0

    def record_success(self, prediction: int, latency_ms: float, model_version: str):
        with self._lock:
            self.total_requests += 1
            self.successful_requests += 1
            self.prediction_class_counts[prediction] += 1
            self.recent_latencies_ms.append(latency_ms)
            self.all_time_latency_sum_ms += latency_ms
            self.all_time_latency_count += 1
        # Log only non-sensitive, aggregated fields -- never the raw patient
        # input (age, glucose level, bmi, etc).
        logger.info(
            "prediction_ok model_version=%s prediction=%s latency_ms=%.2f",
            model_version, prediction, latency_ms,
        )

    def record_failure(self, reason: str, model_version: str = "unknown"):
        with self._lock:
            self.total_requests += 1
            self.failed_requests += 1
        logger.warning("prediction_failed model_version=%s reason=%s", model_version, reason)

    def record_invalid_input(self, model_version: str = "unknown"):
        with self._lock:
            self.total_requests += 1
            self.failed_requests += 1
            self.invalid_input_requests += 1
        logger.warning("invalid_input model_version=%s", model_version)

    def snapshot(self) -> dict:
        with self._lock:
            recent_avg = (
                sum(self.recent_latencies_ms) / len(self.recent_latencies_ms)
                if self.recent_latencies_ms else None
            )
            all_time_avg = (
                self.all_time_latency_sum_ms / self.all_time_latency_count
                if self.all_time_latency_count else None
            )
            return {
                "uptime_seconds": round(time.time() - self.start_time, 1),
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "invalid_input_requests": self.invalid_input_requests,
                "prediction_class_distribution": dict(self.prediction_class_counts),
                "average_latency_ms_recent_100": (
                    round(recent_avg, 3) if recent_avg is not None else None
                ),
                "average_latency_ms_all_time": (
                    round(all_time_avg, 3) if all_time_avg is not None else None
                ),
            }


# A single shared instance the API module imports and updates.
stats = MonitoringStats()
