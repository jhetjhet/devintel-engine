"""
Entry point for the AI-powered repo audit worker.

Usage
-----
    python orchestrator.py <repo_url>

Environment variables required
-------------------------------
    LLM_API_KEY        — LLM provider API key
    LLM_BASE_URL       — (optional) OpenAI-compatible base URL
    LLM_MODEL          — (optional) model ID, default: gpt-4o-mini
    LLM_MAX_RETRIES    — (optional) retry attempts per LLM node, default: 3
    LLM_RETRY_BASE_S   — (optional) back-off base seconds, default: 2.0

Output
------
    Prints the final dashboard-ready JSON report to stdout.
    Structured logs are published to Redis channel `devintel_engine_<job_id>`.
"""
import asyncio
import json
import logging
import os
import sys

import redis

from graph import audit_graph
from models import AuditState
from nodes import set_progress_reporter
from utils import utc_now_iso


# ---------------------------------------------------------------------------
# Redis log handler
# ---------------------------------------------------------------------------

class RedisHandler(logging.Handler):
    def __init__(self, redis_client: redis.Redis, channel: str = "logs") -> None:
        super().__init__()
        self.redis_client = redis_client
        self.channel = channel

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = {
                "timestamp": utc_now_iso(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "is_terminal": False,          # True only on job-done events
            }
            if record.exc_info:
                payload["exception"] = self.formatException(record.exc_info)
            self.redis_client.publish(self.channel, json.dumps(payload))
        except Exception:
            self.handleError(record)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(redis_client: redis.Redis, channel: str = "devintel_engine") -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Console handler — always present
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    )
    root_logger.addHandler(console)

    # Redis handler — publishes structured JSON to the per-job channel
    redis_handler = RedisHandler(redis_client, channel=channel)
    root_logger.addHandler(redis_handler)


# ---------------------------------------------------------------------------
# Redis job state helpers
# ---------------------------------------------------------------------------

def _set_status(r: redis.Redis, job_id: str, status: str) -> None:
    r.set(f"devintel:{job_id}:status", status)


def _set_result(r: redis.Redis, job_id: str, payload: dict) -> None:
    r.set(f"devintel:{job_id}:result", json.dumps(payload))


def _publish_terminal(
    r: redis.Redis,
    job_id: str,
    channel: str,
    level: str,
    message: str,
) -> None:
    """Publish a terminal event and persist it so subscribers that miss the
    pub/sub message can still detect job completion by polling the key
    ``devintel:<job_id>:terminal``.
    """
    event = json.dumps({
        "timestamp": utc_now_iso(),
        "level": level,
        "logger": __name__,
        "message": message,
        "is_terminal": True,
    })
    try:
        r.set(f"devintel:{job_id}:terminal", event)
        r.publish(channel, event)
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: failed to publish terminal event: {exc}", file=sys.stderr)


def _publish_progress(
    r: redis.Redis,
    job_id: str,
    channel: str,
    percent: int,
    stage: str,
) -> None:
    """Persist current progress to ``devintel:<job_id>:progress`` and publish
    a progress event on the job channel.  Subscribers can poll the key or
    listen on the channel; both carry ``progress_percent`` and ``progress_stage``.
    """
    event = json.dumps({
        "timestamp": utc_now_iso(),
        "level": "INFO",
        "logger": __name__,
        "message": f"[Progress] {stage}: {percent}%",
        "is_terminal": False,
        "progress_percent": percent,
        "progress_stage": stage,
    })
    try:
        pipe = r.pipeline()
        pipe.set(f"devintel:{job_id}:progress", percent)
        pipe.publish(channel, event)
        pipe.execute()
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: failed to publish progress: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_audit(repo_url: str, job_id: str) -> dict:
    """
    Invoke the LangGraph audit workflow and return the final report as a dict.
    """
    initial_state = AuditState(
        repo_url=repo_url,
        job_id=job_id,
        timestamp=utc_now_iso(),
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting audit — job_id=%s, repo=%s", initial_state.job_id, repo_url)

    try:
        raw: dict = await audit_graph.ainvoke(initial_state)
    except Exception as exc:
        logger.error("Graph invocation failed: %s", exc, exc_info=True)
        raise

    try:
        final_state = AuditState.model_validate(raw)
    except Exception as exc:
        logger.error("State validation failed after graph run: %s", exc, exc_info=True)
        raise

    if final_state.errors:
        logger.warning("Audit completed with %d error(s):", len(final_state.errors))
        for err in final_state.errors:
            logger.warning("  • %s", err)

    if final_state.final_report is None:
        logger.error("Aggregator did not produce a final report — state: %s", final_state.model_dump())
        raise RuntimeError("Aggregator did not produce a final report.")

    # Inject transport envelope — these are caller-owned fields not stored
    # in FinalAuditReport itself.
    return {
        "job_id": final_state.job_id,
        "repo_url": final_state.repo_url,
        "status": "completed",
        "timestamp": final_state.timestamp,
        **final_state.final_report.model_dump(),
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python orchestrator.py <repo_url> <job_id>", file=sys.stderr)
        sys.exit(1)

    target_repo = sys.argv[1]
    target_job_id = sys.argv[2]

    # Redis connection (used for log streaming and job state)
    r = redis.Redis(host=os.environ.get("REDIS_HOST", "redis"), port=6379, db=0)
    _setup_logging(r, channel="devintel_engine_" + target_job_id)

    # Wire per-node progress events into Redis
    _job_channel = "devintel_engine_" + target_job_id
    set_progress_reporter(
        lambda pct, stage: _publish_progress(r, target_job_id, _job_channel, pct, stage)
    )

    _set_status(r, target_job_id, "progress")
    logger = logging.getLogger(__name__)
    try:
        report = asyncio.run(run_audit(target_repo, target_job_id))
    except Exception as exc:
        _set_status(r, target_job_id, "error")
        _set_result(r, target_job_id, {"job_id": target_job_id, "status": "error", "error": str(exc)})
        logger.error("Audit failed: %s", exc, exc_info=True)
        # Publish a terminal event so subscribers know the job is done
        _publish_terminal(r, target_job_id, "devintel_engine_" + target_job_id, "ERROR", f"Audit failed: {exc}")
        sys.exit(1)

    _set_status(r, target_job_id, "completed")
    payload = {"full_audit_report": report}
    _set_result(r, target_job_id, payload)
    _publish_progress(r, target_job_id, "devintel_engine_" + target_job_id, 100, "completed")
    # Publish a terminal event so subscribers know the job is done
    _publish_terminal(r, target_job_id, "devintel_engine_" + target_job_id, "INFO", "Audit completed successfully.")

    # Emit the dashboard-ready JSON to stdout
    print(json.dumps(payload, indent=2))