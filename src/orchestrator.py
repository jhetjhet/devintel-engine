"""
Entry point for the AI-powered repo audit worker.

Usage
-----
    python orchestrator.py <repo_url> <job_id> [commit_hash]

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

def _clear_job_keys(r: redis.Redis, job_id: str, commit_hash: str) -> None:
    """Delete all persisted state keys for a job before starting a fresh run."""
    keys = [
        f"devintel:{job_id}:{commit_hash}:status",
        f"devintel:{job_id}:{commit_hash}:result",
        f"devintel:{job_id}:{commit_hash}:terminal",
        f"devintel:{job_id}:{commit_hash}:progress",
    ]
    existing = [k for k in keys if r.exists(k)]
    if existing:
        r.delete(*existing)


def _set_status(r: redis.Redis, job_id: str, commit_hash: str, status: str) -> None:
    r.set(f"devintel:{job_id}:{commit_hash}:status", status)


def _set_result(r: redis.Redis, job_id: str, commit_hash: str, payload: dict) -> None:
    r.set(f"devintel:{job_id}:{commit_hash}:result", json.dumps(payload))


def _publish_terminal(
    r: redis.Redis,
    job_id: str,
    channel: str,
    commit_hash: str,
    level: str,
    message: str,
) -> None:
    """Publish a terminal event and persist it so subscribers that miss the
    pub/sub message can still detect job completion by polling the key
    ``devintel:<job_id>:<commit_hash>:terminal``.
    """
    event = json.dumps({
        "timestamp": utc_now_iso(),
        "level": level,
        "logger": __name__,
        "message": message,
        "is_terminal": True,
    })
    try:
        r.set(f"devintel:{job_id}:{commit_hash}:terminal", event)
        r.publish(channel, event)
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: failed to publish terminal event: {exc}", file=sys.stderr)


def _publish_progress(
    r: redis.Redis,
    job_id: str,
    channel: str,
    commit_hash: str,
    percent: int,
    stage: str,
) -> None:
    """Persist current progress to ``devintel:<job_id>:<commit_hash>:progress`` and publish
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
        pipe.set(f"devintel:{job_id}:{commit_hash}:progress", percent)
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
        print("Usage: python orchestrator.py <repo_url> <job_id> [commit_hash]", file=sys.stderr)
        sys.exit(1)

    target_repo = sys.argv[1]
    target_job_id = sys.argv[2]
    commit_hash_arg = sys.argv[3] if len(sys.argv) > 3 else None

    # Redis connection (used for log streaming and job state)
    r = redis.Redis(host=os.environ.get("REDIS_HOST", "redis"), port=6379, db=0)
    _setup_logging(r, channel="devintel_engine_" + target_job_id)

    # Wire per-node progress events into Redis
    # If commit_hash is supplied as a CLI arg it is used immediately.
    # Otherwise the "unknown" placeholder is used and keys are renamed after
    # the audit completes and the real hash is extracted from the report.
    _job_channel = "devintel_engine_" + target_job_id
    commit_hash = commit_hash_arg or "unknown"
    set_progress_reporter(
        lambda pct, stage: _publish_progress(r, target_job_id, _job_channel, commit_hash, pct, stage)
    )

    _clear_job_keys(r, target_job_id, commit_hash)
    _set_status(r, target_job_id, commit_hash, "progress")
    logger = logging.getLogger(__name__)
    try:
        report = asyncio.run(run_audit(target_repo, target_job_id))
    except Exception as exc:
        _set_status(r, target_job_id, commit_hash, "error")
        _set_result(r, target_job_id, commit_hash, {"job_id": target_job_id, "status": "error", "error": str(exc)})
        logger.error("Audit failed: %s", exc, exc_info=True)
        # Publish a terminal event so subscribers know the job is done
        _publish_terminal(r, target_job_id, _job_channel, commit_hash, "ERROR", f"Audit failed: {exc}")
        sys.exit(1)

    # If no commit_hash was provided upfront, extract it from the report.
    if not commit_hash_arg:
        commit_hash = (
            report.get("deterministic_report", {})
            .get("repository_summary", {})
            .get("commit_hash", "unknown")
        )

    payload = {"full_audit_report": report}
    _set_status(r, target_job_id, commit_hash, "completed")
    _set_result(r, target_job_id, commit_hash, payload)
    _publish_progress(r, target_job_id, _job_channel, commit_hash, 100, "completed")
    # Publish a terminal event so subscribers know the job is done
    _publish_terminal(r, target_job_id, _job_channel, commit_hash, "INFO", "Audit completed successfully.")

    # Rename keys written under "unknown" placeholder to the real commit hash
    # (only needed when commit_hash was not provided as a CLI arg).
    if not commit_hash_arg and commit_hash != "unknown":
        for suffix in ("status", "progress"):
            old_key = f"devintel:{target_job_id}:unknown:{suffix}"
            new_key = f"devintel:{target_job_id}:{commit_hash}:{suffix}"
            try:
                if r.exists(old_key):
                    r.rename(old_key, new_key)
            except redis.ResponseError as exc:  # noqa: BLE001
                logger.warning("Could not rename Redis key %s → %s: %s", old_key, new_key, exc)