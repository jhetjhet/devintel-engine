"""
Entry point for the AI-powered repo audit worker.

Usage
-----
    python orchestrator.py <repo_url>

Environment variables required
-------------------------------
    DEEPSEEK_API_KEY   — DeepSeek API key
    DEEPSEEK_BASE_URL  — (optional) override API base URL
    LLM_MODEL          — (optional) model ID, default: deepseek-chat
    LLM_MAX_RETRIES    — (optional) retry attempts per LLM node, default: 3
    LLM_RETRY_BASE_S   — (optional) back-off base seconds, default: 2.0

Output
------
    Prints the final dashboard-ready JSON report to stdout.
    Structured logs are published to Redis channel `devaudt_logs`.
"""
import asyncio
import json
import logging
import os
import sys

import redis

from graph import audit_graph
from models import AuditState
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
            self.redis_client.publish(self.channel, self.format(record))
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

    # Redis handler — publishes to devintel_engine channel
    redis_handler = RedisHandler(redis_client, channel=channel)
    redis_handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    )
    root_logger.addHandler(redis_handler)


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

    raw: dict = await audit_graph.ainvoke(initial_state)
    final_state = AuditState.model_validate(raw)

    if final_state.errors:
        logger.warning("Audit completed with %d error(s):", len(final_state.errors))
        for err in final_state.errors:
            logger.warning("  • %s", err)

    if final_state.final_report is None:
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

    # Redis connection (used for log streaming)
    r = redis.Redis(host=os.environ.get("REDIS_HOST", "redis"), port=6379, db=0)
    _setup_logging(r, channel="devintel_engine_" + target_job_id)

    report = asyncio.run(run_audit(target_repo, target_job_id))

    # Emit the dashboard-ready JSON to stdout
    print(json.dumps({"full_audit_report": report}, indent=2))