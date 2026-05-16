"""
LangGraph node implementations for the audit workflow.

Node execution order
--------------------
1. ingestor_node      — synchronous; calls Layer 1-4 engines
2. (parallel fan-out)
   architect_node     — async LLM; L4 + L3 → ai_reasoning, verdict
   quantifier_node    — async LLM; L2 + L3 + logic → radar_metrics, scores
   mapmaker_node      — sync logic; L2 → file_structure_health
   refactor_node      — async LLM; L1 smells + L4 snippets → diffs
3. aggregator_node    — sync; merges all outputs → FinalAuditReport
"""
from __future__ import annotations

import json
import logging
from typing import Any

from devaudt.analyzer import analyze_url
from devaudt.analyzer.risk import RiskScoringEngine
from devaudt.analyzer.correlation import EvidenceCorrelationEngine
from devaudt.analyzer.context import ContextCompressor

from langgraph_config import LLM_PROVENANCE, get_llm, with_retry
from models import (
    ArchitectOutput,
    AuditState,
    DeterministicReport,
    FinalAuditReport,
    LayerData,
    LLMInsights,
    MapmakerOutput,
    QuantifierOutput,
    RadarMetric,
    RefactorLLMOutput,
    RefactorLLMMeta,
    RefactorAnalysis,
    RefactorDiff,
    RefactorOutput,
    RefactorSource,
    RefactorSuggestion,
)
from utils import (
    build_refactor_scaffold,
    compute_base_radar_scores,
    compute_file_structure_health,
    extract_analysis_metrics,
    extract_cluster_summary,
    extract_dependency_summary,
    extract_finding_summary,
    extract_quick_wins,
    extract_repository_summary,
    extract_security_audit,
    extract_top_risky_entities,
    get_progress_message,
    normalise_severity,
    scale_progress,
    utc_now_iso,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Progress reporting (configured by orchestrator before graph invocation)
# ---------------------------------------------------------------------------

_progress_reporter = None
_progress_hwm: int = -1  # high-water mark — never go backward


def set_progress_reporter(fn) -> None:
    """Register a progress callback: fn(percent: int, stage: str) -> None."""
    global _progress_reporter, _progress_hwm
    _progress_reporter = fn
    _progress_hwm = -1  # reset for each new job


def _emit_progress(percent: int, stage: str) -> None:
    global _progress_hwm
    if _progress_reporter is not None and percent > _progress_hwm:
        _progress_hwm = percent
        try:
            _progress_reporter(percent, stage)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Fallback outputs — returned when an LLM node exhausts all retries
# ---------------------------------------------------------------------------

_FALLBACK_ARCHITECT = ArchitectOutput(
    ai_reasoning=(
        "Automated architectural analysis encountered a processing error. "
        "Manual review is recommended."
    ),
    overall_verdict="Needs Review",
    confidence=0.1,
    architectural_recommendations=[],
)

_FALLBACK_QUANTIFIER = QuantifierOutput(
    radar_metrics=[
        RadarMetric(subject="Maintainability", score=50, max=100),
        RadarMetric(subject="Scalability", score=50, max=100),
        RadarMetric(subject="Security", score=50, max=100),
        RadarMetric(subject="Performance", score=50, max=100),
        RadarMetric(subject="Test Coverage", score=50, max=100),
        RadarMetric(subject="Architecture", score=50, max=100),
    ],
    overall_score=50,
    technical_debt_score=50,
    growth_points=0,
)


# ---------------------------------------------------------------------------
# Node 1: Ingestor
# ---------------------------------------------------------------------------

def ingestor_node(state: AuditState) -> dict[str, Any]:
    """
    Synchronous node.
    Runs the existing Layer 1-4 engines and populates AuditState.layers.
    This node produces only deterministic hard facts — no LLM involved.
    """
    logger.info("[Ingestor] Running Layer 1-4 engines for %s", state.repo_url)
    _emit_progress(scale_progress(0, 0, 25), "ingestor_start")

    # Layer 1 — raw analysis
    audit_result = analyze_url(state.repo_url)

    # Layer 2 — risk scoring
    risk_report = RiskScoringEngine().score(audit_result)

    # Layer 3 — evidence correlation
    correlation_report = EvidenceCorrelationEngine().correlate(risk_report)

    # Layer 4 — context compression
    context_packet = ContextCompressor().compress(
        audit_result,
        risk_report,
        correlation_report,
        top_n=3,
        token_budget=8000,
    )

    _emit_progress(scale_progress(100, 0, 25), "ingestor_done")
    return {
        "layers": LayerData(
            audit_result=audit_result.to_dict(),
            risk_report=risk_report.to_dict(),
            correlation_report=correlation_report.to_dict(),
            context_packet=context_packet.to_dict(),
        )
    }


# ---------------------------------------------------------------------------
# Node 2: Architect (async LLM)
# ---------------------------------------------------------------------------

@with_retry()
async def _call_architect_llm(formatted_prompt: str, clusters_json: str) -> ArchitectOutput:
    llm = get_llm(temperature=0.3)
    structured = llm.with_structured_output(ArchitectOutput, method="function_calling")
    prompt = (
        "You are an expert software architect reviewing a codebase audit report.\n\n"
        f"## Audit Context\n{formatted_prompt}\n\n"
        f"## Top Hotspot Clusters\n```json\n{clusters_json}\n```\n\n"
        "Based on the above, generate:\n"
        "1. `ai_reasoning` — 2-3 sentences on architecture quality and systemic issues.\n"
        "2. `overall_verdict` — Exactly one of: "
        "'Production Ready' | 'Ready with Caution' | 'Needs Review' | 'Critical Issues'\n"
        "3. `confidence` — Float 0.0-1.0 reflecting your certainty.\n"
        "4. `architectural_recommendations` — 3-5 actionable structural improvements.\n"
    )
    result = await structured.ainvoke(prompt)
    if result is None:
        raise ValueError("LLM returned no structured output for ArchitectOutput")
    return result


async def architect_node(state: AuditState) -> dict[str, Any]:
    """Async LLM node. Generates ai_reasoning and overall_verdict."""
    logger.info("[Architect] Starting architectural reasoning")
    _emit_progress(scale_progress(0, 25, 40), "architect_start")

    context_packet: dict[str, Any] = state.layers.context_packet or {}
    correlation_report: dict[str, Any] = state.layers.correlation_report or {}

    formatted_prompt: str = context_packet.get("formatted_prompt", "")
    top_clusters = (correlation_report.get("clusters") or [])[:3]
    clusters_json = json.dumps(top_clusters, indent=2)

    try:
        result = await _call_architect_llm(formatted_prompt, clusters_json)
    except Exception as exc:
        logger.error("[Architect] All retries exhausted: %s", exc)
        _emit_progress(scale_progress(100, 25, 40), "architect_done")
        return {
            "architect_output": _FALLBACK_ARCHITECT,
            "errors": [f"architect_node: {exc}"],
        }

    logger.info("[Architect] Verdict: %s (confidence=%.2f)", result.overall_verdict, result.confidence)
    _emit_progress(scale_progress(100, 25, 40), "architect_done")
    return {"architect_output": result}


# ---------------------------------------------------------------------------
# Node 3: Quantifier (async LLM)
# ---------------------------------------------------------------------------

@with_retry()
async def _call_quantifier_llm(context: str) -> QuantifierOutput:
    llm = get_llm(temperature=0.1)
    structured = llm.with_structured_output(QuantifierOutput, method="function_calling")
    result = await structured.ainvoke(context)
    if result is None:
        raise ValueError("LLM returned no structured output for QuantifierOutput")
    return result


async def quantifier_node(state: AuditState) -> dict[str, Any]:
    """
    Async hybrid node.
    Computes heuristic base scores from L2 then asks the LLM to calibrate
    them and derive composite scores.
    """
    logger.info("[Quantifier] Computing radar metrics")
    _emit_progress(scale_progress(0, 40, 55), "quantifier_start")

    risk_report: dict[str, Any] = state.layers.risk_report or {}
    audit_result: dict[str, Any] = state.layers.audit_result or {}
    correlation_report: dict[str, Any] = state.layers.correlation_report or {}

    risk_summary: dict[str, Any] = risk_report.get("summary", {})
    metrics_info: dict[str, Any] = audit_result.get("metrics", {})
    issue_kind_totals: dict[str, int] = risk_summary.get("issue_kind_totals", {})
    total_findings: int = int(risk_summary.get("total_findings", 0))

    base_scores = compute_base_radar_scores(
        risk_summary, metrics_info, issue_kind_totals, total_findings
    )

    correlation_summary = correlation_report.get("summary", {})

    prompt = (
        "You are a software quality quantifier. "
        "Given the deterministic base scores below, calibrate them "
        "and produce the final radar metrics with an overall composite score.\n\n"
        "## Deterministic Base Scores (0-100)\n"
        + json.dumps(base_scores, indent=2)
        + "\n\n## Risk Summary\n"
        + json.dumps(risk_summary, indent=2)
        + "\n\n## Correlation Summary\n"
        + json.dumps(correlation_summary, indent=2)
        + "\n\n"
        "Instructions:\n"
        "- Return exactly 6 `radar_metrics` with subjects: "
        "Maintainability, Scalability, Security, Performance, Test Coverage, Architecture.\n"
        "- `overall_score`: weighted composite of the 6 metrics (0-100 int).\n"
        "- `technical_debt_score`: estimated technical debt level (0-100 int).\n"
        "- `growth_points`: number of high-ROI improvements (0-10 int).\n"
        "Adjust scores based on the risk summary context; do not blindly copy base scores."
    )

    try:
        result = await _call_quantifier_llm(prompt)
    except Exception as exc:
        logger.error("[Quantifier] All retries exhausted: %s", exc)
        # Fill fallback with logic-derived scores instead of fixed 50s
        fallback_metrics = [
            RadarMetric(subject=k, score=int(v), max=100)
            for k, v in base_scores.items()
        ]
        overall = round(sum(m.score for m in fallback_metrics) / len(fallback_metrics))
        _emit_progress(scale_progress(100, 40, 55), "quantifier_done")
        return {
            "quantifier_output": QuantifierOutput(
                radar_metrics=fallback_metrics,
                overall_score=overall,
                technical_debt_score=50,
                growth_points=0,
            ),
            "errors": [f"quantifier_node: {exc}"],
        }

    logger.info("[Quantifier] Overall score: %d", result.overall_score)
    _emit_progress(scale_progress(100, 40, 55), "quantifier_done")
    return {"quantifier_output": result}


# ---------------------------------------------------------------------------
# Node 4: Mapmaker (synchronous logic — no LLM)
# ---------------------------------------------------------------------------

def mapmaker_node(state: AuditState) -> dict[str, Any]:
    """
    Synchronous deterministic node.
    Groups risk profiles by directory and computes health scores.
    """
    logger.info("[Mapmaker] Computing file structure health")
    _emit_progress(scale_progress(0, 55, 65), "mapmaker_start")

    risk_report: dict[str, Any] = state.layers.risk_report or {}
    profiles: list[dict[str, Any]] = risk_report.get("profiles", [])

    health = compute_file_structure_health(profiles)
    _emit_progress(scale_progress(100, 55, 65), "mapmaker_done")
    return {"mapmaker_output": MapmakerOutput(file_structure_health=health)}


# ---------------------------------------------------------------------------
# Node 5: Refactor (async LLM — iterates over top code smells)
# ---------------------------------------------------------------------------

@with_retry()
async def _call_refactor_llm(smell: dict[str, Any], formatted_prompt: str) -> RefactorLLMOutput:
    llm = get_llm(temperature=0.4)
    structured = llm.with_structured_output(RefactorLLMOutput, method="function_calling")

    file_path = smell.get("file", "unknown")
    line = smell.get("line", 0)
    symbol = smell.get("symbol", "unknown")
    smell_type = smell.get("type", "unknown")

    prompt = (
        "You are an expert code reviewer. "
        "A code smell has been identified in the following codebase audit.\n\n"
        f"## Audit Context (relevant excerpt)\n{formatted_prompt[:3000]}\n\n"
        f"## Code Smell Details\n"
        f"- **File**: `{file_path}`\n"
        f"- **Symbol**: `{symbol}`\n"
        f"- **Line**: {line}\n"
        f"- **Type**: {smell_type}\n"
        f"- **Severity**: {smell.get('severity', 'unknown')}\n\n"
        "Generate a refactor suggestion:\n"
        "- `title`: Short descriptive title of the issue.\n"
        "- `confidence`: Float 0.0-1.0.\n"
        "- `reasoning`: Explain the issue and why the refactor improves it.\n"
        "- `diff.before`: The problematic code (realistic pseudocode or real snippet).\n"
        "- `diff.after`: The refactored version.\n"
    )
    result = await structured.ainvoke(prompt)
    if result is None:
        raise ValueError("LLM returned no structured output for RefactorLLMOutput")
    return result


async def refactor_node(state: AuditState) -> dict[str, Any]:
    """
    Async LLM node.
    Generates refactor suggestions for the top N code smells from L1.
    """
    logger.info("[Refactor] Generating refactor suggestions")
    _emit_progress(scale_progress(0, 65, 85), "refactor_start")

    audit_result: dict[str, Any] = state.layers.audit_result or {}
    context_packet: dict[str, Any] = state.layers.context_packet or {}

    code_smells: list[dict[str, Any]] = audit_result.get("code_smells", [])
    commit_hash: str = (
        audit_result.get("repository", {}).get("commit_hash", "unknown")
    )
    formatted_prompt: str = context_packet.get("formatted_prompt", "")

    # Work on top 5 smells by severity to keep LLM costs manageable
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    top_smells = sorted(
        code_smells,
        key=lambda s: severity_order.get(normalise_severity(s.get("severity", "low")), 4),
    )[:5]
    _smell_count = len(top_smells)

    suggestions: list[RefactorSuggestion] = []
    errors: list[str] = []

    for idx, smell in enumerate(top_smells, start=1):
        scaffold = build_refactor_scaffold(smell, idx, commit_hash, LLM_PROVENANCE)
        try:
            llm_out: RefactorLLMOutput = await _call_refactor_llm(smell, formatted_prompt)
        except Exception as exc:
            logger.error("[Refactor] Smell %d failed after retries: %s", idx, exc)
            errors.append(f"refactor_node[{idx}]: {exc}")
            # Degenerate suggestion — deterministic metadata still present
            llm_out = RefactorLLMOutput(
                title=smell.get("title", "Code Smell"),
                confidence=0.0,
                reasoning="Refactor suggestion unavailable due to a processing error.",
                diff=RefactorDiff(
                    before=f"# {smell.get('file', 'unknown')}:{smell.get('line', 0)}",
                    after="# Refactor required — see code smell details above.",
                ),
            )

        suggestion = RefactorSuggestion(
            title=llm_out.title,
            suggestion_id=scaffold["suggestion_id"],
            source=RefactorSource(**scaffold["source"]),
            analysis=RefactorAnalysis(
                **{**scaffold["analysis"], "confidence": llm_out.confidence}
            ),
            llm=RefactorLLMMeta(**scaffold["llm"]),
            reasoning=llm_out.reasoning,
            diff=llm_out.diff,
        )
        suggestions.append(suggestion)
        local_pct = round(idx / max(_smell_count, 1) * 100)
        _emit_progress(
            scale_progress(local_pct, 65, 85),
            f"refactor_{idx}_of_{_smell_count}",
        )

    _emit_progress(scale_progress(100, 65, 85), "refactor_done")
    logger.info("[Refactor] Generated %d suggestions", len(suggestions))
    return {
        "refactor_output": RefactorOutput(refactor_suggestions=suggestions),
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Node 6: Aggregator (synchronous logic)
# ---------------------------------------------------------------------------

def aggregator_node(state: AuditState) -> dict[str, Any]:
    """
    Synchronous node.
    Merges all partial outputs into a FinalAuditReport.
    The orchestrator injects the transport envelope (job_id, repo_url, etc.).
    """
    logger.info("[Aggregator] Assembling final report for job %s", state.job_id)
    _emit_progress(scale_progress(0, 85, 100), "aggregator_start")

    audit_result: dict[str, Any] = state.layers.audit_result or {}
    risk_report: dict[str, Any] = state.layers.risk_report or {}
    correlation_report: dict[str, Any] = state.layers.correlation_report or {}
    commit_hash: str = audit_result.get("repository", {}).get("commit_hash", "unknown")

    # ---- Architect --------------------------------------------------------
    arch = state.architect_output or _FALLBACK_ARCHITECT

    # ---- Quantifier -------------------------------------------------------
    quant = state.quantifier_output or _FALLBACK_QUANTIFIER

    # ---- Mapmaker ---------------------------------------------------------
    mapmaker = state.mapmaker_output or MapmakerOutput()

    # ---- Refactor ---------------------------------------------------------
    refactor = state.refactor_output or RefactorOutput()

    # ---- Deterministic report sections ------------------------------------
    findings: list[dict[str, Any]] = audit_result.get("findings", [])
    security_info: dict[str, Any] = audit_result.get("security", {})

    det_report = DeterministicReport(
        repository_summary=extract_repository_summary(audit_result),
        analysis_metrics=extract_analysis_metrics(audit_result),
        finding_summary=extract_finding_summary(audit_result, risk_report.get("summary", {})),
        top_risky_entities=extract_top_risky_entities(risk_report),
        cluster_summary=extract_cluster_summary(correlation_report),
        dependency_summary=extract_dependency_summary(audit_result),
        file_structure_health=mapmaker.file_structure_health,
        security_audit=extract_security_audit(security_info),
        quick_wins=extract_quick_wins(findings, commit_hash),
    )

    # ---- LLM insights section --------------------------------------------
    llm_insights = LLMInsights(
        overall_score=quant.overall_score,
        technical_debt_score=quant.technical_debt_score,
        growth_pts=quant.growth_points,
        radar_metrics=quant.radar_metrics,
        overall_verdict=arch.overall_verdict,
        confidence=arch.confidence,
        ai_reasoning=arch.ai_reasoning,
        architectural_recommendations=arch.architectural_recommendations,
        refactor_suggestions=refactor.refactor_suggestions,
    )

    report = FinalAuditReport(
        deterministic_report=det_report,
        llm_insights=llm_insights,
    )

    logger.info(
        "[Aggregator] Report complete — overall_score=%d, verdict=%s",
        llm_insights.overall_score,
        llm_insights.overall_verdict,
    )
    _emit_progress(scale_progress(100, 85, 100), "aggregator_done")
    return {"final_report": report}
