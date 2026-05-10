"""
Stateless helper functions used across the audit workflow nodes.
No LLM calls or side effects — pure deterministic transformations.
"""
from __future__ import annotations

import hashlib
import re
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from models import (
    AnalysisMetrics,
    ClusterRef,
    ClusterSummaryBlock,
    DependencySummary,
    FindingSummary,
    HealthEntry,
    QuickWin,
    RepositorySummary,
    RiskyEntityRef,
    SecurityAudit,
    SecurityVulnerability,
)


# ---------------------------------------------------------------------------
# Identity helpers
# ---------------------------------------------------------------------------

def generate_job_id() -> str:
    """Generate a short, human-readable audit job ID."""
    short = uuid.uuid4().hex[:8].upper()
    return f"AUDIT-{short}"


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# File structure health (Mapmaker logic)
# ---------------------------------------------------------------------------

def normalise_severity(raw: Any) -> str:
    """
    Convert any severity representation coming out of AnalysisResult.to_dict()
    to a canonical string label.

    to_dict() serialises Severity(IntEnum) as a normalised float in [0, 1]:
        LOW=1  → ~0.25    MEDIUM=2 → ~0.50
        HIGH=3 → ~0.75    CRITICAL=4 → 1.0

    String labels ("low", "high", etc.) are passed through unchanged.
    Integer enum values (1-4) are also handled.
    """
    if isinstance(raw, str):
        label = raw.lower()
        return label if label in ("low", "medium", "high", "critical") else "low"
    val = float(raw)
    # Integer enum path (1-4)
    if val >= 4:
        return "critical"
    if val >= 3:
        return "high"
    if val >= 2:
        return "medium"
    if val >= 1:
        return "low"
    # Normalised float path (0.0-1.0)
    if val >= 0.75:
        return "critical"
    if val >= 0.5:
        return "high"
    if val >= 0.25:
        return "medium"
    return "low"


def _health_status(score: int) -> str:
    if score >= 85:
        return "excellent"
    if score >= 70:
        return "good"
    if score >= 50:
        return "warning"
    return "critical"


def _dir_prefix(file_path: str, depth: int = 2) -> str:
    """
    Extract the leading directory path up to `depth` components.

    Examples
    --------
    >>> _dir_prefix("src/services/auth.py", depth=2)
    'src/services'
    >>> _dir_prefix("main.py", depth=2)
    '.'
    """
    parts = file_path.replace("\\", "/").split("/")
    # Drop the filename (last part)
    dirs = parts[:-1]
    if not dirs:
        return "."
    return "/".join(dirs[:depth])


def compute_file_structure_health(
    risk_profiles: list[dict[str, Any]],
    dir_depth: int = 2,
) -> list[HealthEntry]:
    """
    Group risk profiles by directory prefix and compute a health score per dir.

    health_score = max(0, round(100 - avg_pain_in_dir))

    The result is sorted ascending by health_score so the most critical
    directories surface first.
    """
    dir_pain: defaultdict[str, list[float]] = defaultdict(list)
    dir_files: defaultdict[str, set[str]] = defaultdict(set)

    for profile in risk_profiles:
        file_path: str = profile.get("file", "unknown")
        pain: float = float(profile.get("pain_score", 0.0))
        prefix = _dir_prefix(file_path, depth=dir_depth)
        dir_pain[prefix].append(pain)
        dir_files[prefix].add(file_path)

    entries: list[HealthEntry] = []
    for prefix, pains in dir_pain.items():
        avg_pain = sum(pains) / len(pains)
        health_score = max(0, round(100 - avg_pain))
        entries.append(
            HealthEntry(
                path=prefix,
                health_score=health_score,
                file_count=len(dir_files[prefix]),
                status=_health_status(health_score),
            )
        )

    entries.sort(key=lambda e: e.health_score)
    return entries


# ---------------------------------------------------------------------------
# Quick-wins extraction (deterministic — derived from L1 findings)
# ---------------------------------------------------------------------------

_IMPACT_MAP = {
    "critical": "High",
    "high": "High",
    "medium": "Medium",
    "low": "Low",
}


def extract_quick_wins(
    findings: list[dict[str, Any]],
    commit_hash: str,
    analysis_version: str = "v1.0.0",
    max_wins: int = 10,
) -> list[QuickWin]:
    """
    Derive quick-win items from the top findings, sorted by severity.
    """
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

    sorted_findings = sorted(
        findings,
        key=lambda f: (
            severity_order.get(normalise_severity(f.get("severity", "low")), 4),
            -float(f.get("confidence", 0.0)),
        ),
    )

    wins: list[QuickWin] = []
    for idx, finding in enumerate(sorted_findings[:max_wins], start=1):
        file_path: str = finding.get("file", "unknown")
        line: int = finding.get("line", 0)
        severity: str = normalise_severity(finding.get("severity", "low"))

        wins.append(
            QuickWin(
                id=f"win_{idx}",
                title=finding.get("title") or finding.get("type") or "Code Issue",
                description=(
                    f"{file_path}:{line} — "
                    + (finding.get("title") or finding.get("type") or "")
                ),
                impact=_IMPACT_MAP.get(severity, "Low"),
                type=finding.get("kind", "general"),
                metadata={
                    "detected_by": "deterministic_analyzer",
                    "confidence": finding.get("confidence", 0.0),
                    "source_files": [file_path],
                    "commit_hash": commit_hash,
                    "analysis_version": analysis_version,
                    "rule_id": finding.get("id", "UNKNOWN"),
                },
            )
        )
    return wins


# ---------------------------------------------------------------------------
# Security audit extraction (deterministic — derived from L1 SecurityInfo)
# ---------------------------------------------------------------------------

def extract_security_audit(security_info: dict[str, Any]) -> SecurityAudit:
    """
    Convert L1 SecurityInfo dict to a SecurityAudit Pydantic model.
    """
    raw_vulns: list[dict[str, Any]] = security_info.get("findings", [])
    vulns: list[SecurityVulnerability] = []
    for idx, v in enumerate(raw_vulns, start=1):
        file_path = v.get("file", "unknown")
        line = v.get("line", 0)
        vulns.append(
            SecurityVulnerability(
                id=f"vuln_{idx}",
                title=v.get("type", "Security Issue"),
                location=f"{file_path}:{line}",
                severity=normalise_severity(v.get("severity", "low")),
                description=v.get("description", ""),
                remediation="",
                cve=None,
            )
        )

    return SecurityAudit(
        critical_count=int(security_info.get("critical_count", 0)),
        high_count=int(security_info.get("high_count", 0)),
        medium_count=int(security_info.get("medium_count", 0)),
        low_count=int(security_info.get("low_count", 0)),
        vulnerabilities=vulns,
    )


# ---------------------------------------------------------------------------
# Refactor suggestion scaffolding (deterministic metadata, LLM fills the rest)
# ---------------------------------------------------------------------------

def make_suggestion_id(index: int, file_path: str) -> str:
    """Generate a stable, readable suggestion ID."""
    digest = hashlib.sha1(file_path.encode()).hexdigest()[:4].upper()
    return f"REF-{digest}-{index:03d}"


def _derive_rule_id(smell: dict[str, Any]) -> str:
    """Map a code smell to a rule ID from its tags or type."""
    tags: list[str] = smell.get("tags", [])
    for tag in tags:
        if re.match(r"[A-Z]+-[A-Z]+-\d+", tag):
            return tag
    smell_type: str = smell.get("type", "UNKNOWN").upper().replace("-", "_")
    return f"SMELL-{smell_type[:20]}"


def build_refactor_scaffold(
    smell: dict[str, Any],
    index: int,
    commit_hash: str,
    llm_meta: dict[str, Any],
) -> dict[str, Any]:
    """
    Return the deterministic skeleton for a RefactorSuggestion.
    The LLM will fill: title, confidence, reasoning, diff.
    """
    file_path: str = smell.get("file", "unknown")
    line: int = smell.get("line", 0)
    symbol: str = smell.get("symbol", "unknown")
    severity: str = normalise_severity(smell.get("severity", "low"))

    return {
        "suggestion_id": make_suggestion_id(index, file_path),
        "source": {
            "file": file_path,
            "start_line": line,
            "end_line": line,  # refined by LLM if context available
            "commit_hash": commit_hash,
            "symbol": symbol,
        },
        "analysis": {
            "rule": _derive_rule_id(smell),
            "detected_by": "llm_refactor_agent",
            "confidence": 0.0,  # replaced by LLM output
            "severity": severity,
        },
        "llm": llm_meta,
    }


# ---------------------------------------------------------------------------
# Radar metric base calculation (logic side of hybrid scores)
# ---------------------------------------------------------------------------

def compute_base_radar_scores(
    risk_summary: dict[str, Any],
    metrics_info: dict[str, Any],
    issue_kind_totals: dict[str, int],
    total_findings: int,
) -> dict[str, float]:
    """
    Compute heuristic base scores for each radar dimension.
    The Quantifier LLM uses these as starting-point context.

    Returns a dict keyed by dimension name with values in [0, 100].
    """

    def _safe_ratio(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0

    avg_pain: float = float(risk_summary.get("average_pain", 50.0))

    # Maintainability — inverse of average pain
    maintainability = max(0.0, 100.0 - avg_pain)

    # Security — penalty per security finding
    security_issues = issue_kind_totals.get("security", 0)
    security = max(0.0, 100.0 - security_issues * 8)

    # Performance — penalty per performance finding
    perf_issues = issue_kind_totals.get("performance", 0)
    performance = max(0.0, 100.0 - perf_issues * 6)

    # Test Coverage — from MetricsInfo.test_coverage_estimate (0.0–1.0)
    test_coverage = float(metrics_info.get("test_coverage_estimate", 0.5)) * 100

    # Architecture — penalty for high-complexity ratio
    total_functions = int(metrics_info.get("total_functions", 1))
    high_complexity = int(metrics_info.get("high_complexity_functions", 0))
    arch_penalty = _safe_ratio(high_complexity, total_functions) * 100
    architecture = max(0.0, 100.0 - arch_penalty)

    # Scalability — rough composite: architecture + no dependency issues
    scalability = (architecture * 0.6) + (maintainability * 0.4)

    return {
        "Maintainability": round(maintainability, 1),
        "Scalability": round(min(100.0, scalability), 1),
        "Security": round(min(100.0, security), 1),
        "Performance": round(min(100.0, performance), 1),
        "Test Coverage": round(test_coverage, 1),
        "Architecture": round(architecture, 1),
    }


# ---------------------------------------------------------------------------
# Deterministic extractor functions for DeterministicReport sub-sections
# ---------------------------------------------------------------------------

def extract_repository_summary(audit_result: dict[str, Any]) -> RepositorySummary:
    """Build RepositorySummary from RepositoryInfo + ArchitectureInfo in L1."""
    repo: dict[str, Any] = audit_result.get("repository", {})
    arch: dict[str, Any] = audit_result.get("architecture", {})
    return RepositorySummary(
        name=repo.get("name", "unknown"),
        languages=repo.get("language", []),
        frameworks=repo.get("frameworks", []),
        commit_hash=repo.get("commit_hash", "unknown"),
        branch=repo.get("branch", "unknown"),
        file_count=int(repo.get("file_count", 0)),
        detected_pattern=arch.get("detected_pattern", "unknown"),
    )


def extract_analysis_metrics(audit_result: dict[str, Any]) -> AnalysisMetrics:
    """Build AnalysisMetrics from MetricsInfo in L1."""
    m: dict[str, Any] = audit_result.get("metrics", {})
    return AnalysisMetrics(
        total_functions=int(m.get("total_functions", 0)),
        avg_function_length=float(m.get("avg_function_length", 0.0)),
        high_complexity_functions=int(m.get("high_complexity_functions", 0)),
        test_coverage_estimate=float(m.get("test_coverage_estimate", 0.0)),
        todo_count=int(m.get("todo_count", 0)),
    )


def extract_finding_summary(
    audit_result: dict[str, Any],
    risk_summary: dict[str, Any],
) -> FindingSummary:
    """Aggregate finding and smell counts by severity and kind from L1."""
    findings: list[dict[str, Any]] = audit_result.get("findings", [])
    smells: list[dict[str, Any]] = audit_result.get("code_smells", [])

    by_severity: dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    by_kind: dict[str, int] = {}

    for item in findings + smells:
        sev = normalise_severity(item.get("severity", "low"))
        by_severity[sev] = by_severity.get(sev, 0) + 1
        kind: str = item.get("kind", "general")
        by_kind[kind] = by_kind.get(kind, 0) + 1

    return FindingSummary(
        total_findings=int(risk_summary.get("total_findings", len(findings))),
        total_smells=int(risk_summary.get("total_smells", len(smells))),
        by_severity=by_severity,
        by_kind=by_kind,
    )


def extract_top_risky_entities(risk_report: dict[str, Any]) -> list[RiskyEntityRef]:
    """Pull top-5 risky entities from RiskReport.summary, enriched with profile data."""
    top: list[dict[str, Any]] = risk_report.get("summary", {}).get("top_risky_entities", [])
    profiles_by_id: dict[str, dict] = {
        p["entity_id"]: p for p in risk_report.get("profiles", []) if "entity_id" in p
    }
    result: list[RiskyEntityRef] = []
    for ref in top:
        entity_id: str = ref.get("entity_id", "")
        profile: dict[str, Any] = profiles_by_id.get(entity_id, {})
        result.append(RiskyEntityRef(
            rank=int(ref.get("rank", 0)),
            entity_id=entity_id,
            name=profile.get("name", entity_id),
            file=profile.get("file", "unknown"),
            pain_score=float(ref.get("pain_score", 0.0)),
        ))
    return result


def extract_cluster_summary(correlation_report: dict[str, Any]) -> ClusterSummaryBlock:
    """Build ClusterSummaryBlock from CorrelationReport.summary."""
    summary: dict[str, Any] = correlation_report.get("summary", {})
    top_clusters: list[dict[str, Any]] = summary.get("top_clusters", [])
    cluster_refs = [
        ClusterRef(
            rank=int(c.get("rank", 0)),
            cluster_id=c.get("cluster_id", ""),
            anchor_file=c.get("anchor_file", ""),
            total_pain=float(c.get("total_pain", 0.0)),
            members=int(c.get("members", 0)),
            is_hotspot=bool(c.get("is_hotspot", False)),
        )
        for c in top_clusters
    ]
    return ClusterSummaryBlock(
        total_clusters=int(summary.get("total_clusters", 0)),
        hotspot_clusters=int(summary.get("hotspot_clusters", 0)),
        total_clustered=int(summary.get("total_clustered", 0)),
        total_pain_in_clusters=float(summary.get("total_pain_in_clusters", 0.0)),
        hotspot_files=summary.get("hotspot_files", [])[:10],
        top_clusters=cluster_refs,
    )


def extract_dependency_summary(audit_result: dict[str, Any]) -> DependencySummary:
    """Build DependencySummary from RelationshipsInfo + DependenciesInfo in L1."""
    relationships: dict[str, Any] = audit_result.get("relationships", {})
    dependencies: dict[str, Any] = audit_result.get("dependencies", {})

    cycles: list[list[str]] = relationships.get("cycles", [])
    cycle_files: list[str] = list({f for cycle in cycles for f in cycle})

    outdated: list[dict[str, str]] = [
        {
            "name": p.get("name", ""),
            "current": p.get("current", ""),
            "recommended": p.get("recommended", ""),
        }
        for p in dependencies.get("outdated_packages", [])
    ]
    unused: list[str] = [
        d.get("name", "") for d in dependencies.get("unused_dependencies", [])
    ]

    return DependencySummary(
        coupling_score=float(relationships.get("coupling_score", 0.0)),
        cycles_count=len(cycles),
        cycle_files=cycle_files,
        outdated_packages=outdated,
        unused_dependencies=unused,
    )
