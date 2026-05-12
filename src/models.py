"""
Pydantic models for the LangGraph audit workflow.

Field annotations guide node ownership:
  [deterministic]  — populated by Ingestor or Mapmaker (pure logic, no LLM)
  [probabilistic]  — populated by an LLM node (Architect, Quantifier, Refactor)
  [hybrid]         — deterministic base; LLM may enrich or override
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Layer wrappers (thin typed containers for devaudt engine outputs)
# ---------------------------------------------------------------------------

class LayerData(BaseModel):
    """Holds raw engine outputs from Layers 1-4. All [deterministic]."""

    audit_result: Optional[dict[str, Any]] = Field(
        default=None,
        description="[deterministic] Serialised AnalysisResult from Layer 1",
    )
    risk_report: Optional[dict[str, Any]] = Field(
        default=None,
        description="[deterministic] Serialised RiskReport from Layer 2",
    )
    correlation_report: Optional[dict[str, Any]] = Field(
        default=None,
        description="[deterministic] Serialised CorrelationReport from Layer 3",
    )
    context_packet: Optional[dict[str, Any]] = Field(
        default=None,
        description="[deterministic] Serialised ContextPacket from Layer 4",
    )


# ---------------------------------------------------------------------------
# LLM structured-output models (one per probabilistic agent)
# ---------------------------------------------------------------------------

class ArchitectOutput(BaseModel):
    """Output schema for the Architect node. All [probabilistic]."""

    ai_reasoning: str = Field(
        description="2-3 sentence architectural quality assessment and identification of systemic issues",
    )
    overall_verdict: str = Field(
        description="One of: 'Production Ready' | 'Ready with Caution' | 'Needs Review' | 'Critical Issues'",
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="LLM confidence in the verdict (0.0–1.0)",
    )
    architectural_recommendations: list[str] = Field(
        default_factory=list,
        description="3-5 actionable structural improvement recommendations",
    )


class RadarMetric(BaseModel):
    """A single radar chart data point. [hybrid]."""

    subject: str = Field(description="Metric name, e.g. 'Security'")
    score: int = Field(ge=0, le=100, description="[hybrid] Computed score")
    max: int = Field(default=100, description="[deterministic] Always 100")


class QuantifierOutput(BaseModel):
    """Output schema for the Quantifier node."""

    radar_metrics: list[RadarMetric] = Field(
        description="[hybrid] Radar chart data — 6 dimensions: Maintainability, Scalability, Security, Performance, Test Coverage, Architecture",
    )
    overall_score: int = Field(
        ge=0, le=100,
        description="[hybrid] Composite repo health score (0-100)",
    )
    technical_debt_score: int = Field(
        ge=0, le=100,
        description="[probabilistic] Estimated technical debt level (0 = no debt, 100 = extreme debt)",
    )
    growth_points: int = Field(
        ge=0, le=10,
        description="[probabilistic] Number of high-ROI improvements identified (0-10)",
    )


class HealthEntry(BaseModel):
    """File structure health entry for a directory. [deterministic]."""

    path: str = Field(description="[deterministic] Directory path relative to repo root")
    health_score: int = Field(ge=0, le=100, description="[deterministic] Aggregated health score")
    file_count: int = Field(ge=0, description="[deterministic] Number of files in directory")
    status: str = Field(description="[deterministic] 'excellent' | 'good' | 'warning' | 'critical'")


class MapmakerOutput(BaseModel):
    """Output schema for the Mapmaker node. All [deterministic]."""

    file_structure_health: list[HealthEntry] = Field(
        default_factory=list,
        description="[deterministic] Per-directory health breakdown",
    )


class RefactorSource(BaseModel):
    """Deterministic source locator for a refactor suggestion."""

    file: str = Field(description="[deterministic] Source file path")
    start_line: int = Field(description="[deterministic] First line of the smell")
    end_line: int = Field(description="[deterministic] Last line of the smell")
    commit_hash: str = Field(description="[deterministic] Commit SHA at time of analysis")
    symbol: str = Field(description="[deterministic] Function/class/method name")


class RefactorAnalysis(BaseModel):
    """Deterministic analysis metadata for a refactor suggestion."""

    rule: str = Field(description="[deterministic] Rule ID, e.g. 'SOLID-SRP-001'")
    detected_by: str = Field(
        default="llm_refactor_agent",
        description="[deterministic] Agent that produced this suggestion",
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="[probabilistic] LLM confidence in the suggestion",
    )
    severity: str = Field(description="[deterministic] low | medium | high | critical")


class RefactorLLMMeta(BaseModel):
    """LLM provenance metadata. [deterministic]."""

    provider: str = Field(default="")
    model: str = Field(default="")
    prompt_version: str = Field(default="prompt-refactor-v1")


class RefactorDiff(BaseModel):
    """Before/after code diff. [probabilistic]."""

    before: str = Field(description="[probabilistic] Code before refactoring")
    after: str = Field(description="[probabilistic] Code after refactoring")


class RefactorSuggestion(BaseModel):
    """A single refactor suggestion. Mix of deterministic + probabilistic."""

    title: str = Field(description="[probabilistic] Short title of the code smell")
    suggestion_id: str = Field(description="[deterministic] Unique ID, e.g. 'REF-001'")
    source: RefactorSource = Field(description="[deterministic] Source location")
    analysis: RefactorAnalysis = Field(description="[hybrid] Rule info + LLM confidence")
    llm: RefactorLLMMeta = Field(description="[deterministic] LLM provenance")
    reasoning: str = Field(description="[probabilistic] LLM explanation of the issue and fix")
    diff: RefactorDiff = Field(description="[probabilistic] Before/after code diff")


class RefactorLLMOutput(BaseModel):
    """Structured output the LLM must produce per smell. [probabilistic]."""

    title: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    diff: RefactorDiff


class RefactorOutput(BaseModel):
    """Output schema for the Refactor node."""

    refactor_suggestions: list[RefactorSuggestion] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Final report schemas — deterministic
# ---------------------------------------------------------------------------

class QuickWin(BaseModel):
    """High-impact, low-effort fix derived from deterministic findings. [deterministic]."""

    id: str
    title: str
    description: str
    impact: str  # "High" | "Medium" | "Low"
    type: str    # "architectural" | "security" | "performance" | ...
    metadata: dict[str, Any] = Field(default_factory=dict)


class SecurityVulnerability(BaseModel):
    """Individual security vulnerability. [deterministic]."""

    id: str
    title: str
    location: str  # "file:line"
    severity: str
    description: str
    remediation: str = ""
    cve: Optional[str] = None


class SecurityAudit(BaseModel):
    """Security section of the final report. [deterministic]."""

    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    vulnerabilities: list[SecurityVulnerability] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Deterministic report sub-models (sourced from L1 / L2 / L3 engines)
# ---------------------------------------------------------------------------

class RepositorySummary(BaseModel):
    """Repo identity and language metadata from RepositoryInfo + ArchitectureInfo. [deterministic]"""

    name: str = "unknown"
    languages: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    commit_hash: str = "unknown"
    branch: str = "unknown"
    file_count: int = 0
    detected_pattern: str = "unknown"


class AnalysisMetrics(BaseModel):
    """Code-level metrics from MetricsInfo. [deterministic]"""

    total_functions: int = 0
    avg_function_length: float = 0.0
    high_complexity_functions: int = 0
    test_coverage_estimate: float = 0.0
    todo_count: int = 0


class FindingSummary(BaseModel):
    """Aggregated finding/smell counts from L1. [deterministic]"""

    total_findings: int = 0
    total_smells: int = 0
    by_severity: dict[str, int] = Field(default_factory=dict)
    by_kind: dict[str, int] = Field(default_factory=dict)


class RiskyEntityRef(BaseModel):
    """Top-risky entity reference from RiskReport.summary. [deterministic]"""

    rank: int
    entity_id: str
    name: str
    file: str
    pain_score: float


class ClusterRef(BaseModel):
    """Lightweight cluster reference from CorrelationReport.summary. [deterministic]"""

    rank: int
    cluster_id: str
    anchor_file: str
    total_pain: float
    members: int
    is_hotspot: bool


class ClusterSummaryBlock(BaseModel):
    """Cluster-level aggregates from CorrelationReport. [deterministic]"""

    total_clusters: int = 0
    hotspot_clusters: int = 0
    total_clustered: int = 0
    total_pain_in_clusters: float = 0.0
    hotspot_files: list[str] = Field(default_factory=list)
    top_clusters: list[ClusterRef] = Field(default_factory=list)


class DependencySummary(BaseModel):
    """Import graph and package health from RelationshipsInfo + DependenciesInfo. [deterministic]"""

    coupling_score: float = 0.0
    cycles_count: int = 0
    cycle_files: list[str] = Field(default_factory=list)
    outdated_packages: list[dict[str, str]] = Field(default_factory=list)
    unused_dependencies: list[str] = Field(default_factory=list)


class DeterministicReport(BaseModel):
    """
    Pure deterministic audit product — all fields sourced from L1/L2/L3 engines.
    Stable and reproducible across runs on the same commit. No LLM involved.
    """

    repository_summary: RepositorySummary = Field(default_factory=RepositorySummary)
    analysis_metrics: AnalysisMetrics = Field(default_factory=AnalysisMetrics)
    finding_summary: FindingSummary = Field(default_factory=FindingSummary)
    top_risky_entities: list[RiskyEntityRef] = Field(default_factory=list)
    cluster_summary: ClusterSummaryBlock = Field(default_factory=ClusterSummaryBlock)
    dependency_summary: DependencySummary = Field(default_factory=DependencySummary)
    file_structure_health: list[HealthEntry] = Field(default_factory=list)
    security_audit: SecurityAudit = Field(default_factory=SecurityAudit)
    quick_wins: list[QuickWin] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM insights sub-model (probabilistic / hybrid)
# ---------------------------------------------------------------------------

class LLMInsights(BaseModel):
    """
    AI-generated insights — populated by Architect, Quantifier, and Refactor nodes.
    Fields are probabilistic or hybrid; confidence reflects LLM certainty.
    """

    # From QuantifierOutput [hybrid]
    overall_score: int = Field(ge=0, le=100, default=50)
    technical_debt_score: int = Field(ge=0, le=100, default=50)
    growth_pts: int = Field(ge=0, le=10, default=0)
    radar_metrics: list[RadarMetric] = Field(default_factory=list)

    # From ArchitectOutput [probabilistic]
    overall_verdict: str = "Needs Review"
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    ai_reasoning: str = ""
    architectural_recommendations: list[str] = Field(default_factory=list)

    # From RefactorOutput [probabilistic] — omitted from payload when empty
    refactor_suggestions: list[RefactorSuggestion] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Final report — graph output (envelope injected by orchestrator)
# ---------------------------------------------------------------------------

class FinalAuditReport(BaseModel):
    """
    Root graph output payload. Envelope fields (job_id, repo_url, status, timestamp)
    are NOT part of this model — the orchestrator injects them at serialization time.
    """

    deterministic_report: DeterministicReport
    llm_insights: LLMInsights


# ---------------------------------------------------------------------------
# AuditState — root LangGraph state object
# ---------------------------------------------------------------------------

class AuditState(BaseModel):
    """
    Root state object for the LangGraph workflow.

    LangGraph uses TypedDict for state by default, but we subclass BaseModel
    for full Pydantic validation. The `model_config` sets allow arbitrary types
    so runtime devaudt engine objects can also be stored in the `_raw` fields.
    """

    model_config = {"arbitrary_types_allowed": True}

    # Inputs [deterministic] -------------------------------------------------
    repo_url: str = ""
    job_id: str = ""
    timestamp: str = ""

    # Layer data [deterministic] ---------------------------------------------
    layers: LayerData = Field(default_factory=LayerData)

    # Partial reports — populated by parallel workers ------------------------
    # Annotated with operator.add so LangGraph merges list updates correctly.
    architect_output: Optional[ArchitectOutput] = None        # [probabilistic]
    quantifier_output: Optional[QuantifierOutput] = None      # [hybrid]
    mapmaker_output: Optional[MapmakerOutput] = None          # [deterministic]
    refactor_output: Optional[RefactorOutput] = None          # [probabilistic]

    # Error tracking [deterministic] -----------------------------------------
    errors: Annotated[list[str], operator.add] = Field(default_factory=list)

    # Final output [hybrid] --------------------------------------------------
    final_report: Optional[FinalAuditReport] = None
