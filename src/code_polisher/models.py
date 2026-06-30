"""Pydantic models for tasks, results, and configuration validation."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskPriority(str, Enum):
    CRITICAL = "critical"  # security, bugs
    HIGH = "high"  # tech debt impacting maintainability
    MEDIUM = "medium"
    LOW = "low"  # style, minor cleanup


class AtomicTask(BaseModel):
    """Single atomic refactoring task. Never large refactors."""

    task_id: str = Field(..., description="Unique ID e.g. task-001")
    description: str = Field(..., min_length=10, description="Precise, actionable description of the change")
    file_path: str = Field(..., description="Relative path from repo root to target file")
    priority: TaskPriority = TaskPriority.MEDIUM
    rationale: str = Field(..., description="Why this improves the code (DRY, SOLID, perf, security, readability)")
    estimated_effort: Literal["trivial", "small", "medium"] = "small"
    requires_new_test: bool = Field(default=True, description="If true, Refactor Agent must write/verify test first")
    status: TaskStatus = TaskStatus.PENDING

    @field_validator("file_path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        if v.startswith("/") or ".." in v:
            raise ValueError("file_path must be relative and safe (no .. or absolute)")
        return v


class TaskList(BaseModel):
    """Output from Analyzer Agent."""

    tasks: list[AtomicTask] = Field(default_factory=list)
    summary: str = Field(..., description="High-level overview of technical debt found and refactoring strategy")
    total_estimated_tokens: int = 0
    analyzer_model: str = "grok-4.20"


class VerificationResult(BaseModel):
    """Output from Verifier Agent."""

    approved: bool
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    complexity_delta: float = 0.0  # positive = worse
    security_flags: list[str] = Field(default_factory=list)
    solid_dry_score: float = Field(ge=0, le=10, default=8.0)
    final_recommendation: str


class RefactorResult(BaseModel):
    """Result after one task execution."""

    task_id: str
    success: bool
    test_passed: bool
    lint_passed: bool
    diff: str = ""
    error_message: str | None = None
    worktree_path: str | None = None
    retries: int = 0


class OrchestratorConfig(BaseModel):
    """Runtime configuration loaded from env + CLI overrides."""

    xai_api_key: str = Field(default_factory=lambda: __import__("os").environ.get("XAI_API_KEY", ""))
    model_analyzer: str = "grok-4.20"  # 2M context for full codebase scan
    model_refactor: str = "grok-4.3"
    model_verifier: str = "grok-4.3"
    max_tasks: int = Field(default=30, ge=1, le=200)
    max_retries_per_task: int = 2
    lint_score_threshold: float = Field(default=8.5, ge=0, le=10)
    complexity_increase_tolerance: float = 0.5  # allow small increases for clarity sometimes
    worktree_base: str = "/tmp/code-polish-worktrees"
    timeout_seconds: int = 300
    dry_run: bool = False  # mock LLM for testing without key/cost
    target_dir: str = "."
    verbose: bool = False
    audit_enabled: bool = True  # Always-on production audit trail for traceability of all AI decisions and changes. New audit.py integrates here without breaking existing flows.

    @field_validator("xai_api_key")
    @classmethod
    def validate_key(cls, v: str, info: Any) -> str:
        if not info.data.get("dry_run") and not v:
            raise ValueError("XAI_API_KEY required unless --dry-run")
        return v
