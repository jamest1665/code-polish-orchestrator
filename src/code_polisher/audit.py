"""Production audit logging for the Code Polish Orchestrator.
Records every significant agent decision, tool invocation, LLM interaction (sanitized), and outcome to JSONL files.
This provides full traceability for code changes made by AI - essential for production trust, debugging, and compliance.
Never deletes existing logs; appends only. Integrates seamlessly with existing logging and agents without side effects on real or dry-run paths.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AuditLogger:
    """Thread-safe-ish append-only JSONL auditor. One instance per run, persisted under target/.code-polish/audit/"""

    def __init__(self, target_dir: str | Path, enabled: bool = True):
        self.enabled = enabled
        self.target_dir = Path(target_dir).resolve()
        self.audit_dir = self.target_dir / ".code-polish" / "audit"
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.log_file = self.audit_dir / f"session-{self.session_id}.jsonl"
        if self.enabled:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
            # Touch file
            self.log_file.touch(exist_ok=True)
            self._write_event(
                "session_start",
                {
                    "session_id": self.session_id,
                    "target": str(self.target_dir),
                    "version": "0.1.0",
                },
            )
            logger.info(f"Audit logging enabled. Writing to {self.log_file}")

    def _write_event(self, event_type: str, data: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "session_id": self.session_id,
                **data,
            }
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Audit write failed for {event_type}: {e}")  # non-fatal

    def log_llm_call(
        self,
        role: str,
        model: str,
        prompt_preview: str,
        tool_calls: Optional[list[str]] = None,
        response_preview: Optional[str] = None,
        tokens_in: Optional[int] = None,
        tokens_out: Optional[int] = None,
    ) -> None:
        """Log LLM interaction (prompts/responses truncated for size/privacy)."""
        self._write_event(
            "llm_call",
            {
                "role": role,
                "model": model,
                "prompt_preview": prompt_preview[:500] if prompt_preview else None,
                "tool_calls": tool_calls or [],
                "response_preview": response_preview[:500] if response_preview else None,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
            },
        )

    def log_tool_call(self, tool_name: str, args: dict[str, Any], result_preview: str, success: bool) -> None:
        self._write_event(
            "tool_call",
            {
                "tool": tool_name,
                "args_preview": {k: str(v)[:100] for k, v in args.items()},
                "result_preview": result_preview[:300],
                "success": success,
            },
        )

    def log_task_event(self, task_id: str, event: str, details: Optional[dict[str, Any]] = None) -> None:
        self._write_event(
            "task_event",
            {
                "task_id": task_id,
                "event": event,
                "details": details or {},
            },
        )

    def log_decision(
        self, decision_type: str, approved: bool, rationale: str, task_id: Optional[str] = None
    ) -> None:
        self._write_event(
            "decision",
            {
                "type": decision_type,
                "approved": approved,
                "rationale_preview": rationale[:400],
                "task_id": task_id,
            },
        )

    def close(self) -> None:
        if self.enabled:
            self._write_event("session_end", {"status": "completed"})
            logger.info(f"Audit session closed: {self.log_file}")
