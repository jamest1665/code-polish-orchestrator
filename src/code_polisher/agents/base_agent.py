"""Base class for all agents providing common LLM client setup, tool dispatch loop, retry logic, and structured output handling."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Callable, Optional

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..models import OrchestratorConfig
from ..prompts import ANALYZER_SYSTEM_PROMPT, REFACTOR_SYSTEM_PROMPT, VERIFIER_SYSTEM_PROMPT
from ..tools import TOOL_MAP, get_tool_definitions, ToolError
from ..audit import AuditLogger  # New production audit module - additive, non-breaking integration for full traceability

logger = logging.getLogger(__name__)


class BaseAgent:
    """Shared infrastructure for Analyzer, Refactor, Verifier agents."""

    def __init__(self, config: OrchestratorConfig, role: str):
        self.config = config
        self.role = role
        self.client = AsyncOpenAI(
            api_key=config.xai_api_key,
            base_url="https://api.x.ai/v1",
        )
        self.model = {
            "analyzer": config.model_analyzer,
            "refactor": config.model_refactor,
            "verifier": config.model_verifier,
        }[role]
        self.system_prompt = {
            "analyzer": ANALYZER_SYSTEM_PROMPT,
            "refactor": REFACTOR_SYSTEM_PROMPT.format(max_retries=config.max_retries_per_task),
            "verifier": VERIFIER_SYSTEM_PROMPT,
        }[role]
        self.tools = get_tool_definitions()
        self.tool_map: dict[str, Callable] = TOOL_MAP
        self.max_tool_iterations = 12 if role == "refactor" else 8  # prevent runaway
        # Audit integration (new, works with existing dry_run/real paths, never affects core logic)
        self.auditor: Optional[AuditLogger] = None
        if getattr(config, "audit_enabled", True):
            try:
                self.auditor = AuditLogger(config.target_dir, enabled=True)
            except Exception:
                logger.warning("AuditLogger init failed; continuing without audit (non-fatal)")

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict]] = None,
        tool_choice: str | dict = "auto",
        temperature: float = 0.1,  # low for determinism in coding tasks
        max_tokens: int = 8000,
    ) -> Any:
        """Central async LLM call with xAI Grok via OpenAI compat layer."""
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.95,
            )
            msg = resp.choices[0].message
            if self.auditor:
                self.auditor.log_llm_call(
                    role=self.role,
                    model=self.model,
                    prompt_preview=str(messages[-1].get("content", "")) if messages else "",
                    tool_calls=[tc.function.name for tc in (msg.tool_calls or [])] if hasattr(msg, "tool_calls") else [],
                    response_preview=getattr(msg, "content", None) or "",
                )
            return msg
        except Exception as e:
            logger.error(f"LLM call failed for {self.role}: {e}")
            raise

    async def _execute_tool(self, tool_call: Any) -> str:
        """Execute a single tool call from LLM and return string result for message history."""
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            return f"ERROR: Invalid JSON arguments for tool {name}"

        if name not in self.tool_map:
            return f"ERROR: Unknown tool '{name}'. Available: {list(self.tool_map.keys())}"

        func = self.tool_map[name]
        try:
            # Inject worktree context if agent knows current one (set by orchestrator)
            if "worktree" in args and hasattr(self, "current_worktree"):
                args["worktree"] = self.current_worktree or args.get("worktree")

            if asyncio.iscoroutinefunction(func):
                result = await func(**args)
            else:
                result = await asyncio.to_thread(func, **args)
            preview = str(result)[:300] if result else ""
            success = not str(result).startswith(("ERROR", "TOOL_ERROR", "UNEXPECTED"))
            if self.auditor:
                self.auditor.log_tool_call(tool_name=name, args=args, result_preview=preview, success=success)
            return str(result)[:8000]  # cap result size
        except ToolError as e:
            if self.auditor:
                self.auditor.log_tool_call(tool_name=name, args=args, result_preview=str(e), success=False)
            return f"TOOL_ERROR: {e}"
        except Exception as e:
            logger.exception(f"Tool {name} crashed")
            if self.auditor:
                self.auditor.log_tool_call(tool_name=name, args=args, result_preview=str(e), success=False)
            return f"UNEXPECTED_ERROR in {name}: {e}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    async def run_agent_loop(
        self,
        user_prompt: str,
        initial_context: Optional[dict[str, Any]] = None,
        expect_json: bool = False,
        current_worktree: Optional[str] = None,
    ) -> str | dict[str, Any]:
        """Main ReAct-style loop: LLM -> tool calls -> results -> repeat until final answer.
        Returns final message content (text or parsed JSON if expect_json).
        """
        self.current_worktree = current_worktree
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
        ]
        if initial_context:
            messages.append({"role": "user", "content": f"CONTEXT:\n{json.dumps(initial_context, indent=2)[:4000]}"})

        messages.append({"role": "user", "content": user_prompt})

        for iteration in range(self.max_tool_iterations):
            logger.debug(f"{self.role} iteration {iteration + 1}")
            msg = await self._call_llm(
                messages=messages,
                tools=self.tools if self.role in ("analyzer", "refactor") else None,  # verifier mostly reads
                tool_choice="auto" if self.role != "verifier" else "none",
            )

            if msg.tool_calls:
                messages.append({"role": "assistant", "tool_calls": msg.tool_calls})
                for tc in msg.tool_calls:
                    result = await self._execute_tool(tc)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.function.name,
                            "content": result,
                        }
                    )
                continue  # loop for next decision

            # Final answer
            content = msg.content or ""
            if expect_json:
                # Try to extract JSON even if wrapped in ```json
                import re

                match = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", content)
                if match:
                    try:
                        return json.loads(match.group(0))
                    except json.JSONDecodeError:
                        pass
                # fallback: ask LLM to fix? but for simplicity return raw for orchestrator to handle
            return content

        raise RuntimeError(f"{self.role} exceeded max tool iterations without final answer")
