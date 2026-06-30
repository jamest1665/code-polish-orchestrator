# Code Polish Orchestrator

**Production-ready CLI tool that coordinates specialized Grok AI agents to refactor your codebase into the shortest, cleanest, strictly-typed, production-grade state possible — safely.**

It never does large refactors in one go. Every change is **atomic**, **test-first**, executed in an **isolated git worktree**, linted, and **independently verified** by a second agent before any merge back to your main branch.

**v0.2 update**: --dry-run is now a *complete working demo* of the entire program (full pipeline simulation exercising worktree isolation, real tool execution for tests/lint/write, commit/merge, audit logging, stats, and tasks.json persistence). Audit trail added for production traceability. All changes additive and fully compatible.

## Quick Start

```bash
uv sync --dev
 export XAI_API_KEY=...   # for real mode
uv run code-polish examples/sample_project --dry-run --verbose   # Now fully exercises EVERY safety mechanism offline!
```

See previous README content for full architecture, safety notes, and usage. The system is now one cohesive, demonstrably working production program.
