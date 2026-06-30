"""CLI entrypoint using Typer. Production-grade with logging, validation, and helpful UX."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

from .models import OrchestratorConfig
from .orchestrator import CodePolishOrchestrator
from .config_loader import build_config  # New additive config loader - supports persistent project settings without breaking CLI or existing flows

app = typer.Typer(
    name="code-polish",
    help="Production CLI: Coordinate Grok AI agents to safely polish codebases atomically (test-first + worktree + verifier gates). Supports project config files.",
    add_completion=False,
    rich_markup_mode="markdown",
)

console = Console()


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_time=False)],
    )
    # quiet noisy libs
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


@app.command()
def polish(
    target: Path = typer.Argument(
        Path("."),
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Path to git repository root to polish.",
    ),
    max_tasks: int = typer.Option(25, "--max-tasks", "-n", min=1, max=100, help="Safety cap on number of atomic tasks."),
    model_analyzer: str = typer.Option("grok-4.20", help="Long-context model for full codebase analysis (2M tokens)."),
    model_refactor: str = typer.Option("grok-4.3", help="Primary model for precise, test-first refactoring."),
    model_verifier: str = typer.Option("grok-4.3", help="Model for independent quality/security review."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Use mock tasks + no LLM calls. Perfect for testing the pipeline on examples/sample_project."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
    resume: bool = typer.Option(True, help="Resume from code-polish-tasks.json if present (disable with --no-resume)."),
) -> None:
    """Run the full Code Polish Orchestrator on the target git repo.

    Project config (pyproject.toml [tool.code-polish] or .code-polish.toml) is loaded automatically.
    CLI flags override project settings.

    **Safety first**: Always commit or stash your work before running. Changes are isolated in git worktrees and only merged on full approval.
    """
    load_dotenv()
    setup_logging(verbose)

    if not dry_run and not Path(".env").exists() and "XAI_API_KEY" not in __import__("os").environ:
        console.print("[red]ERROR:[/red] XAI_API_KEY not found. Set it in environment or .env file, or use --dry-run for demo.")
        raise typer.Exit(1)

    # Build config: project file (pyproject.toml [tool.code-polish] or .code-polish.toml) + CLI overrides (CLI wins)
    # This is fully compatible with dry-run, simulation, audit, linter config, and all prior features.
    cli_overrides = {
        "max_tasks": max_tasks,
        "model_analyzer": model_analyzer,
        "model_refactor": model_refactor,
        "model_verifier": model_verifier,
        "dry_run": dry_run,
        "verbose": verbose,
    }
    config = build_config(target_dir=target.resolve(), cli_overrides=cli_overrides)

    orchestrator = CodePolishOrchestrator(config)

    try:
        asyncio.run(orchestrator.run())
        console.print("\n[bold green]✓ Orchestrator finished successfully.[/bold green]")
        console.print("Review 'code-polish-tasks.json' and git log for changes. Run your full test suite + linter manually as final human gate.")
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user. Partial progress saved in code-polish-tasks.json and worktree branches.[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"\n[red]FATAL ERROR:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
