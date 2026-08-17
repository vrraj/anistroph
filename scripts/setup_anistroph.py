#!/usr/bin/env python3
"""One-shot Anistroph setup.

Runs the full first-time setup pipeline:
  1. Check macOS libomp dependency (required by XGBoost)
  2. Create a virtualenv at ./.venv
  3. Install the package in editable mode (pip install -e .)
  4. Generate + register all reference datasets (delegates to setup_datasets.py)
  5. Print the ready-to-paste Claude Desktop MCP config with absolute paths

After this script completes, the only remaining manual step is adding the
printed MCP config block to Claude Desktop.

Usage:
    python scripts/setup_anistroph.py
    python scripts/setup_anistroph.py --skip-venv     # assume .venv already exists
    python scripts/setup_anistroph.py --force          # re-register datasets even if present

Exit codes:
    0  success
    1  a step failed
    2  libomp missing and user declined to install it
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = REPO_ROOT / ".venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def step(n: int, total: int, msg: str) -> None:
    print(f"\n[{n}/{total}] {msg}")
    print("-" * 60)


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> int:
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if check and result.returncode != 0:
        print(f"  FAILED (exit {result.returncode}): {' '.join(cmd)}")
        sys.exit(1)
    return result.returncode


def venv_exists() -> bool:
    return VENV_PYTHON.exists()


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def check_libomp() -> None:
    """On macOS, verify libomp is installed (XGBoost dependency)."""
    if platform.system() != "Darwin":
        return
    # Check if libomp is findable via otool
    probe = subprocess.run(
        ["otool", "-L", "/usr/local/lib/libomp.dylib"],
        capture_output=True,
    )
    if probe.returncode == 0:
        print("  libomp already installed.")
        return
    # Also check the Homebrew opt path (Apple Silicon)
    brew_lib = Path("/opt/homebrew/lib/libomp.dylib")
    if brew_lib.exists():
        print("  libomp already installed (Homebrew /opt/homebrew).")
        return

    print("  libomp (required by XGBoost) was not found.")
    if shutil.which("brew") is None:
        print("  Homebrew is not installed. Please install libomp manually,")
        print("  then re-run this script.")
        sys.exit(2)
    answer = input("  Install libomp via `brew install libomp`? [y/N] ").strip().lower()
    if answer != "y":
        print("  Skipped. XGBoost will fail to load until libomp is installed.")
        print("  Install it manually with: brew install libomp")
        sys.exit(2)
    run(["brew", "install", "libomp"])


def create_venv(skip_venv: bool) -> None:
    if skip_venv or venv_exists():
        if venv_exists():
            print(f"  Virtualenv already exists at {VENV_DIR}")
        else:
            print("  --skip-venv given but no .venv found; creating anyway.")
            run([sys.executable, "-m", "venv", str(VENV_DIR)])
        return
    run([sys.executable, "-m", "venv", str(VENV_DIR)])
    print(f"  Created virtualenv at {VENV_DIR}")


def install_package() -> None:
    run([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(VENV_PYTHON), "-m", "pip", "install", "-e", "."], cwd=REPO_ROOT)
    print("  Package installed in editable mode.")


def setup_datasets(force: bool) -> None:
    cmd = [str(VENV_PYTHON), "scripts/setup_datasets.py"]
    if force:
        cmd.append("--force")
    run(cmd, cwd=REPO_ROOT)
    print("  Datasets generated and registered.")


def print_claude_config() -> None:
    config = {
        "mcpServers": {
            "anistroph": {
                "command": str(VENV_PYTHON),
                "args": ["-m", "backend.integrations.mcp.server"],
                "cwd": str(REPO_ROOT),
            }
        }
    }
    print("\n" + "=" * 60)
    print("  SETUP COMPLETE — next step: Claude Desktop config")
    print("=" * 60)
    print()
    print("Add the following to:")
    print("  ~/Library/Application Support/Claude/claude_desktop_config.json")
    print()
    print(json.dumps(config, indent=2))
    print()
    print("Then fully quit Claude Desktop (Cmd+Q), reopen, and start a new")
    print("conversation. Verify Anistroph tools appear (hammer/tools icon).")
    print()
    print("Optional — start the Web UI / REST / MCP HTTP server:")
    print(f"  cd {REPO_ROOT} && make start-native")
    print("  Web UI:     http://localhost:9500")
    print("  MCP (HTTP): http://localhost:9500/mcp")
    print()
    print("  (Claude Desktop stdio does NOT require the server to be running.)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="One-shot Anistroph setup.")
    parser.add_argument("--skip-venv", action="store_true", help="Assume .venv already exists")
    parser.add_argument("--force", action="store_true", help="Re-register datasets even if present")
    args = parser.parse_args()

    total_steps = 5
    print("Anistroph one-shot setup")
    print(f"  Repo root: {REPO_ROOT}")
    print(f"  Platform:  {platform.system()} {platform.release()}")

    step(1, total_steps, "Checking macOS libomp dependency (XGBoost)")
    check_libomp()

    step(2, total_steps, "Creating virtualenv")
    create_venv(skip_venv=args.skip_venv)

    step(3, total_steps, "Installing Anistroph package (editable)")
    install_package()

    step(4, total_steps, "Generating + registering reference datasets")
    setup_datasets(force=args.force)

    step(5, total_steps, "Preparing Claude Desktop MCP config")
    print_claude_config()

    print("\nDone.\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
