"""
Release helper: copy the local `open_provence/modeling_open_provence_standalone.py`
into the four published HF model repos (README list) without touching git-lfs
artifacts, then commit and push.

Runbook (Nov 22, 2025):
1) Update the standalone file locally as needed.
2) Execute `python scripts/hf_utils/update_standalone.py`.
   - Clones / pulls into `tmp/release_models/<model-id>` with
     `GIT_LFS_SKIP_SMUDGE=1` to avoid LFS downloads.
   - Copies the standalone file, commits with a dated message, and pushes.
3) Verify: `git -C tmp/release_models/<model-id> log -1 --oneline`
   should show `chore: update standalone file (<date>)`.
4) Optional: run `python scripts/hf_utils/hf_model_process_check.py`
   to smoke-test the pushed code via AutoModel.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

DEFAULT_MODELS: tuple[str, ...] = (
    "hotchpotch/open-provence-reranker-v1",
    "hotchpotch/open-provence-reranker-xsmall-v1",
    "hotchpotch/open-provence-reranker-large-v1",
    "hotchpotch/open-provence-reranker-v1-gte-modernbert-base",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
STANDALONE_SRC = REPO_ROOT / "open_provence" / "modeling_open_provence_standalone.py"


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    merged_env = os.environ.copy()
    merged_env.update(env or {})
    print(f"[cmd] {' '.join(cmd)} (cwd={cwd})")
    subprocess.run(cmd, cwd=cwd, env=merged_env, check=True)


def ensure_repo(repo_id: str, base_dir: Path, env: dict[str, str]) -> Path:
    target_dir = base_dir / repo_id.split("/", maxsplit=1)[1]
    if not target_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)
        run(
            ["git", "clone", f"https://huggingface.co/{repo_id}", str(target_dir)],
            env=env,
        )
    else:
        run(["git", "-C", str(target_dir), "pull", "--rebase"], env=env)
    return target_dir


def copy_standalone(dest_repo: Path) -> None:
    dest = dest_repo / "modeling_open_provence_standalone.py"
    shutil.copy2(STANDALONE_SRC, dest)
    print(f"[copy] {STANDALONE_SRC} -> {dest}")


def has_changes(repo_dir: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() != ""


def commit_and_push(repo_dir: Path, message: str, env: dict[str, str]) -> None:
    run(["git", "-C", str(repo_dir), "add", "modeling_open_provence_standalone.py"], env=env)
    if not has_changes(repo_dir):
        print("[skip] No changes to commit.")
        return
    run(["git", "-C", str(repo_dir), "commit", "-m", message], env=env)
    run(["git", "-C", str(repo_dir), "push"], env=env)


def update_models(models: Iterable[str], base_dir: Path, commit_message: str) -> None:
    git_env = {"GIT_LFS_SKIP_SMUDGE": "1"}
    for repo_id in models:
        print(f"\n=== Updating {repo_id} ===")
        repo_dir = ensure_repo(repo_id, base_dir, git_env)
        copy_standalone(repo_dir)
        commit_and_push(repo_dir, commit_message, git_env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync modeling_open_provence_standalone.py into HF model repos without git-lfs.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Hugging Face model IDs to update (defaults to the four models in README.md).",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("tmp/release_models"),
        help="Local directory for cloning HF model repos.",
    )
    parser.add_argument(
        "--message",
        default=f"chore: update standalone file ({datetime.now().date().isoformat()})",
        help="Git commit message to use for pushes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    update_models(models=args.models, base_dir=args.base_dir, commit_message=args.message)


if __name__ == "__main__":
    main()
