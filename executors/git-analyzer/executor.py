"""
Git Analyzer Executor
Handles repository cloning and git history analysis
"""
import asyncio
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from base import BaseExecutor

logger = logging.getLogger(__name__)


class GitAnalyzerExecutor(BaseExecutor):
    """Executor for git operations"""

    name = "git-analyzer"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.clone_dir = config.get("clone_dir", "/tmp/audit-clones") if config else "/tmp/audit-clones"
        self.timeout = config.get("timeout", 120) if config else 120

    async def run(self, action: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Route to appropriate method"""
        if action == "load":
            return await self.load(**inputs)
        elif action == "clone":
            return await self.clone(**inputs)
        elif action == "analyze_history":
            return await self.analyze_history(**inputs)
        else:
            raise ValueError(f"Unknown action: {action}")

    async def load(
        self,
        source_type: str,
        source_path: str,
        branch: str = "main",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Load source - universal entry point for Stage 1.

        Handles both:
        - git: Clone repository to local path
        - directory: Validate and prepare local directory

        Returns:
            local_path: Path to files for analysis
            source_info: Metadata about source
            is_git_repo: Whether it's a git repo
        """
        if source_type == "git":
            # Clone git repository
            clone_result = await self.clone(url=source_path, branch=branch)
            local_path = clone_result["repo_path"]

            # Get git history info
            history = await self.analyze_history(path=local_path)

            return {
                "local_path": local_path,
                "source_info": {
                    "type": "git",
                    "url": source_path,
                    "branch": branch,
                    "commit": clone_result.get("commit_hash"),
                    "commit_count": history.get("commit_count", 0),
                    "contributors": history.get("contributors", 0),
                    "last_commit": history.get("last_commit_date"),
                },
                "is_git_repo": True
            }

        elif source_type == "directory":
            # Local directory - validate it exists
            path = Path(source_path)

            if not path.exists():
                raise FileNotFoundError(f"Directory not found: {source_path}")

            if not path.is_dir():
                raise ValueError(f"Not a directory: {source_path}")

            # Check if it's a git repo
            is_git = (path / ".git").exists()

            source_info = {
                "type": "directory",
                "path": str(path.resolve()),
                "name": path.name,
            }

            # If it's a git repo, add git info
            if is_git:
                history = await self.analyze_history(path=str(path))
                source_info.update({
                    "commit_count": history.get("commit_count", 0),
                    "contributors": history.get("contributors", 0),
                    "last_commit": history.get("last_commit_date"),
                })

            return {
                "local_path": str(path.resolve()),
                "source_info": source_info,
                "is_git_repo": is_git
            }

        else:
            raise ValueError(f"Unknown source_type: {source_type}. Use 'git' or 'directory'")

    async def clone(self, url: str, branch: str = "main", **kwargs) -> Dict[str, Any]:
        """Clone a git repository"""
        # Create unique temp directory
        repo_name = url.split("/")[-1].replace(".git", "")
        repo_path = Path(self.clone_dir) / f"{repo_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        os.makedirs(repo_path.parent, exist_ok=True)

        try:
            # Clone with timeout
            proc = await asyncio.create_subprocess_exec(
                "git", "clone", "--depth", "100", "--branch", branch, url, str(repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)

            if proc.returncode != 0:
                raise Exception(f"Git clone failed: {stderr.decode()}")

            # Get commit hash
            proc = await asyncio.create_subprocess_exec(
                "git", "-C", str(repo_path), "rev-parse", "HEAD",
                stdout=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            commit_hash = stdout.decode().strip()

            return {
                "repo_path": str(repo_path),
                "commit_hash": commit_hash,
                "branch_info": branch,
                "cloned_at": datetime.now().isoformat()
            }

        except asyncio.TimeoutError:
            if repo_path.exists():
                shutil.rmtree(repo_path)
            raise Exception(f"Clone timed out after {self.timeout}s")

    async def analyze_history(self, path: str, **kwargs) -> Dict[str, Any]:
        """Analyze git history"""
        repo_path = Path(path)

        if not (repo_path / ".git").exists():
            return {"error": "Not a git repository"}

        # Get commit count
        proc = await asyncio.create_subprocess_exec(
            "git", "-C", str(repo_path), "rev-list", "--count", "HEAD",
            stdout=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        commit_count = int(stdout.decode().strip()) if stdout else 0

        # Get contributors
        proc = await asyncio.create_subprocess_exec(
            "git", "-C", str(repo_path), "shortlog", "-sn", "--all",
            stdout=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        contributors = len(stdout.decode().strip().split("\n")) if stdout else 0

        # Get last commit date
        proc = await asyncio.create_subprocess_exec(
            "git", "-C", str(repo_path), "log", "-1", "--format=%ci",
            stdout=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        last_commit = stdout.decode().strip() if stdout else ""

        # Calculate commit frequency (commits per week over last 3 months)
        three_months_ago = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        proc = await asyncio.create_subprocess_exec(
            "git", "-C", str(repo_path), "rev-list", "--count", f"--since={three_months_ago}", "HEAD",
            stdout=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        recent_commits = int(stdout.decode().strip()) if stdout else 0
        commit_frequency = round(recent_commits / 13, 2)  # commits per week

        # Get branch count
        proc = await asyncio.create_subprocess_exec(
            "git", "-C", str(repo_path), "branch", "-r",
            stdout=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        branch_count = len([b for b in stdout.decode().split("\n") if b.strip()]) if stdout else 1

        return {
            "commit_count": commit_count,
            "contributors": contributors,
            "last_commit_date": last_commit,
            "commit_frequency": commit_frequency,
            "branch_count": branch_count
        }

    def get_capabilities(self) -> list[str]:
        return ["load", "clone", "analyze_history"]


# Factory function
def create_executor(config: Dict[str, Any] = None) -> GitAnalyzerExecutor:
    return GitAnalyzerExecutor(config)
