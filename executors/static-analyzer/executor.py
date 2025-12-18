"""
Static Analyzer Executor
Analyzes code structure, dependencies, and quality
"""
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from base import BaseExecutor

logger = logging.getLogger(__name__)


class StaticAnalyzerExecutor(BaseExecutor):
    """Executor for static code analysis"""

    name = "static-analyzer"

    # Language detection patterns
    LANGUAGE_PATTERNS = {
        "Python": ["*.py", "requirements.txt", "setup.py", "pyproject.toml"],
        "JavaScript": ["*.js", "*.jsx", "package.json"],
        "TypeScript": ["*.ts", "*.tsx", "tsconfig.json"],
        "Go": ["*.go", "go.mod"],
        "Rust": ["*.rs", "Cargo.toml"],
        "Java": ["*.java", "pom.xml", "build.gradle"],
        "Ruby": ["*.rb", "Gemfile"],
        "PHP": ["*.php", "composer.json"],
    }

    async def run(self, action: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Route to appropriate method"""
        if action == "analyze_structure":
            return await self.analyze_structure(**inputs)
        elif action == "analyze_dependencies":
            return await self.analyze_dependencies(**inputs)
        elif action == "analyze_quality":
            return await self.analyze_quality(**inputs)
        else:
            raise ValueError(f"Unknown action: {action}")

    async def analyze_structure(self, path: str, **kwargs) -> Dict[str, Any]:
        """Analyze repository structure"""
        repo_path = Path(path)

        # Count files
        all_files = list(repo_path.rglob("*"))
        code_files = [f for f in all_files if f.is_file() and not self._is_ignored(f)]
        file_count = len(code_files)

        # Count lines of code
        loc = 0
        for f in code_files:
            try:
                if f.suffix in [".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".rb", ".php"]:
                    loc += sum(1 for _ in open(f, "r", errors="ignore"))
            except:
                pass

        # Detect languages
        languages = self._detect_languages(repo_path)

        # Check for common files
        has_readme = any((repo_path / name).exists() for name in ["README.md", "README.rst", "README.txt", "README"])
        has_license = any((repo_path / name).exists() for name in ["LICENSE", "LICENSE.md", "LICENSE.txt"])
        has_tests = any([
            (repo_path / "tests").exists(),
            (repo_path / "test").exists(),
            (repo_path / "__tests__").exists(),
            any(repo_path.rglob("*_test.py")),
            any(repo_path.rglob("test_*.py")),
            any(repo_path.rglob("*.test.js")),
            any(repo_path.rglob("*.spec.ts")),
        ])
        has_ci = any([
            (repo_path / ".github" / "workflows").exists(),
            (repo_path / ".gitlab-ci.yml").exists(),
            (repo_path / "Jenkinsfile").exists(),
            (repo_path / ".circleci").exists(),
        ])
        has_docker = any([
            (repo_path / "Dockerfile").exists(),
            (repo_path / "docker-compose.yml").exists(),
            (repo_path / "docker-compose.yaml").exists(),
        ])

        return {
            "file_count": file_count,
            "loc": loc,
            "languages": languages,
            "has_readme": has_readme,
            "has_license": has_license,
            "has_tests": has_tests,
            "has_ci": has_ci,
            "has_docker": has_docker
        }

    async def analyze_dependencies(self, path: str, languages: list = None, **kwargs) -> Dict[str, Any]:
        """Analyze project dependencies"""
        repo_path = Path(path)
        dependencies = []
        outdated_count = 0
        vulnerable_count = 0

        # Python
        req_file = repo_path / "requirements.txt"
        if req_file.exists():
            deps = self._parse_requirements(req_file)
            dependencies.extend(deps)

        # Node.js
        pkg_file = repo_path / "package.json"
        if pkg_file.exists():
            deps = self._parse_package_json(pkg_file)
            dependencies.extend(deps)

        # Go
        go_mod = repo_path / "go.mod"
        if go_mod.exists():
            deps = self._parse_go_mod(go_mod)
            dependencies.extend(deps)

        return {
            "dependencies": dependencies[:50],  # Limit for response size
            "total_count": len(dependencies),
            "outdated_count": outdated_count,  # Would need API calls to check
            "vulnerable_count": vulnerable_count  # Would need security DB
        }

    async def analyze_quality(self, path: str, languages: list = None, **kwargs) -> Dict[str, Any]:
        """Analyze code quality metrics"""
        repo_path = Path(path)

        # Try to run quality tools if available
        complexity_score = await self._calculate_complexity(repo_path)
        duplication = await self._calculate_duplication(repo_path)
        lint_issues = await self._run_linters(repo_path, languages or [])

        return {
            "complexity_score": complexity_score,
            "duplication_percent": duplication,
            "lint_issues": lint_issues,
            "lint_issues_per_kloc": round(lint_issues / max(1, (await self._count_loc(repo_path)) / 1000), 2)
        }

    def _is_ignored(self, path: Path) -> bool:
        """Check if path should be ignored"""
        ignored = [".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".next"]
        return any(part in path.parts for part in ignored)

    def _detect_languages(self, repo_path: Path) -> list[str]:
        """Detect programming languages in repo"""
        detected = []
        for lang, patterns in self.LANGUAGE_PATTERNS.items():
            for pattern in patterns:
                if list(repo_path.rglob(pattern)):
                    detected.append(lang)
                    break
        return detected or ["Unknown"]

    def _parse_requirements(self, path: Path) -> list[dict]:
        """Parse Python requirements.txt"""
        deps = []
        try:
            for line in open(path):
                line = line.strip()
                if line and not line.startswith("#"):
                    name = line.split("==")[0].split(">=")[0].split("<=")[0].split("[")[0]
                    deps.append({"name": name, "type": "python"})
        except:
            pass
        return deps

    def _parse_package_json(self, path: Path) -> list[dict]:
        """Parse Node.js package.json"""
        deps = []
        try:
            data = json.loads(open(path).read())
            for name in data.get("dependencies", {}):
                deps.append({"name": name, "type": "npm"})
            for name in data.get("devDependencies", {}):
                deps.append({"name": name, "type": "npm-dev"})
        except:
            pass
        return deps

    def _parse_go_mod(self, path: Path) -> list[dict]:
        """Parse Go go.mod"""
        deps = []
        try:
            in_require = False
            for line in open(path):
                if line.startswith("require"):
                    in_require = True
                elif in_require:
                    if line.strip() == ")":
                        in_require = False
                    elif line.strip():
                        name = line.strip().split()[0]
                        deps.append({"name": name, "type": "go"})
        except:
            pass
        return deps

    async def _calculate_complexity(self, repo_path: Path) -> float:
        """Calculate cyclomatic complexity"""
        # Try radon for Python
        try:
            proc = await asyncio.create_subprocess_exec(
                "radon", "cc", str(repo_path), "-a", "-j",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
            if stdout:
                data = json.loads(stdout.decode())
                # Extract average complexity
                total = 0
                count = 0
                for file_data in data.values():
                    for func in file_data:
                        if isinstance(func, dict) and "complexity" in func:
                            total += func["complexity"]
                            count += 1
                return round(total / max(count, 1), 2)
        except:
            pass
        return 15.0  # Default estimate

    async def _calculate_duplication(self, repo_path: Path) -> float:
        """Calculate code duplication percentage"""
        # Try jscpd
        try:
            proc = await asyncio.create_subprocess_exec(
                "jscpd", str(repo_path), "--reporters", "json", "--silent",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
            if stdout:
                data = json.loads(stdout.decode())
                return data.get("statistics", {}).get("percentage", 5.0)
        except:
            pass
        return 5.0  # Default estimate

    async def _run_linters(self, repo_path: Path, languages: list) -> int:
        """Run linters and count issues"""
        total_issues = 0

        # Python - try ruff or flake8
        if "Python" in languages:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "ruff", "check", str(repo_path), "--output-format", "json",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
                if stdout:
                    issues = json.loads(stdout.decode())
                    total_issues += len(issues)
            except:
                pass

        return total_issues

    async def _count_loc(self, repo_path: Path) -> int:
        """Count lines of code"""
        loc = 0
        for ext in [".py", ".js", ".ts", ".go", ".rs", ".java"]:
            for f in repo_path.rglob(f"*{ext}"):
                if not self._is_ignored(f):
                    try:
                        loc += sum(1 for _ in open(f, errors="ignore"))
                    except:
                        pass
        return loc

    def get_capabilities(self) -> list[str]:
        return ["analyze_structure", "analyze_dependencies", "analyze_quality"]


def create_executor(config: Dict[str, Any] = None) -> StaticAnalyzerExecutor:
    return StaticAnalyzerExecutor(config)
