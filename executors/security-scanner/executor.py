"""
Security Scanner Executor
Scans for vulnerabilities, secrets, and security issues
"""
import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from base import BaseExecutor

logger = logging.getLogger(__name__)


class SecurityScannerExecutor(BaseExecutor):
    """Executor for security scanning"""

    name = "security-scanner"

    # Secret patterns
    SECRET_PATTERNS = [
        (r"(?i)api[_-]?key\s*[=:]\s*['\"]?[\w-]{20,}", "API Key"),
        (r"(?i)secret[_-]?key\s*[=:]\s*['\"]?[\w-]{20,}", "Secret Key"),
        (r"(?i)password\s*[=:]\s*['\"]?[^\s'\"]{8,}", "Password"),
        (r"(?i)token\s*[=:]\s*['\"]?[\w-]{20,}", "Token"),
        (r"ghp_[a-zA-Z0-9]{36}", "GitHub PAT"),
        (r"gsk_[a-zA-Z0-9]{50,}", "Groq API Key"),
        (r"sk-[a-zA-Z0-9]{48}", "OpenAI API Key"),
        (r"sk-ant-[a-zA-Z0-9-]{90,}", "Anthropic API Key"),
        (r"AKIA[0-9A-Z]{16}", "AWS Access Key"),
    ]

    async def run(self, action: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Route to appropriate method"""
        if action == "scan":
            return await self.scan(**inputs)
        else:
            raise ValueError(f"Unknown action: {action}")

    async def scan(self, path: str, languages: list = None, **kwargs) -> Dict[str, Any]:
        """Run full security scan"""
        repo_path = Path(path)

        vulnerabilities = []
        secrets_found = []

        # Scan for secrets
        secrets = await self._scan_secrets(repo_path)
        secrets_found.extend(secrets)

        # Run Semgrep if available
        semgrep_results = await self._run_semgrep(repo_path)
        vulnerabilities.extend(semgrep_results)

        # Run Bandit for Python (with timeout and limits)
        if languages and "Python" in languages:
            bandit_results = await self._run_bandit(repo_path)
            vulnerabilities.extend(bandit_results)

        # Calculate security score
        critical = sum(1 for v in vulnerabilities if v.get("severity") == "critical")
        high = sum(1 for v in vulnerabilities if v.get("severity") == "high")
        secrets_count = len(secrets_found)

        # Score: 3 = clean, 2 = minor issues, 1 = concerns, 0 = critical
        if critical > 0 or secrets_count > 0:
            security_score = 0
        elif high > 0:
            security_score = 1
        elif len(vulnerabilities) > 5:
            security_score = 2
        else:
            security_score = 3

        return {
            "vulnerabilities": vulnerabilities[:20],  # Limit response size
            "vulnerability_count": len(vulnerabilities),
            "secrets_found": secrets_count,
            "secret_locations": [s["file"] for s in secrets_found[:5]],
            "security_score": security_score,
            "critical_count": critical,
            "high_count": high
        }

    async def _scan_secrets(self, repo_path: Path) -> list[dict]:
        """Scan for hardcoded secrets"""
        secrets = []
        ignored = [".git", "node_modules", "__pycache__", ".venv", "venv", "dist", ".env.example"]

        for f in repo_path.rglob("*"):
            if not f.is_file():
                continue
            if any(part in f.parts for part in ignored):
                continue
            if f.suffix in [".png", ".jpg", ".gif", ".ico", ".woff", ".ttf", ".lock"]:
                continue

            try:
                content = f.read_text(errors="ignore")
                for pattern, secret_type in self.SECRET_PATTERNS:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        # Skip if in .env.example or test files
                        if ".example" in str(f) or "test" in str(f).lower():
                            continue
                        secrets.append({
                            "type": secret_type,
                            "file": str(f.relative_to(repo_path)),
                            "pattern": pattern[:30]
                        })
            except:
                pass

        return secrets

    async def _run_semgrep(self, repo_path: Path) -> list[dict]:
        """Run Semgrep security scanner"""
        vulnerabilities = []
        try:
            proc = await asyncio.create_subprocess_exec(
                "semgrep", "scan", "--config", "auto", "--json", "--quiet",
                "--timeout", "30", "--max-target-bytes", "1000000",
                str(repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)

            if stdout:
                data = json.loads(stdout.decode())
                for result in data.get("results", []):
                    vulnerabilities.append({
                        "type": "semgrep",
                        "rule": result.get("check_id", "unknown"),
                        "severity": result.get("extra", {}).get("severity", "medium").lower(),
                        "file": result.get("path", ""),
                        "line": result.get("start", {}).get("line", 0),
                        "message": result.get("extra", {}).get("message", "")[:200]
                    })
        except asyncio.TimeoutError:
            logger.warning("Semgrep timed out")
        except FileNotFoundError:
            logger.info("Semgrep not installed, skipping")
        except Exception as e:
            logger.warning(f"Semgrep error: {e}")

        return vulnerabilities

    async def _run_bandit(self, repo_path: Path) -> list[dict]:
        """Run Bandit for Python security"""
        vulnerabilities = []
        try:
            # Use --exclude to skip large directories
            proc = await asyncio.create_subprocess_exec(
                "bandit", "-r", str(repo_path), "-f", "json",
                "--exclude", ".venv,venv,node_modules,__pycache__,.git",
                "-ll",  # Only medium+ severity
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60)

            if stdout:
                data = json.loads(stdout.decode())
                for result in data.get("results", []):
                    severity = result.get("issue_severity", "MEDIUM").lower()
                    vulnerabilities.append({
                        "type": "bandit",
                        "rule": result.get("test_id", ""),
                        "severity": severity,
                        "file": result.get("filename", ""),
                        "line": result.get("line_number", 0),
                        "message": result.get("issue_text", "")[:200]
                    })
        except asyncio.TimeoutError:
            logger.warning("Bandit timed out")
        except FileNotFoundError:
            logger.info("Bandit not installed, skipping")
        except Exception as e:
            logger.warning(f"Bandit error: {e}")

        return vulnerabilities

    def get_capabilities(self) -> list[str]:
        return ["scan"]


def create_executor(config: Dict[str, Any] = None) -> SecurityScannerExecutor:
    return SecurityScannerExecutor(config)
