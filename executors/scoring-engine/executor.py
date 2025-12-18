"""
Scoring Engine Executor
Calculates repo health, tech debt, and product level scores
"""
from typing import Any, Dict
from pathlib import Path
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from base import BaseExecutor

logger = logging.getLogger(__name__)


# Product level thresholds
PRODUCT_LEVELS = [
    {"name": "Near-Product", "min_health": 10, "min_debt": 12, "min_security": 2},
    {"name": "Platform Module Candidate", "min_health": 8, "min_debt": 10, "min_security": 1},
    {"name": "Internal Tool", "min_health": 6, "min_debt": 7, "min_security": 0},
    {"name": "Prototype", "min_health": 4, "min_debt": 4, "min_security": 0},
    {"name": "R&D Spike", "min_health": 0, "min_debt": 0, "min_security": 0},
]

# Complexity thresholds based on LOC
COMPLEXITY_THRESHOLDS = {
    "XS": 1000,
    "S": 5000,
    "M": 20000,
    "L": 100000,
    "XL": float("inf"),
}


class ScoringEngineExecutor(BaseExecutor):
    """Executor for calculating scores"""

    name = "scoring-engine"

    async def run(self, action: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if action == "calculate":
            return await self.calculate(**inputs)
        raise ValueError(f"Unknown action: {action}")

    async def calculate(
        self,
        structure: dict,
        git_history: dict,
        dependencies: dict,
        security: dict,
        code_quality: dict,
        llm_review: dict = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate all scores from collected metrics.
        """
        # Calculate Repo Health (0-12)
        repo_health = self._calculate_repo_health(structure, git_history)

        # Calculate Tech Debt (0-15)
        tech_debt = self._calculate_tech_debt(structure, dependencies, code_quality)

        # Get security score
        security_score = security.get("security_score", 0)

        # Determine complexity category
        loc = structure.get("loc", 0)
        complexity = self._calculate_complexity(loc)

        # Determine product level
        product_level = self._classify_product_level(
            repo_health, tech_debt, security_score, structure
        )

        # Calculate overall readiness (0-100%)
        overall_readiness = self._calculate_readiness(
            repo_health, tech_debt, security_score
        )

        return {
            "repo_health": repo_health,
            "repo_health_max": 12,
            "tech_debt": tech_debt,
            "tech_debt_max": 15,
            "security_score": security_score,
            "security_max": 3,
            "product_level": product_level,
            "overall_readiness": round(overall_readiness, 1),
            "complexity_category": complexity,
            "breakdown": {
                "health": self._health_breakdown(structure, git_history),
                "debt": self._debt_breakdown(structure, dependencies, code_quality),
            }
        }

    def _calculate_repo_health(self, structure: dict, git_history: dict) -> int:
        """
        Calculate Repo Health Score (0-12 points)

        Criteria:
        - README: 2 points
        - License: 1 point
        - Tests: 2 points
        - CI/CD: 2 points
        - Docker: 1 point
        - Active commits: 2 points
        - Multiple contributors: 2 points
        """
        score = 0

        # Documentation (3 points)
        if structure.get("has_readme"):
            score += 2
        if structure.get("has_license"):
            score += 1

        # Testing & CI (4 points)
        if structure.get("has_tests"):
            score += 2
        if structure.get("has_ci"):
            score += 2

        # Containerization (1 point)
        if structure.get("has_docker"):
            score += 1

        # Activity (2 points)
        commit_freq = git_history.get("commit_frequency", 0)
        if commit_freq >= 1:  # >= 1 commit/week
            score += 2
        elif commit_freq >= 0.25:  # >= 1 commit/month
            score += 1

        # Team (2 points)
        contributors = git_history.get("contributors", 1)
        if contributors >= 3:
            score += 2
        elif contributors >= 2:
            score += 1

        return min(score, 12)

    def _calculate_tech_debt(
        self,
        structure: dict,
        dependencies: dict,
        code_quality: dict
    ) -> int:
        """
        Calculate Tech Debt Score (0-15 points, higher = less debt)

        Criteria:
        - Low complexity: 3 points
        - Low duplication: 3 points
        - Few lint issues: 3 points
        - Updated dependencies: 3 points
        - Good test coverage: 3 points
        """
        score = 0

        # Complexity (3 points)
        complexity = code_quality.get("complexity_score", 20)
        if complexity <= 10:
            score += 3
        elif complexity <= 15:
            score += 2
        elif complexity <= 25:
            score += 1

        # Duplication (3 points)
        duplication = code_quality.get("duplication_percent", 10)
        if duplication <= 3:
            score += 3
        elif duplication <= 5:
            score += 2
        elif duplication <= 10:
            score += 1

        # Lint issues per KLOC (3 points)
        lint_per_kloc = code_quality.get("lint_issues_per_kloc", 15)
        if lint_per_kloc <= 5:
            score += 3
        elif lint_per_kloc <= 10:
            score += 2
        elif lint_per_kloc <= 20:
            score += 1

        # Dependencies (3 points)
        total_deps = dependencies.get("total_count", 0)
        outdated = dependencies.get("outdated_count", 0)
        if total_deps > 0:
            outdated_pct = (outdated / total_deps) * 100
            if outdated_pct <= 10:
                score += 3
            elif outdated_pct <= 25:
                score += 2
            elif outdated_pct <= 50:
                score += 1
        else:
            score += 2  # No deps = less maintenance

        # Test coverage (3 points)
        coverage = code_quality.get("test_coverage", 0)
        if coverage >= 80:
            score += 3
        elif coverage >= 60:
            score += 2
        elif coverage >= 40:
            score += 1

        return min(score, 15)

    def _calculate_complexity(self, loc: int) -> str:
        """Determine complexity category from LOC"""
        for category, threshold in COMPLEXITY_THRESHOLDS.items():
            if loc < threshold:
                return category
        return "XL"

    def _classify_product_level(
        self,
        health: int,
        debt: int,
        security: int,
        structure: dict
    ) -> str:
        """Classify project into product level"""
        for level in PRODUCT_LEVELS:
            if (health >= level["min_health"] and
                debt >= level["min_debt"] and
                security >= level["min_security"]):

                # Additional checks for higher levels
                if level["name"] == "Near-Product":
                    if not all([structure.get("has_tests"),
                               structure.get("has_ci"),
                               structure.get("has_docker")]):
                        continue

                if level["name"] == "Platform Module Candidate":
                    if not all([structure.get("has_tests"),
                               structure.get("has_ci")]):
                        continue

                return level["name"]

        return "R&D Spike"

    def _calculate_readiness(self, health: int, debt: int, security: int) -> float:
        """
        Calculate overall readiness percentage.

        Weighted average:
        - Health: 40%
        - Tech Debt: 40%
        - Security: 20%
        """
        health_pct = (health / 12) * 40
        debt_pct = (debt / 15) * 40
        security_pct = (security / 3) * 20

        return health_pct + debt_pct + security_pct

    def _health_breakdown(self, structure: dict, git_history: dict) -> dict:
        """Detailed health score breakdown"""
        return {
            "readme": {"present": structure.get("has_readme", False), "points": 2 if structure.get("has_readme") else 0, "max": 2},
            "license": {"present": structure.get("has_license", False), "points": 1 if structure.get("has_license") else 0, "max": 1},
            "tests": {"present": structure.get("has_tests", False), "points": 2 if structure.get("has_tests") else 0, "max": 2},
            "ci": {"present": structure.get("has_ci", False), "points": 2 if structure.get("has_ci") else 0, "max": 2},
            "docker": {"present": structure.get("has_docker", False), "points": 1 if structure.get("has_docker") else 0, "max": 1},
            "activity": {"frequency": git_history.get("commit_frequency", 0), "max": 2},
            "contributors": {"count": git_history.get("contributors", 1), "max": 2},
        }

    def _debt_breakdown(
        self,
        structure: dict,
        dependencies: dict,
        code_quality: dict
    ) -> dict:
        """Detailed debt score breakdown"""
        return {
            "complexity": {"value": code_quality.get("complexity_score", 0), "threshold": "<=10 good", "max": 3},
            "duplication": {"value": code_quality.get("duplication_percent", 0), "threshold": "<=3% good", "max": 3},
            "lint_issues": {"value": code_quality.get("lint_issues_per_kloc", 0), "threshold": "<=5 good", "max": 3},
            "dependencies": {"outdated": dependencies.get("outdated_count", 0), "total": dependencies.get("total_count", 0), "max": 3},
            "test_coverage": {"value": code_quality.get("test_coverage", 0), "threshold": ">=80% good", "max": 3},
        }

    def get_capabilities(self) -> list[str]:
        return ["calculate"]


def create_executor(config: Dict[str, Any] = None) -> ScoringEngineExecutor:
    return ScoringEngineExecutor(config)
