"""
Contract Checker Executor
Checks compliance with contract requirements
"""
from typing import Any, Dict, Optional
from pathlib import Path
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from base import BaseExecutor

logger = logging.getLogger(__name__)


# Sample contract templates (would be loaded from database in production)
CONTRACT_TEMPLATES = {
    "standard": {
        "name": "Standard Contract",
        "requirements": {
            "repo_health": {"min": 6, "description": "Basic repository hygiene"},
            "tech_debt": {"min": 8, "description": "Manageable technical debt"},
            "security_score": {"min": 1, "description": "No critical vulnerabilities"},
            "has_tests": {"value": True, "description": "Must have tests"},
            "has_ci": {"value": True, "description": "Must have CI/CD"},
        }
    },
    "enterprise": {
        "name": "Enterprise Contract",
        "requirements": {
            "repo_health": {"min": 10, "description": "High repository quality"},
            "tech_debt": {"min": 12, "description": "Low technical debt"},
            "security_score": {"min": 2, "description": "Strong security posture"},
            "has_tests": {"value": True, "description": "Must have tests"},
            "has_ci": {"value": True, "description": "Must have CI/CD"},
            "has_docker": {"value": True, "description": "Must be containerized"},
            "product_level": {"min": "Platform Module Candidate", "description": "Production-ready level"},
        }
    },
    "global_fund": {
        "name": "Global Fund R13",
        "requirements": {
            "repo_health": {"min": 8, "description": "Good documentation and practices"},
            "tech_debt": {"min": 10, "description": "Sustainable codebase"},
            "security_score": {"min": 2, "description": "Security audit passed"},
            "has_tests": {"value": True, "description": "Testing required"},
            "has_ci": {"value": True, "description": "Automated builds required"},
            "has_api_docs": {"value": True, "description": "API documentation required"},
        }
    },
    "minimal": {
        "name": "Minimal Requirements",
        "requirements": {
            "repo_health": {"min": 4, "description": "Basic organization"},
            "has_readme": {"value": True, "description": "Must have README"},
        }
    }
}

# Product level ordering for comparison
PRODUCT_LEVEL_ORDER = [
    "R&D Spike",
    "Prototype",
    "Internal Tool",
    "Platform Module Candidate",
    "Near-Product",
]


class ContractCheckerExecutor(BaseExecutor):
    """Executor for checking contract compliance"""

    name = "contract-checker"

    async def run(self, action: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if action == "check":
            return await self.check(**inputs)
        elif action == "list_templates":
            return await self.list_templates()
        raise ValueError(f"Unknown action: {action}")

    async def check(
        self,
        contract_id: Optional[str],
        scores: dict,
        cost: dict,
        structure: dict,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Check if project meets contract requirements.

        Args:
            contract_id: Contract template ID or custom requirements
            scores: Scoring results
            cost: Cost estimation results
            structure: Structure analysis results
        """
        if not contract_id:
            return {
                "compliant": True,
                "gaps": [],
                "requirements_met": 0,
                "requirements_total": 0,
                "blockers": [],
                "warnings": [],
                "message": "No contract specified"
            }

        # Get contract template
        template = CONTRACT_TEMPLATES.get(contract_id, CONTRACT_TEMPLATES["standard"])
        requirements = template["requirements"]

        gaps = []
        blockers = []
        warnings = []
        met = 0
        total = len(requirements)

        for req_name, req_value in requirements.items():
            actual = self._get_actual_value(req_name, scores, structure)

            if isinstance(req_value, dict):
                if "min" in req_value:
                    # Numeric minimum
                    required = req_value["min"]
                    description = req_value.get("description", "")

                    if req_name == "product_level":
                        # Special handling for product level
                        if self._compare_product_levels(actual, required):
                            met += 1
                        else:
                            gap = {
                                "requirement": req_name,
                                "required": required,
                                "actual": actual,
                                "description": description,
                                "severity": "high"
                            }
                            gaps.append(gap)
                            blockers.append(f"{description}: need {required}, have {actual}")
                    else:
                        if actual >= required:
                            met += 1
                        else:
                            gap = {
                                "requirement": req_name,
                                "required": required,
                                "actual": actual,
                                "gap": required - actual,
                                "description": description,
                                "severity": "high" if (required - actual) > 2 else "medium"
                            }
                            gaps.append(gap)
                            if gap["severity"] == "high":
                                blockers.append(f"{description}: need {required}, have {actual}")
                            else:
                                warnings.append(f"{description}: need {required}, have {actual}")

                elif "value" in req_value:
                    # Boolean requirement
                    required = req_value["value"]
                    description = req_value.get("description", "")

                    if actual == required:
                        met += 1
                    else:
                        gap = {
                            "requirement": req_name,
                            "required": required,
                            "actual": actual,
                            "description": description,
                            "severity": "high"
                        }
                        gaps.append(gap)
                        blockers.append(f"{description}: missing")

        compliant = len(blockers) == 0

        return {
            "compliant": compliant,
            "gaps": gaps,
            "requirements_met": met,
            "requirements_total": total,
            "blockers": blockers,
            "warnings": warnings,
            "contract_name": template["name"],
            "compliance_percent": round((met / total) * 100, 1) if total > 0 else 100,
            "summary": f"{'Compliant' if compliant else 'Non-compliant'}: {met}/{total} requirements met"
        }

    async def list_templates(self) -> Dict[str, Any]:
        """List available contract templates"""
        return {
            "templates": [
                {
                    "id": template_id,
                    "name": template["name"],
                    "requirements_count": len(template["requirements"])
                }
                for template_id, template in CONTRACT_TEMPLATES.items()
            ]
        }

    def _get_actual_value(self, req_name: str, scores: dict, structure: dict) -> Any:
        """Get actual value for a requirement"""
        # Check in scores first
        if req_name in scores:
            return scores[req_name]

        # Check in structure
        if req_name in structure:
            return structure[req_name]

        # Special mappings
        mappings = {
            "has_readme": structure.get("has_readme", False),
            "has_tests": structure.get("has_tests", False),
            "has_ci": structure.get("has_ci", False),
            "has_docker": structure.get("has_docker", False),
            "has_api_docs": structure.get("has_api_docs", False),
        }

        return mappings.get(req_name, None)

    def _compare_product_levels(self, actual: str, required: str) -> bool:
        """Compare product levels"""
        try:
            actual_idx = PRODUCT_LEVEL_ORDER.index(actual)
            required_idx = PRODUCT_LEVEL_ORDER.index(required)
            return actual_idx >= required_idx
        except ValueError:
            return False

    def get_capabilities(self) -> list[str]:
        return ["check", "list_templates"]


def create_executor(config: Dict[str, Any] = None) -> ContractCheckerExecutor:
    return ContractCheckerExecutor(config)
