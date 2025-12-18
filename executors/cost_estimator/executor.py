"""
Cost Estimator Executor - Complete Edition

Features:
- COCOMO II (Modern + Classic)
- 8 Estimation Methodologies
- PERT 3-Point Analysis
- AI Efficiency Comparison
- ROI Calculator
- 8 Regional Rate Profiles
- Tech Debt Multipliers
"""
import math
from dataclasses import dataclass
from typing import Any, Dict, List
from pathlib import Path
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from base import BaseExecutor
from .formulas import (
    REGIONAL_RATES,
    COCOMO_CONSTANTS,
    EFFORT_MULTIPLIERS,
    TECH_DEBT_MULTIPLIERS,
    COMPLEXITY_THRESHOLDS,
    ACTIVITY_RATIOS,
    AI_PRODUCTIVITY,
    METHODOLOGIES,
    estimate_cocomo_modern,
    estimate_methodology,
    estimate_all_methodologies,
    calculate_pert,
    estimate_ai_efficiency,
    calculate_roi,
    get_regional_cost,
    get_all_regional_costs,
    get_complexity,
    get_tech_debt_multiplier,
    get_all_formulas,
    get_all_constants,
)

logger = logging.getLogger(__name__)


@dataclass
class CocomoResult:
    """COCOMO II estimation result"""
    effort_person_months: float
    duration_months: float
    team_size: float
    hours_typical: float
    hours_min: float
    hours_max: float


class CostEstimatorExecutor(BaseExecutor):
    """
    Executor for comprehensive cost estimation.

    Actions:
    - estimate: COCOMO II Modern estimate
    - estimate_classic: COCOMO II Classic estimate
    - estimate_comprehensive: All 8 methodologies
    - estimate_methodology: Single methodology
    - estimate_pert: PERT 3-point analysis
    - estimate_ai_efficiency: AI vs Human comparison
    - calculate_roi: ROI analysis
    - get_regional_costs: Cost for all 8 regions
    - get_formulas: All formulas
    - get_constants: All constants
    """

    name = "cost-estimator"

    async def run(self, action: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        actions = {
            "estimate": self.estimate,
            "estimate_classic": self.estimate_classic,
            "estimate_comprehensive": self.estimate_comprehensive,
            "estimate_methodology": self.estimate_single_methodology,
            "estimate_pert": self.estimate_pert,
            "estimate_ai_efficiency": self.ai_efficiency,
            "calculate_roi": self.roi,
            "get_regional_costs": self.regional_costs,
            "get_formulas": self.formulas,
            "get_constants": self.constants,
            "compare_cost": self.compare_cost,
        }

        if action not in actions:
            return {"error": f"Unknown action: {action}", "available": list(actions.keys())}

        return await actions[action](**inputs)

    async def estimate(
        self,
        loc: int,
        region: str = "eu",
        tech_debt_score: int = 10,
        team_experience: str = "nominal",
        **kwargs
    ) -> Dict[str, Any]:
        """
        COCOMO II Modern estimate.

        Formula: Effort = 0.5 × (KLOC)^0.85 × EAF
        """
        cocomo = estimate_cocomo_modern(loc, tech_debt_score, team_experience)

        # Add regional cost
        hours = cocomo["hours"]["typical"]
        regional = get_all_regional_costs(hours)

        # Activity breakdown
        breakdown = {
            activity: round(hours * ratios["typical"])
            for activity, ratios in ACTIVITY_RATIOS.items()
        }

        return {
            **cocomo,
            "hours_breakdown": breakdown,
            "cost_by_region": regional["regions"],
            "complexity": get_complexity(loc),
            "tech_debt_multiplier": get_tech_debt_multiplier(tech_debt_score),
        }

    async def estimate_classic(
        self,
        loc: int,
        project_type: str = "semi",
        region: str = "eu",
        **kwargs
    ) -> Dict[str, Any]:
        """
        COCOMO II Classic estimate.

        Project types: organic, semi, embedded
        """
        from .formulas import COCOMO_CLASSIC

        coef = COCOMO_CLASSIC.get(project_type, COCOMO_CLASSIC["semi"])
        kloc = max(loc / 1000, 0.1)

        # Classic COCOMO II
        effort_pm = coef["a"] * (kloc ** coef["b"])
        duration = coef["c"] * (effort_pm ** coef["d"])
        team_size = effort_pm / max(duration, 0.5)
        hours = effort_pm * 152

        regional = get_all_regional_costs(hours)

        return {
            "methodology": "COCOMO II (Classic)",
            "project_type": project_type,
            "formula": f"Effort = {coef['a']} × ({kloc:.2f})^{coef['b']} = {effort_pm:.2f} PM",
            "inputs": {"loc": loc, "kloc": round(kloc, 2)},
            "effort_pm": round(effort_pm, 2),
            "schedule_months": round(duration, 1),
            "team_size": round(team_size, 1),
            "hours": {
                "min": round(hours * 0.7),
                "typical": round(hours),
                "max": round(hours * 1.3),
            },
            "cost_by_region": regional["regions"],
        }

    async def estimate_comprehensive(
        self,
        loc: int,
        complexity: float = 1.5,
        hourly_rate: float = 35,
        **kwargs
    ) -> Dict[str, Any]:
        """
        All 8 methodologies + PERT.

        Methodologies: COCOMO II, Gartner, IEEE 1063, Microsoft,
                      Google, PMI, SEI SLIM, Function Points
        """
        return estimate_all_methodologies(loc, complexity, hourly_rate)

    async def estimate_single_methodology(
        self,
        methodology: str,
        loc: int,
        complexity: float = 1.5,
        hourly_rate: float = 35,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Single methodology estimate.

        Available: cocomo, gartner, ieee, microsoft, google, pmi, sei_slim, function_points
        """
        return estimate_methodology(methodology, loc, complexity, hourly_rate)

    async def estimate_pert(
        self,
        optimistic_hours: float,
        most_likely_hours: float,
        pessimistic_hours: float,
        hourly_rate: float = 35,
        **kwargs
    ) -> Dict[str, Any]:
        """
        PERT 3-point analysis.

        Expected = (O + 4×M + P) / 6
        σ = (P - O) / 6

        Returns 68%, 95%, 99% confidence intervals.
        """
        pert = calculate_pert(optimistic_hours, most_likely_hours, pessimistic_hours)

        # Add cost estimates
        pert["cost"] = {
            "expected": round(pert["expected"] * hourly_rate, 2),
            "range_68": {
                "min": round(pert["confidence_68"]["min"] * hourly_rate, 2),
                "max": round(pert["confidence_68"]["max"] * hourly_rate, 2),
            },
            "range_95": {
                "min": round(pert["confidence_95"]["min"] * hourly_rate, 2),
                "max": round(pert["confidence_95"]["max"] * hourly_rate, 2),
            },
        }
        pert["hourly_rate"] = hourly_rate

        return pert

    async def ai_efficiency(
        self,
        loc: int,
        hourly_rate: float = 35,
        complexity: float = 1.5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compare Pure Human vs AI-Assisted vs Hybrid.

        Productivity (hrs/KLOC):
        - Pure Human: 25
        - AI-Assisted: 8
        - Hybrid: 6.5
        """
        return estimate_ai_efficiency(loc, hourly_rate, complexity)

    async def roi(
        self,
        investment_cost: float,
        annual_support_savings: float = 0,
        annual_training_savings: float = 0,
        annual_efficiency_gain: float = 0,
        annual_risk_reduction: float = 0,
        maintenance_percent: float = 20,
        **kwargs
    ) -> Dict[str, Any]:
        """
        ROI analysis.

        Returns: payback period, NPV 3yr, ROI 1yr/3yr %
        """
        return calculate_roi(
            investment_cost,
            annual_support_savings,
            annual_training_savings,
            annual_efficiency_gain,
            annual_risk_reduction,
            maintenance_percent,
        )

    async def regional_costs(
        self,
        hours: float,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate cost for all 8 regions.

        Regions: ua, ua_compliance, pl, eu, de, uk, us, in
        """
        return get_all_regional_costs(hours)

    async def compare_cost(
        self,
        actual_cost: float,
        loc: int,
        region: str = "ua",
        actual_hours: float = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compare actual cost vs COCOMO estimate.

        Returns verdict: overpaid, underpaid, within_range
        """
        cocomo = estimate_cocomo_modern(loc)
        hours = cocomo["hours"]["typical"]
        rates = REGIONAL_RATES.get(region, REGIONAL_RATES["eu"])
        estimated = hours * rates["rates"]["typical"]

        deviation = ((actual_cost - estimated) / estimated) * 100 if estimated > 0 else 0

        if deviation > 20:
            verdict = "overpaid"
            advice = "Cost significantly exceeds estimate. Review scope or negotiate."
        elif deviation < -20:
            verdict = "underpaid"
            advice = "Cost below estimate. Verify quality and completeness."
        else:
            verdict = "within_range"
            advice = "Cost is within acceptable ±20% range."

        return {
            "actual": {"cost": actual_cost, "hours": actual_hours},
            "estimated": {
                "cost_typical": round(estimated),
                "hours_typical": round(hours),
            },
            "analysis": {
                "deviation_percent": round(deviation, 1),
                "deviation_amount": round(actual_cost - estimated),
                "verdict": verdict,
                "advice": advice,
            },
            "inputs": {"loc": loc, "region": region},
        }

    async def formulas(self, **kwargs) -> Dict[str, Any]:
        """Get all formulas documentation."""
        return get_all_formulas()

    async def constants(self, **kwargs) -> Dict[str, Any]:
        """Get all constants."""
        return get_all_constants()

    def get_capabilities(self) -> list[str]:
        return [
            "estimate",
            "estimate_classic",
            "estimate_comprehensive",
            "estimate_methodology",
            "estimate_pert",
            "estimate_ai_efficiency",
            "calculate_roi",
            "get_regional_costs",
            "compare_cost",
            "get_formulas",
            "get_constants",
        ]


def create_executor(config: Dict[str, Any] = None) -> CostEstimatorExecutor:
    return CostEstimatorExecutor(config)
