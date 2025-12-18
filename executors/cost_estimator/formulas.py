"""
Estimation Formulas & Constants - Complete Edition

All estimation methodologies, regional rates, and formulas
hardcoded for audit-platform.

Based on:
- COCOMO II (Boehm 2000, Modern Calibration 2020s)
- Gartner Research 2023
- IEEE 1063, PMI, Google, Microsoft Standards
- SEI SLIM Model
- ISO/IEC 20926 (Function Points)
"""
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


# =============================================================================
# REGIONAL RATES (8 regions)
# =============================================================================

REGIONAL_RATES = {
    "ua": {
        "name": "Ukraine",
        "currency": "USD",
        "symbol": "$",
        "rates": {"junior": 15, "middle": 25, "senior": 45, "typical": 28},
        "overhead": 1.20,
    },
    "ua_compliance": {
        "name": "Ukraine (Compliance)",
        "currency": "USD",
        "symbol": "$",
        "rates": {"junior": 20, "middle": 35, "senior": 55, "typical": 37},
        "overhead": 1.25,
    },
    "pl": {
        "name": "Poland",
        "currency": "EUR",
        "symbol": "€",
        "rates": {"junior": 25, "middle": 40, "senior": 60, "typical": 42},
        "overhead": 1.25,
    },
    "eu": {
        "name": "EU Standard",
        "currency": "EUR",
        "symbol": "€",
        "rates": {"junior": 35, "middle": 55, "senior": 85, "typical": 58},
        "overhead": 1.35,
    },
    "de": {
        "name": "Germany",
        "currency": "EUR",
        "symbol": "€",
        "rates": {"junior": 45, "middle": 70, "senior": 100, "typical": 72},
        "overhead": 1.40,
    },
    "uk": {
        "name": "United Kingdom",
        "currency": "GBP",
        "symbol": "£",
        "rates": {"junior": 40, "middle": 65, "senior": 95, "typical": 67},
        "overhead": 1.35,
    },
    "us": {
        "name": "United States",
        "currency": "USD",
        "symbol": "$",
        "rates": {"junior": 50, "middle": 85, "senior": 130, "typical": 88},
        "overhead": 1.40,
    },
    "in": {
        "name": "India",
        "currency": "USD",
        "symbol": "$",
        "rates": {"junior": 12, "middle": 22, "senior": 35, "typical": 23},
        "overhead": 1.15,
    },
}


# =============================================================================
# COCOMO II CONSTANTS (Modern Calibration 2020s)
# =============================================================================

COCOMO_CONSTANTS = {
    "a": 0.5,        # Effort coefficient (modern: reduced from classic 2.94)
    "b": 0.85,       # Base exponent (modern: reduced from 1.1)
    "c": 2.0,        # Schedule coefficient
    "d": 0.35,       # Schedule exponent
    "hours_per_pm": 160,  # Hours per person-month
}

# Classic COCOMO II coefficients (for comparison)
COCOMO_CLASSIC = {
    "organic": {"a": 2.4, "b": 1.05, "c": 2.5, "d": 0.38},
    "semi": {"a": 3.0, "b": 1.12, "c": 2.5, "d": 0.35},
    "embedded": {"a": 3.6, "b": 1.20, "c": 2.5, "d": 0.32},
}

# Effort Multipliers (EAF components)
EFFORT_MULTIPLIERS = {
    "team_experience": {"low": 1.3, "nominal": 1.0, "high": 0.8},
    "process_maturity": {"low": 1.2, "nominal": 1.0, "high": 0.85},
    "tool_support": {"low": 1.15, "nominal": 1.0, "high": 0.9},
    "requirements_volatility": {"low": 0.9, "nominal": 1.0, "high": 1.25},
}


# =============================================================================
# TECH DEBT MULTIPLIERS
# =============================================================================

TECH_DEBT_MULTIPLIERS = {
    (0, 3): 1.5,     # Very high debt
    (4, 6): 1.3,     # High debt
    (7, 9): 1.15,    # Moderate debt
    (10, 12): 1.05,  # Low debt
    (13, 15): 1.0,   # Minimal debt
}


# =============================================================================
# COMPLEXITY THRESHOLDS (LOC)
# =============================================================================

COMPLEXITY_THRESHOLDS = {
    "XS": {"min": 0, "max": 2000, "base_hours": {"min": 16, "typical": 40, "max": 80}},
    "S": {"min": 2000, "max": 8000, "base_hours": {"min": 40, "typical": 80, "max": 160}},
    "M": {"min": 8000, "max": 40000, "base_hours": {"min": 160, "typical": 400, "max": 800}},
    "L": {"min": 40000, "max": 120000, "base_hours": {"min": 800, "typical": 1600, "max": 3200}},
    "XL": {"min": 120000, "max": float("inf"), "base_hours": {"min": 3200, "typical": 6400, "max": 12800}},
}


# =============================================================================
# ACTIVITY DISTRIBUTION
# =============================================================================

ACTIVITY_RATIOS = {
    "analysis": {"min": 0.08, "typical": 0.12, "max": 0.15},
    "design": {"min": 0.12, "typical": 0.18, "max": 0.22},
    "implementation": {"min": 0.35, "typical": 0.42, "max": 0.50},
    "testing": {"min": 0.15, "typical": 0.20, "max": 0.25},
    "documentation": {"min": 0.05, "typical": 0.08, "max": 0.12},
}


# =============================================================================
# PRODUCT STAGES
# =============================================================================

PRODUCT_STAGES = [
    {"id": "spike", "name": "R&D Spike", "min_health": 0, "min_debt": 0},
    {"id": "poc", "name": "Proof of Concept", "min_health": 2, "min_debt": 3},
    {"id": "prototype", "name": "Prototype", "min_health": 4, "min_debt": 5},
    {"id": "mvp", "name": "MVP", "min_health": 5, "min_debt": 6},
    {"id": "alpha", "name": "Alpha", "min_health": 7, "min_debt": 8},
    {"id": "beta", "name": "Beta", "min_health": 8, "min_debt": 10},
    {"id": "rc", "name": "Release Candidate", "min_health": 9, "min_debt": 12},
    {"id": "production", "name": "Production Ready", "min_health": 10, "min_debt": 13},
]


# =============================================================================
# AI PRODUCTIVITY RATES (hours per 1000 LOC)
# =============================================================================

AI_PRODUCTIVITY = {
    "pure_human": 25,      # Traditional development
    "ai_assisted": 8,      # AI generates, human reviews
    "hybrid": 6.5,         # Optimized AI+Human workflow
}


# =============================================================================
# 8 ESTIMATION METHODOLOGIES
# =============================================================================

METHODOLOGIES = {
    "cocomo": {
        "id": "cocomo",
        "name": "COCOMO II (Modern)",
        "formula": "Effort = 0.5 × (KLOC)^0.85 × EAF",
        "source": "Boehm et al. (2000), Modern calibration",
        "confidence": "High",
        "description": "Industry-standard parametric model for software cost estimation",
    },
    "gartner": {
        "id": "gartner",
        "name": "Gartner Standard",
        "formula": "Days = words ÷ 650 × complexity",
        "source": "Gartner Research 2023",
        "confidence": "High",
        "description": "Enterprise documentation standard (500-800 words/day)",
    },
    "ieee": {
        "id": "ieee",
        "name": "IEEE 1063",
        "formula": "Days = pages ÷ 1.5 × complexity",
        "source": "IEEE Standard 1063",
        "confidence": "High",
        "description": "Technical documentation standard (1-2 pages/day)",
    },
    "microsoft": {
        "id": "microsoft",
        "name": "Microsoft Standard",
        "formula": "Days = words ÷ 650 × complexity",
        "source": "Microsoft Documentation Standards",
        "confidence": "Medium",
        "description": "Tech industry standard (650 words/day)",
    },
    "google": {
        "id": "google",
        "name": "Google Guidelines",
        "formula": "Hours = pages × 4 × complexity",
        "source": "Google Technical Writing Guidelines",
        "confidence": "Medium",
        "description": "UX-driven approach (4 hours per page)",
    },
    "pmi": {
        "id": "pmi",
        "name": "PMI Standard",
        "formula": "Days = pages × 0.25 × complexity",
        "source": "PMI Project Management Standards",
        "confidence": "Medium",
        "description": "Project management approach (25% of project effort)",
    },
    "sei_slim": {
        "id": "sei_slim",
        "name": "SEI SLIM",
        "formula": "Days = 180 × 0.4 × complexity",
        "source": "SEI SLIM Model",
        "confidence": "Medium",
        "description": "For regulated industries (0.30-0.50 factor), 10K+ LOC only",
        "min_loc": 10000,
    },
    "function_points": {
        "id": "function_points",
        "name": "Function Points",
        "formula": "Days = (LOC ÷ 50) × 0.25 × 0.5 × complexity",
        "source": "ISO/IEC 20926",
        "confidence": "Medium",
        "description": "Based on functional requirements estimation",
    },
}


# =============================================================================
# PERT FORMULAS
# =============================================================================

PERT_FORMULAS = {
    "expected": "(O + 4×M + P) / 6",
    "std_dev": "(P - O) / 6",
    "variance": "σ²",
    "confidence_68": "μ ± 1σ",
    "confidence_95": "μ ± 2σ",
    "confidence_99": "μ ± 3σ",
}


# =============================================================================
# LOC CONVERSION
# =============================================================================

LOC_CONVERSION = {
    "words_per_loc": 10,
    "words_per_page": 300,
}


# =============================================================================
# ESTIMATION FUNCTIONS
# =============================================================================

def estimate_cocomo_modern(
    loc: int,
    tech_debt_score: int = 10,
    team_experience: str = "nominal",
) -> Dict[str, Any]:
    """
    COCOMO II Modern Calibration estimate.

    Formula: Effort (PM) = 0.5 × (KLOC)^0.85 × EAF
    """
    kloc = max(loc / 1000, 0.1)
    c = COCOMO_CONSTANTS

    # Effort multiplier from team experience
    eaf = EFFORT_MULTIPLIERS["team_experience"].get(team_experience, 1.0)

    # Tech debt affects exponent
    debt_adjust = 0
    for (low, high), mult in TECH_DEBT_MULTIPLIERS.items():
        if low <= tech_debt_score <= high:
            debt_adjust = (mult - 1) * 0.1
            break

    exponent = c["b"] + debt_adjust

    # Base effort
    effort_pm = c["a"] * (kloc ** exponent) * eaf
    hours = effort_pm * c["hours_per_pm"]

    # Schedule
    schedule_months = c["c"] * (effort_pm ** c["d"])
    team_size = effort_pm / max(schedule_months, 0.5)

    return {
        "methodology": "COCOMO II (Modern)",
        "formula": f"Effort = {c['a']} × ({kloc:.2f})^{exponent:.2f} × {eaf} = {effort_pm:.2f} PM",
        "inputs": {
            "loc": loc,
            "kloc": round(kloc, 2),
            "tech_debt_score": tech_debt_score,
            "team_experience": team_experience,
        },
        "effort_pm": round(effort_pm, 2),
        "schedule_months": round(schedule_months, 1),
        "team_size": round(team_size, 1),
        "hours": {
            "min": round(hours * 0.8),
            "typical": round(hours),
            "max": round(hours * 1.2),
        },
    }


def estimate_methodology(
    methodology: str,
    loc: int,
    complexity: float = 1.5,
    hourly_rate: float = 35,
) -> Dict[str, Any]:
    """Calculate single methodology estimate."""
    if methodology not in METHODOLOGIES:
        return {"error": f"Unknown methodology: {methodology}"}

    kloc = loc / 1000
    words = loc * LOC_CONVERSION["words_per_loc"]
    pages = math.ceil(words / LOC_CONVERSION["words_per_page"])

    meta = METHODOLOGIES[methodology]

    if methodology == "cocomo":
        effort_pm = 0.5 * (kloc ** 0.85) * complexity
        hours = effort_pm * 160
        formula_calc = f"0.5 × ({kloc:.1f})^0.85 × {complexity} = {effort_pm:.2f} PM"
    elif methodology == "gartner":
        days = (words / 650) * complexity
        hours = days * 8
        formula_calc = f"{words:,} words ÷ 650 × {complexity} = {days:.1f} days"
    elif methodology == "ieee":
        days = (pages / 1.5) * complexity
        hours = days * 8
        formula_calc = f"{pages} pages ÷ 1.5 × {complexity} = {days:.1f} days"
    elif methodology == "microsoft":
        days = (words / 650) * complexity
        hours = days * 8
        formula_calc = f"{words:,} words ÷ 650 × {complexity} = {days:.1f} days"
    elif methodology == "google":
        hours = (pages * 4) * complexity
        formula_calc = f"{pages} pages × 4 × {complexity} = {hours:.0f} hours"
    elif methodology == "pmi":
        days = (pages * 0.25) * complexity
        hours = days * 8
        formula_calc = f"{pages} pages × 0.25 × {complexity} = {days:.1f} days"
    elif methodology == "sei_slim":
        if loc < 10000:
            return {"error": "SEI SLIM requires minimum 10,000 LOC", "loc": loc}
        days = 180 * 0.4 * complexity
        hours = days * 8
        formula_calc = f"180 × 0.4 × {complexity} = {days:.1f} days"
    elif methodology == "function_points":
        fp = loc / 50
        fp_doc = fp * 0.25
        days = fp_doc * 0.5 * complexity
        hours = days * 8
        formula_calc = f"({loc} ÷ 50) × 0.25 × 0.5 × {complexity} = {days:.1f} days"
    else:
        hours = loc / 10
        formula_calc = "Unknown methodology"

    return {
        "id": methodology,
        "name": meta["name"],
        "hours": round(hours, 1),
        "days": round(hours / 8, 1),
        "cost": round(hours * hourly_rate, 2),
        "confidence": meta["confidence"],
        "formula": meta["formula"],
        "formula_calculated": formula_calc,
        "source": meta["source"],
    }


def estimate_all_methodologies(
    loc: int,
    complexity: float = 1.5,
    hourly_rate: float = 35,
) -> Dict[str, Any]:
    """Estimate using all 8 methodologies."""
    results = []

    for mid in METHODOLOGIES.keys():
        if mid == "sei_slim" and loc < 10000:
            continue
        result = estimate_methodology(mid, loc, complexity, hourly_rate)
        if "error" not in result:
            results.append(result)

    all_hours = [r["hours"] for r in results]
    all_costs = [r["cost"] for r in results]

    # PERT from methodology results
    if len(all_hours) >= 3:
        pert = calculate_pert(min(all_hours), sum(all_hours)/len(all_hours), max(all_hours))
    else:
        pert = None

    return {
        "input": {"loc": loc, "complexity": complexity, "hourly_rate": hourly_rate},
        "methodologies": results,
        "summary": {
            "average_hours": round(sum(all_hours) / len(all_hours), 1) if all_hours else 0,
            "average_cost": round(sum(all_costs) / len(all_costs), 2) if all_costs else 0,
            "min_cost": round(min(all_costs), 2) if all_costs else 0,
            "max_cost": round(max(all_costs), 2) if all_costs else 0,
            "methodologies_count": len(results),
        },
        "pert": pert,
    }


def calculate_pert(
    optimistic: float,
    most_likely: float,
    pessimistic: float,
) -> Dict[str, Any]:
    """
    PERT 3-point estimation.

    Expected = (O + 4×M + P) / 6
    σ = (P - O) / 6
    """
    expected = (optimistic + 4 * most_likely + pessimistic) / 6
    std_dev = (pessimistic - optimistic) / 6
    variance = std_dev ** 2

    return {
        "inputs": {
            "optimistic": round(optimistic, 1),
            "most_likely": round(most_likely, 1),
            "pessimistic": round(pessimistic, 1),
        },
        "expected": round(expected, 1),
        "expected_days": round(expected / 8, 1),
        "standard_deviation": round(std_dev, 2),
        "variance": round(variance, 2),
        "confidence_68": {"min": round(expected - std_dev, 1), "max": round(expected + std_dev, 1)},
        "confidence_95": {"min": round(expected - 2*std_dev, 1), "max": round(expected + 2*std_dev, 1)},
        "confidence_99": {"min": round(expected - 3*std_dev, 1), "max": round(expected + 3*std_dev, 1)},
        "formulas": PERT_FORMULAS,
    }


def estimate_ai_efficiency(
    loc: int,
    hourly_rate: float = 35,
    complexity: float = 1.5,
) -> Dict[str, Any]:
    """
    Compare Pure Human vs AI-Assisted vs Hybrid.

    Productivity rates (hrs/KLOC):
    - Pure Human: 25
    - AI-Assisted: 8
    - Hybrid: 6.5
    """
    kloc = loc / 1000

    # Calculate hours
    pure_human_hrs = kloc * AI_PRODUCTIVITY["pure_human"] * complexity
    ai_assisted_hrs = kloc * AI_PRODUCTIVITY["ai_assisted"] * complexity
    hybrid_hrs = kloc * AI_PRODUCTIVITY["hybrid"] * complexity

    # AI subscription cost
    project_months = max(1, pure_human_hrs / 160)
    ai_cost = 20 * project_months * 0.3  # $20/month, 30% usage

    # Total costs
    pure_human_cost = pure_human_hrs * hourly_rate
    ai_assisted_cost = (ai_assisted_hrs * hourly_rate) + ai_cost
    hybrid_cost = (hybrid_hrs * hourly_rate) + (ai_cost * 1.2)

    # Savings
    savings_ai = pure_human_cost - ai_assisted_cost
    savings_hybrid = pure_human_cost - hybrid_cost
    savings_pct_ai = (savings_ai / pure_human_cost) * 100 if pure_human_cost > 0 else 0
    savings_pct_hybrid = (savings_hybrid / pure_human_cost) * 100 if pure_human_cost > 0 else 0

    return {
        "inputs": {"loc": loc, "hourly_rate": hourly_rate, "complexity": complexity},
        "productivity_rates": AI_PRODUCTIVITY,
        "approaches": {
            "pure_human": {
                "hours": round(pure_human_hrs, 1),
                "days": round(pure_human_hrs / 8, 1),
                "cost": round(pure_human_cost, 2),
                "productivity": f"{AI_PRODUCTIVITY['pure_human']} hrs/KLOC",
            },
            "ai_assisted": {
                "hours": round(ai_assisted_hrs, 1),
                "days": round(ai_assisted_hrs / 8, 1),
                "labor_cost": round(ai_assisted_hrs * hourly_rate, 2),
                "ai_cost": round(ai_cost, 2),
                "total_cost": round(ai_assisted_cost, 2),
                "productivity": f"{AI_PRODUCTIVITY['ai_assisted']} hrs/KLOC",
                "speedup": f"{AI_PRODUCTIVITY['pure_human']/AI_PRODUCTIVITY['ai_assisted']:.1f}x faster",
            },
            "hybrid": {
                "hours": round(hybrid_hrs, 1),
                "days": round(hybrid_hrs / 8, 1),
                "labor_cost": round(hybrid_hrs * hourly_rate, 2),
                "ai_cost": round(ai_cost * 1.2, 2),
                "total_cost": round(hybrid_cost, 2),
                "productivity": f"{AI_PRODUCTIVITY['hybrid']} hrs/KLOC",
                "speedup": f"{AI_PRODUCTIVITY['pure_human']/AI_PRODUCTIVITY['hybrid']:.1f}x faster",
            },
        },
        "savings": {
            "ai_vs_human_dollars": round(savings_ai, 2),
            "hybrid_vs_human_dollars": round(savings_hybrid, 2),
            "ai_vs_human_percent": round(savings_pct_ai, 1),
            "hybrid_vs_human_percent": round(savings_pct_hybrid, 1),
        },
    }


def calculate_roi(
    investment_cost: float,
    annual_support_savings: float = 0,
    annual_training_savings: float = 0,
    annual_efficiency_gain: float = 0,
    annual_risk_reduction: float = 0,
    maintenance_percent: float = 20,
) -> Dict[str, Any]:
    """
    ROI analysis.

    Formulas:
    - ROI 1yr = (net_annual - investment) / investment × 100
    - ROI 3yr = (net_annual × 3 - investment) / investment × 100
    - Payback = investment / (net_annual / 12) months
    - NPV 3yr = net_annual × 3 - investment
    """
    annual_maintenance = investment_cost * (maintenance_percent / 100)
    annual_benefits = (
        annual_support_savings +
        annual_training_savings +
        annual_efficiency_gain +
        annual_risk_reduction
    )
    net_annual = annual_benefits - annual_maintenance

    roi_1yr = ((net_annual - investment_cost) / investment_cost) * 100 if investment_cost > 0 else 0
    roi_3yr = ((net_annual * 3 - investment_cost) / investment_cost) * 100 if investment_cost > 0 else 0
    payback_months = (investment_cost / (net_annual / 12)) if net_annual > 0 else float('inf')
    npv_3yr = (net_annual * 3) - investment_cost

    return {
        "inputs": {
            "investment_cost": investment_cost,
            "annual_support_savings": annual_support_savings,
            "annual_training_savings": annual_training_savings,
            "annual_efficiency_gain": annual_efficiency_gain,
            "annual_risk_reduction": annual_risk_reduction,
            "maintenance_percent": maintenance_percent,
        },
        "investment": {
            "total": investment_cost,
            "annual_maintenance": round(annual_maintenance, 2),
        },
        "benefits": {
            "annual_gross": round(annual_benefits, 2),
            "annual_net": round(net_annual, 2),
        },
        "metrics": {
            "roi_1yr_percent": round(roi_1yr, 1),
            "roi_3yr_percent": round(roi_3yr, 1),
            "payback_months": round(payback_months, 1) if payback_months != float('inf') else "N/A",
            "npv_3yr": round(npv_3yr, 2),
        },
    }


def get_regional_cost(
    hours: float,
    region: str = "eu",
) -> Dict[str, Any]:
    """Calculate cost for specific region."""
    rates = REGIONAL_RATES.get(region, REGIONAL_RATES["eu"])

    return {
        "region": region,
        "region_name": rates["name"],
        "hours": round(hours),
        "currency": rates["currency"],
        "symbol": rates["symbol"],
        "cost": {
            "min": round(hours * rates["rates"]["junior"]),
            "typical": round(hours * rates["rates"]["middle"]),
            "max": round(hours * rates["rates"]["senior"]),
        },
        "rates": rates["rates"],
    }


def get_all_regional_costs(hours: float) -> Dict[str, Any]:
    """Calculate cost for all 8 regions."""
    results = {}
    for region in REGIONAL_RATES.keys():
        results[region] = get_regional_cost(hours, region)
    return {"hours": round(hours), "regions": results}


def get_complexity(loc: int) -> str:
    """Determine complexity based on LOC."""
    for level, thresholds in COMPLEXITY_THRESHOLDS.items():
        if thresholds["min"] <= loc < thresholds["max"]:
            return level
    return "XL"


def get_tech_debt_multiplier(score: int) -> float:
    """Get multiplier based on tech debt score."""
    for (low, high), mult in TECH_DEBT_MULTIPLIERS.items():
        if low <= score <= high:
            return mult
    return 1.0


def get_all_formulas() -> Dict[str, Any]:
    """Get all formulas documentation."""
    return {
        "cocomo_ii": {
            "effort": "Effort (PM) = a × (KLOC)^b × EAF",
            "schedule": "Duration (months) = c × (Effort)^d",
            "hours": "Hours = Effort × 160",
            "constants_modern": COCOMO_CONSTANTS,
            "constants_classic": COCOMO_CLASSIC,
            "multipliers": EFFORT_MULTIPLIERS,
        },
        "pert": PERT_FORMULAS,
        "methodologies": {m: {"formula": d["formula"], "source": d["source"]} for m, d in METHODOLOGIES.items()},
        "ai_efficiency": {
            "pure_human": f"{AI_PRODUCTIVITY['pure_human']} hrs/KLOC",
            "ai_assisted": f"{AI_PRODUCTIVITY['ai_assisted']} hrs/KLOC",
            "hybrid": f"{AI_PRODUCTIVITY['hybrid']} hrs/KLOC",
        },
        "roi": {
            "roi_1yr": "(net_annual - investment) / investment × 100",
            "roi_3yr": "(net_annual × 3 - investment) / investment × 100",
            "payback": "investment / (net_annual / 12) months",
            "npv_3yr": "net_annual × 3 - investment",
        },
    }


def get_all_constants() -> Dict[str, Any]:
    """Get all constants."""
    return {
        "cocomo": COCOMO_CONSTANTS,
        "cocomo_classic": COCOMO_CLASSIC,
        "effort_multipliers": EFFORT_MULTIPLIERS,
        "tech_debt_multipliers": {f"{k[0]}-{k[1]}": v for k, v in TECH_DEBT_MULTIPLIERS.items()},
        "activity_ratios": ACTIVITY_RATIOS,
        "complexity_thresholds": COMPLEXITY_THRESHOLDS,
        "product_stages": PRODUCT_STAGES,
        "ai_productivity": AI_PRODUCTIVITY,
        "loc_conversion": LOC_CONVERSION,
        "regional_rates": REGIONAL_RATES,
    }
