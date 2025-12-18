"""
Settings Service for Audit Platform

Provides:
- Regional rates configuration
- Estimation parameters management
- Methodology settings
- Anti-hallucination validation layer
- Audit trail for changes
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from copy import deepcopy

from pydantic import BaseModel, Field, validator
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

# =============================================================================
# STORAGE PATH
# =============================================================================

SETTINGS_DIR = Path(__file__).parent.parent.parent / "data" / "settings"
SETTINGS_DIR.mkdir(parents=True, exist_ok=True)

RATES_FILE = SETTINGS_DIR / "regional_rates.json"
PARAMS_FILE = SETTINGS_DIR / "estimation_params.json"
METHODS_FILE = SETTINGS_DIR / "methodologies.json"
AUDIT_LOG_FILE = SETTINGS_DIR / "audit_log.json"


# =============================================================================
# VALIDATION BOUNDS (Anti-Hallucination Protection)
# =============================================================================

class ValidationBounds:
    """
    Hard limits to prevent unrealistic values.
    Based on industry research and historical data.
    """

    # Regional Rates (USD/hour)
    RATE_MIN = 5       # Absolute minimum (developing regions)
    RATE_MAX = 300     # Absolute maximum (specialized consultants)

    # COCOMO II Coefficients
    COCOMO_A_MIN = 0.1      # Very efficient teams
    COCOMO_A_MAX = 5.0      # Complex embedded systems
    COCOMO_B_MIN = 0.5      # Near-linear scaling
    COCOMO_B_MAX = 1.5      # Strong diseconomy of scale

    # Hours per Person-Month
    HOURS_PM_MIN = 120      # Part-time / holidays
    HOURS_PM_MAX = 200      # Overtime allowed

    # Complexity Multipliers
    COMPLEXITY_MIN = 0.3    # Very simple project
    COMPLEXITY_MAX = 5.0    # Extremely complex

    # LOC Thresholds
    LOC_MIN = 100           # Minimum meaningful project
    LOC_MAX = 10_000_000    # Very large codebase

    # PERT Validation
    PERT_RATIO_MAX = 10     # Pessimistic / Optimistic max ratio

    # AI Productivity (hrs/KLOC)
    AI_PROD_MIN = 1         # Future AI (very optimistic)
    AI_PROD_MAX = 100       # Very slow development

    # Overhead Multipliers
    OVERHEAD_MIN = 1.0      # No overhead
    OVERHEAD_MAX = 2.0      # 100% overhead

    @classmethod
    def validate_rate(cls, value: float, field_name: str = "rate") -> Tuple[bool, str]:
        """Validate hourly rate."""
        if value < cls.RATE_MIN:
            return False, f"{field_name}: ${value}/hr is below minimum ${cls.RATE_MIN}/hr"
        if value > cls.RATE_MAX:
            return False, f"{field_name}: ${value}/hr exceeds maximum ${cls.RATE_MAX}/hr"
        return True, ""

    @classmethod
    def validate_cocomo(cls, a: float, b: float) -> Tuple[bool, str]:
        """Validate COCOMO coefficients."""
        if not (cls.COCOMO_A_MIN <= a <= cls.COCOMO_A_MAX):
            return False, f"COCOMO 'a' coefficient {a} outside bounds [{cls.COCOMO_A_MIN}, {cls.COCOMO_A_MAX}]"
        if not (cls.COCOMO_B_MIN <= b <= cls.COCOMO_B_MAX):
            return False, f"COCOMO 'b' exponent {b} outside bounds [{cls.COCOMO_B_MIN}, {cls.COCOMO_B_MAX}]"
        return True, ""

    @classmethod
    def validate_pert(cls, optimistic: float, most_likely: float, pessimistic: float) -> Tuple[bool, str]:
        """Validate PERT inputs."""
        if optimistic <= 0:
            return False, "Optimistic hours must be positive"
        if optimistic > most_likely:
            return False, "Optimistic cannot exceed most likely"
        if most_likely > pessimistic:
            return False, "Most likely cannot exceed pessimistic"
        ratio = pessimistic / optimistic if optimistic > 0 else float('inf')
        if ratio > cls.PERT_RATIO_MAX:
            return False, f"PERT ratio {ratio:.1f}x exceeds maximum {cls.PERT_RATIO_MAX}x (unrealistic spread)"
        return True, ""

    @classmethod
    def validate_estimate(cls, hours: float, loc: int, methodology: str = "unknown") -> Tuple[bool, str]:
        """
        Cross-validate estimate result against known bounds.

        Industry benchmarks:
        - Simple CRUD: 5-15 hrs/KLOC
        - Complex business: 20-40 hrs/KLOC
        - Embedded/Critical: 50-100+ hrs/KLOC
        """
        if loc <= 0:
            return False, "LOC must be positive"

        kloc = loc / 1000
        hrs_per_kloc = hours / kloc if kloc > 0 else 0

        # Reasonable bounds: 2-200 hrs/KLOC
        if hrs_per_kloc < 2:
            return False, f"{methodology}: {hrs_per_kloc:.1f} hrs/KLOC is unrealistically low (min: 2)"
        if hrs_per_kloc > 200:
            return False, f"{methodology}: {hrs_per_kloc:.1f} hrs/KLOC is unrealistically high (max: 200)"

        return True, ""


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class RateLevel(BaseModel):
    """Individual rate level."""
    junior: float = Field(..., ge=5, le=300)
    middle: float = Field(..., ge=5, le=300)
    senior: float = Field(..., ge=5, le=300)
    typical: float = Field(..., ge=5, le=300)

    @validator('middle')
    def middle_gte_junior(cls, v, values):
        if 'junior' in values and v < values['junior']:
            raise ValueError('middle rate must be >= junior rate')
        return v

    @validator('senior')
    def senior_gte_middle(cls, v, values):
        if 'middle' in values and v < values['middle']:
            raise ValueError('senior rate must be >= middle rate')
        return v


class RegionalRate(BaseModel):
    """Regional rate configuration."""
    name: str
    currency: str = "USD"
    symbol: str = "$"
    rates: RateLevel
    overhead: float = Field(1.0, ge=1.0, le=2.0)


class RegionalRatesConfig(BaseModel):
    """All regional rates."""
    regions: Dict[str, RegionalRate]
    updated_at: Optional[str] = None
    updated_by: Optional[str] = None


class CocomoParams(BaseModel):
    """COCOMO II parameters."""
    a: float = Field(0.5, ge=0.1, le=5.0, description="Effort coefficient")
    b: float = Field(0.85, ge=0.5, le=1.5, description="Base exponent")
    c: float = Field(2.0, ge=1.0, le=4.0, description="Schedule coefficient")
    d: float = Field(0.35, ge=0.2, le=0.5, description="Schedule exponent")
    hours_per_pm: int = Field(160, ge=120, le=200, description="Hours per person-month")


class EffortMultipliers(BaseModel):
    """Effort adjustment factors."""
    team_experience: Dict[str, float] = Field(
        default={"low": 1.3, "nominal": 1.0, "high": 0.8}
    )
    process_maturity: Dict[str, float] = Field(
        default={"low": 1.2, "nominal": 1.0, "high": 0.85}
    )
    tool_support: Dict[str, float] = Field(
        default={"low": 1.15, "nominal": 1.0, "high": 0.9}
    )
    requirements_volatility: Dict[str, float] = Field(
        default={"low": 0.9, "nominal": 1.0, "high": 1.25}
    )


class AIProductivity(BaseModel):
    """AI productivity rates (hrs/KLOC)."""
    pure_human: float = Field(25, ge=1, le=100)
    ai_assisted: float = Field(8, ge=1, le=50)
    hybrid: float = Field(6.5, ge=1, le=40)

    @validator('ai_assisted')
    def ai_less_than_human(cls, v, values):
        if 'pure_human' in values and v >= values['pure_human']:
            raise ValueError('AI-assisted must be faster than pure human')
        return v

    @validator('hybrid')
    def hybrid_less_than_ai(cls, v, values):
        if 'ai_assisted' in values and v >= values['ai_assisted']:
            raise ValueError('Hybrid must be faster than AI-assisted alone')
        return v


class EstimationParams(BaseModel):
    """All estimation parameters."""
    cocomo: CocomoParams = Field(default_factory=CocomoParams)
    effort_multipliers: EffortMultipliers = Field(default_factory=EffortMultipliers)
    ai_productivity: AIProductivity = Field(default_factory=AIProductivity)
    updated_at: Optional[str] = None
    updated_by: Optional[str] = None


class MethodologyConfig(BaseModel):
    """Single methodology configuration."""
    id: str
    name: str
    formula: str
    source: str
    confidence: str = "Medium"
    enabled: bool = True
    min_loc: Optional[int] = None
    custom_params: Dict[str, Any] = Field(default_factory=dict)


class MethodologiesConfig(BaseModel):
    """All methodologies configuration."""
    methodologies: Dict[str, MethodologyConfig]
    updated_at: Optional[str] = None
    updated_by: Optional[str] = None


class AuditLogEntry(BaseModel):
    """Audit log entry for tracking changes."""
    timestamp: str
    action: str
    section: str
    user: str = "system"
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    validation_passed: bool = True
    validation_message: str = ""


# =============================================================================
# SETTINGS SERVICE
# =============================================================================

class SettingsService:
    """
    Service for managing estimation settings with:
    - Validation layer (anti-hallucination)
    - Audit trail
    - Rollback capability
    """

    def __init__(self):
        self._ensure_defaults()

    def _ensure_defaults(self):
        """Create default settings files if they don't exist."""
        if not RATES_FILE.exists():
            self._save_default_rates()
        if not PARAMS_FILE.exists():
            self._save_default_params()
        if not METHODS_FILE.exists():
            self._save_default_methods()
        if not AUDIT_LOG_FILE.exists():
            self._init_audit_log()

    def _save_default_rates(self):
        """Save default regional rates."""
        from executors.cost_estimator.formulas import REGIONAL_RATES

        rates = {
            "regions": {
                key: {
                    "name": val["name"],
                    "currency": val["currency"],
                    "symbol": val["symbol"],
                    "rates": val["rates"],
                    "overhead": val["overhead"]
                }
                for key, val in REGIONAL_RATES.items()
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "updated_by": "system"
        }

        with open(RATES_FILE, 'w') as f:
            json.dump(rates, f, indent=2)

    def _save_default_params(self):
        """Save default estimation parameters."""
        from executors.cost_estimator.formulas import (
            COCOMO_CONSTANTS, EFFORT_MULTIPLIERS, AI_PRODUCTIVITY
        )

        params = {
            "cocomo": COCOMO_CONSTANTS,
            "effort_multipliers": EFFORT_MULTIPLIERS,
            "ai_productivity": AI_PRODUCTIVITY,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "updated_by": "system"
        }

        with open(PARAMS_FILE, 'w') as f:
            json.dump(params, f, indent=2)

    def _save_default_methods(self):
        """Save default methodologies."""
        from executors.cost_estimator.formulas import METHODOLOGIES

        methods = {
            "methodologies": {
                key: {
                    **val,
                    "enabled": True,
                    "custom_params": {}
                }
                for key, val in METHODOLOGIES.items()
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "updated_by": "system"
        }

        with open(METHODS_FILE, 'w') as f:
            json.dump(methods, f, indent=2)

    def _init_audit_log(self):
        """Initialize audit log."""
        with open(AUDIT_LOG_FILE, 'w') as f:
            json.dump({"entries": []}, f, indent=2)

    def _log_change(
        self,
        action: str,
        section: str,
        old_value: Any = None,
        new_value: Any = None,
        user: str = "system",
        validation_passed: bool = True,
        validation_message: str = ""
    ):
        """Log a change to audit trail."""
        try:
            with open(AUDIT_LOG_FILE, 'r') as f:
                log = json.load(f)
        except:
            log = {"entries": []}

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "section": section,
            "user": user,
            "old_value": old_value,
            "new_value": new_value,
            "validation_passed": validation_passed,
            "validation_message": validation_message
        }

        log["entries"].append(entry)

        # Keep last 1000 entries
        if len(log["entries"]) > 1000:
            log["entries"] = log["entries"][-1000:]

        with open(AUDIT_LOG_FILE, 'w') as f:
            json.dump(log, f, indent=2)

    # -------------------------------------------------------------------------
    # Regional Rates
    # -------------------------------------------------------------------------

    def get_regional_rates(self) -> Dict[str, Any]:
        """Get current regional rates."""
        try:
            with open(RATES_FILE, 'r') as f:
                return json.load(f)
        except:
            self._save_default_rates()
            with open(RATES_FILE, 'r') as f:
                return json.load(f)

    def update_regional_rate(
        self,
        region_id: str,
        rate_data: Dict[str, Any],
        user: str = "api"
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Update a single regional rate with validation.

        Returns: (success, message, updated_data)
        """
        rates = self.get_regional_rates()

        if region_id not in rates["regions"]:
            return False, f"Unknown region: {region_id}", {}

        old_value = deepcopy(rates["regions"][region_id])

        # Validate rates
        if "rates" in rate_data:
            for level in ["junior", "middle", "senior", "typical"]:
                if level in rate_data["rates"]:
                    valid, msg = ValidationBounds.validate_rate(
                        rate_data["rates"][level],
                        f"{region_id}.{level}"
                    )
                    if not valid:
                        self._log_change(
                            "update_rate_failed",
                            f"regional_rates.{region_id}",
                            old_value, rate_data, user,
                            validation_passed=False,
                            validation_message=msg
                        )
                        return False, msg, {}

        # Validate overhead
        if "overhead" in rate_data:
            if not (1.0 <= rate_data["overhead"] <= 2.0):
                msg = f"Overhead {rate_data['overhead']} outside bounds [1.0, 2.0]"
                self._log_change(
                    "update_rate_failed",
                    f"regional_rates.{region_id}",
                    old_value, rate_data, user,
                    validation_passed=False,
                    validation_message=msg
                )
                return False, msg, {}

        # Apply update
        rates["regions"][region_id].update(rate_data)
        rates["updated_at"] = datetime.now(timezone.utc).isoformat()
        rates["updated_by"] = user

        with open(RATES_FILE, 'w') as f:
            json.dump(rates, f, indent=2)

        self._log_change(
            "update_rate",
            f"regional_rates.{region_id}",
            old_value,
            rates["regions"][region_id],
            user
        )

        return True, "Rate updated successfully", rates["regions"][region_id]

    def add_regional_rate(
        self,
        region_id: str,
        rate_data: Dict[str, Any],
        user: str = "api"
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Add a new regional rate."""
        rates = self.get_regional_rates()

        if region_id in rates["regions"]:
            return False, f"Region {region_id} already exists", {}

        # Validate required fields
        required = ["name", "rates"]
        for field in required:
            if field not in rate_data:
                return False, f"Missing required field: {field}", {}

        # Validate rates
        for level in ["junior", "middle", "senior", "typical"]:
            if level not in rate_data["rates"]:
                return False, f"Missing rate level: {level}", {}

            valid, msg = ValidationBounds.validate_rate(
                rate_data["rates"][level],
                f"{region_id}.{level}"
            )
            if not valid:
                return False, msg, {}

        # Set defaults
        rate_data.setdefault("currency", "USD")
        rate_data.setdefault("symbol", "$")
        rate_data.setdefault("overhead", 1.2)

        rates["regions"][region_id] = rate_data
        rates["updated_at"] = datetime.now(timezone.utc).isoformat()
        rates["updated_by"] = user

        with open(RATES_FILE, 'w') as f:
            json.dump(rates, f, indent=2)

        self._log_change("add_rate", f"regional_rates.{region_id}", None, rate_data, user)

        return True, "Region added successfully", rate_data

    # -------------------------------------------------------------------------
    # Estimation Parameters
    # -------------------------------------------------------------------------

    def get_estimation_params(self) -> Dict[str, Any]:
        """Get current estimation parameters."""
        try:
            with open(PARAMS_FILE, 'r') as f:
                return json.load(f)
        except:
            self._save_default_params()
            with open(PARAMS_FILE, 'r') as f:
                return json.load(f)

    def update_cocomo_params(
        self,
        params: Dict[str, Any],
        user: str = "api"
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Update COCOMO parameters with validation."""
        current = self.get_estimation_params()
        old_cocomo = deepcopy(current.get("cocomo", {}))

        new_cocomo = {**old_cocomo, **params}

        # Validate
        valid, msg = ValidationBounds.validate_cocomo(
            new_cocomo.get("a", 0.5),
            new_cocomo.get("b", 0.85)
        )
        if not valid:
            self._log_change(
                "update_cocomo_failed",
                "estimation_params.cocomo",
                old_cocomo, params, user,
                validation_passed=False,
                validation_message=msg
            )
            return False, msg, {}

        # Validate hours_per_pm
        hours_pm = new_cocomo.get("hours_per_pm", 160)
        if not (ValidationBounds.HOURS_PM_MIN <= hours_pm <= ValidationBounds.HOURS_PM_MAX):
            msg = f"hours_per_pm {hours_pm} outside bounds [{ValidationBounds.HOURS_PM_MIN}, {ValidationBounds.HOURS_PM_MAX}]"
            return False, msg, {}

        current["cocomo"] = new_cocomo
        current["updated_at"] = datetime.now(timezone.utc).isoformat()
        current["updated_by"] = user

        with open(PARAMS_FILE, 'w') as f:
            json.dump(current, f, indent=2)

        self._log_change("update_cocomo", "estimation_params.cocomo", old_cocomo, new_cocomo, user)

        return True, "COCOMO parameters updated", new_cocomo

    def update_ai_productivity(
        self,
        params: Dict[str, float],
        user: str = "api"
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Update AI productivity rates."""
        current = self.get_estimation_params()
        old_ai = deepcopy(current.get("ai_productivity", {}))

        new_ai = {**old_ai, **params}

        # Validate bounds
        for key in ["pure_human", "ai_assisted", "hybrid"]:
            if key in new_ai:
                val = new_ai[key]
                if not (ValidationBounds.AI_PROD_MIN <= val <= ValidationBounds.AI_PROD_MAX):
                    return False, f"{key} = {val} hrs/KLOC outside bounds [{ValidationBounds.AI_PROD_MIN}, {ValidationBounds.AI_PROD_MAX}]", {}

        # Validate logical order
        if new_ai.get("ai_assisted", 8) >= new_ai.get("pure_human", 25):
            return False, "AI-assisted must be faster (lower hrs/KLOC) than pure human", {}

        if new_ai.get("hybrid", 6.5) >= new_ai.get("ai_assisted", 8):
            return False, "Hybrid must be faster than AI-assisted", {}

        current["ai_productivity"] = new_ai
        current["updated_at"] = datetime.now(timezone.utc).isoformat()
        current["updated_by"] = user

        with open(PARAMS_FILE, 'w') as f:
            json.dump(current, f, indent=2)

        self._log_change("update_ai_productivity", "estimation_params.ai_productivity", old_ai, new_ai, user)

        return True, "AI productivity updated", new_ai

    # -------------------------------------------------------------------------
    # Methodologies
    # -------------------------------------------------------------------------

    def get_methodologies(self) -> Dict[str, Any]:
        """Get methodologies configuration."""
        try:
            with open(METHODS_FILE, 'r') as f:
                return json.load(f)
        except:
            self._save_default_methods()
            with open(METHODS_FILE, 'r') as f:
                return json.load(f)

    def toggle_methodology(
        self,
        methodology_id: str,
        enabled: bool,
        user: str = "api"
    ) -> Tuple[bool, str]:
        """Enable/disable a methodology."""
        methods = self.get_methodologies()

        if methodology_id not in methods["methodologies"]:
            return False, f"Unknown methodology: {methodology_id}"

        old_enabled = methods["methodologies"][methodology_id].get("enabled", True)
        methods["methodologies"][methodology_id]["enabled"] = enabled
        methods["updated_at"] = datetime.now(timezone.utc).isoformat()
        methods["updated_by"] = user

        with open(METHODS_FILE, 'w') as f:
            json.dump(methods, f, indent=2)

        self._log_change(
            "toggle_methodology",
            f"methodologies.{methodology_id}",
            {"enabled": old_enabled},
            {"enabled": enabled},
            user
        )

        return True, f"Methodology {methodology_id} {'enabled' if enabled else 'disabled'}"

    # -------------------------------------------------------------------------
    # Validation Service
    # -------------------------------------------------------------------------

    def validate_estimate_result(
        self,
        hours: float,
        loc: int,
        cost: float,
        methodology: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Validate an estimation result against known bounds.

        Returns validation result with warnings/errors.
        """
        result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "confidence": "high"
        }

        # Validate hours per KLOC
        valid, msg = ValidationBounds.validate_estimate(hours, loc, methodology)
        if not valid:
            result["valid"] = False
            result["errors"].append(msg)
            result["confidence"] = "rejected"
            return result

        kloc = loc / 1000
        hrs_per_kloc = hours / kloc if kloc > 0 else 0

        # Warning zones
        if hrs_per_kloc < 5:
            result["warnings"].append(f"Very low estimate ({hrs_per_kloc:.1f} hrs/KLOC). Verify AI usage assumptions.")
            result["confidence"] = "low"
        elif hrs_per_kloc > 80:
            result["warnings"].append(f"High estimate ({hrs_per_kloc:.1f} hrs/KLOC). Suitable for embedded/safety-critical systems only.")
            result["confidence"] = "medium"

        # Validate cost against hours (detect rate anomalies)
        if hours > 0:
            implied_rate = cost / hours
            if implied_rate < ValidationBounds.RATE_MIN:
                result["warnings"].append(f"Implied rate ${implied_rate:.0f}/hr is below minimum. Check regional rate.")
            elif implied_rate > ValidationBounds.RATE_MAX:
                result["warnings"].append(f"Implied rate ${implied_rate:.0f}/hr is above maximum. Verify cost calculation.")

        return result

    def validate_pert_inputs(
        self,
        optimistic: float,
        most_likely: float,
        pessimistic: float
    ) -> Dict[str, Any]:
        """Validate PERT inputs."""
        valid, msg = ValidationBounds.validate_pert(optimistic, most_likely, pessimistic)

        if not valid:
            return {
                "valid": False,
                "error": msg,
                "suggestion": "PERT requires: 0 < Optimistic ≤ Most Likely ≤ Pessimistic, with max 10x spread"
            }

        return {"valid": True}

    # -------------------------------------------------------------------------
    # Audit Trail
    # -------------------------------------------------------------------------

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit log entries."""
        try:
            with open(AUDIT_LOG_FILE, 'r') as f:
                log = json.load(f)
            return log["entries"][-limit:]
        except:
            return []

    def get_settings_snapshot(self) -> Dict[str, Any]:
        """Get complete settings snapshot for backup/restore."""
        return {
            "regional_rates": self.get_regional_rates(),
            "estimation_params": self.get_estimation_params(),
            "methodologies": self.get_methodologies(),
            "exported_at": datetime.now(timezone.utc).isoformat()
        }

    def restore_settings(
        self,
        snapshot: Dict[str, Any],
        user: str = "api"
    ) -> Tuple[bool, str]:
        """Restore settings from snapshot."""
        try:
            if "regional_rates" in snapshot:
                with open(RATES_FILE, 'w') as f:
                    json.dump(snapshot["regional_rates"], f, indent=2)

            if "estimation_params" in snapshot:
                with open(PARAMS_FILE, 'w') as f:
                    json.dump(snapshot["estimation_params"], f, indent=2)

            if "methodologies" in snapshot:
                with open(METHODS_FILE, 'w') as f:
                    json.dump(snapshot["methodologies"], f, indent=2)

            self._log_change(
                "restore_settings",
                "all",
                None,
                {"restored_from": snapshot.get("exported_at", "unknown")},
                user
            )

            return True, "Settings restored successfully"
        except Exception as e:
            return False, f"Restore failed: {str(e)}"

    def reset_to_defaults(self, user: str = "api") -> Tuple[bool, str]:
        """Reset all settings to defaults."""
        self._save_default_rates()
        self._save_default_params()
        self._save_default_methods()

        self._log_change("reset_to_defaults", "all", None, None, user)

        return True, "Settings reset to defaults"


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_settings_service: Optional[SettingsService] = None

def get_settings_service() -> SettingsService:
    """Get singleton settings service instance."""
    global _settings_service
    if _settings_service is None:
        _settings_service = SettingsService()
    return _settings_service


# =============================================================================
# API ROUTER
# =============================================================================

router = APIRouter(prefix="/api/settings", tags=["settings"])


# --- Regional Rates ---

@router.get("/rates")
async def get_rates():
    """Get all regional rates."""
    return get_settings_service().get_regional_rates()


@router.get("/rates/{region_id}")
async def get_rate(region_id: str):
    """Get specific regional rate."""
    rates = get_settings_service().get_regional_rates()
    if region_id not in rates["regions"]:
        raise HTTPException(status_code=404, detail=f"Region not found: {region_id}")
    return rates["regions"][region_id]


@router.put("/rates/{region_id}")
async def update_rate(region_id: str, rate_data: Dict[str, Any]):
    """Update regional rate."""
    success, message, data = get_settings_service().update_regional_rate(region_id, rate_data)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"message": message, "data": data}


@router.post("/rates/{region_id}")
async def add_rate(region_id: str, rate_data: Dict[str, Any]):
    """Add new regional rate."""
    success, message, data = get_settings_service().add_regional_rate(region_id, rate_data)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"message": message, "data": data}


# --- Estimation Parameters ---

@router.get("/params")
async def get_params():
    """Get all estimation parameters."""
    return get_settings_service().get_estimation_params()


@router.put("/params/cocomo")
async def update_cocomo(params: Dict[str, Any]):
    """Update COCOMO parameters."""
    success, message, data = get_settings_service().update_cocomo_params(params)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"message": message, "data": data}


@router.put("/params/ai-productivity")
async def update_ai_prod(params: Dict[str, float]):
    """Update AI productivity rates."""
    success, message, data = get_settings_service().update_ai_productivity(params)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"message": message, "data": data}


# --- Methodologies ---

@router.get("/methodologies")
async def get_methodologies():
    """Get all methodologies."""
    return get_settings_service().get_methodologies()


@router.put("/methodologies/{method_id}/toggle")
async def toggle_methodology(method_id: str, enabled: bool):
    """Enable/disable methodology."""
    success, message = get_settings_service().toggle_methodology(method_id, enabled)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"message": message}


# --- Validation ---

@router.post("/validate/estimate")
async def validate_estimate(
    hours: float,
    loc: int,
    cost: float,
    methodology: str = "unknown"
):
    """Validate estimation result."""
    return get_settings_service().validate_estimate_result(hours, loc, cost, methodology)


@router.post("/validate/pert")
async def validate_pert(optimistic: float, most_likely: float, pessimistic: float):
    """Validate PERT inputs."""
    return get_settings_service().validate_pert_inputs(optimistic, most_likely, pessimistic)


# --- Audit & Backup ---

@router.get("/audit-log")
async def get_audit_log(limit: int = 100):
    """Get audit log."""
    return {"entries": get_settings_service().get_audit_log(limit)}


@router.get("/snapshot")
async def get_snapshot():
    """Export settings snapshot."""
    return get_settings_service().get_settings_snapshot()


@router.post("/restore")
async def restore_snapshot(snapshot: Dict[str, Any]):
    """Restore settings from snapshot."""
    success, message = get_settings_service().restore_settings(snapshot)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"message": message}


@router.post("/reset")
async def reset_defaults():
    """Reset all settings to defaults."""
    success, message = get_settings_service().reset_to_defaults()
    return {"message": message}


# --- Validation Bounds Info ---

@router.get("/validation-bounds")
async def get_validation_bounds():
    """Get validation bounds for reference."""
    return {
        "rates": {
            "min": ValidationBounds.RATE_MIN,
            "max": ValidationBounds.RATE_MAX
        },
        "cocomo": {
            "a": {"min": ValidationBounds.COCOMO_A_MIN, "max": ValidationBounds.COCOMO_A_MAX},
            "b": {"min": ValidationBounds.COCOMO_B_MIN, "max": ValidationBounds.COCOMO_B_MAX}
        },
        "hours_per_pm": {
            "min": ValidationBounds.HOURS_PM_MIN,
            "max": ValidationBounds.HOURS_PM_MAX
        },
        "complexity": {
            "min": ValidationBounds.COMPLEXITY_MIN,
            "max": ValidationBounds.COMPLEXITY_MAX
        },
        "ai_productivity": {
            "min": ValidationBounds.AI_PROD_MIN,
            "max": ValidationBounds.AI_PROD_MAX
        },
        "overhead": {
            "min": ValidationBounds.OVERHEAD_MIN,
            "max": ValidationBounds.OVERHEAD_MAX
        },
        "pert_ratio_max": ValidationBounds.PERT_RATIO_MAX,
        "estimate_hrs_per_kloc": {"min": 2, "max": 200}
    }
