"""
AuditWorkspace - Main workspace manager for audit-platform.

Creates and manages local audit environments:
├── .audit/
│   ├── config.json          # Workspace config
│   ├── session.json         # Active session
│   └── calibration.json     # Calibration data
├── analyses/
│   └── {id}/
│       ├── metrics.json     # Raw metrics
│       ├── scores.json      # Calculated scores
│       ├── cost.json        # Cost estimates
│       └── validation.json  # Validation results
├── reports/
│   ├── summary.md
│   ├── review.md
│   └── compliance.md
└── documents/
    ├── acceptance_act.md
    ├── invoice.md
    └── contracts/
"""
import json
import logging
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class WorkspaceConfig:
    """Workspace configuration."""
    workspace_id: str
    name: str
    created_at: str
    updated_at: str

    # Default settings
    default_region: str = "ua"
    default_profile: str = "standard"
    language: str = "en"

    # Client info (for documents)
    client_name: Optional[str] = None
    contractor_name: Optional[str] = None

    # Thresholds
    repo_health_threshold: int = 8
    tech_debt_threshold: int = 10

    # Validation settings
    enable_validation: bool = True
    strict_validation: bool = False  # Fail on warnings

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkspaceConfig":
        # Filter only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


@dataclass
class AnalysisRecord:
    """Record of a single analysis."""
    analysis_id: str
    repo_path: str
    created_at: str

    # Results
    repo_health: Optional[int] = None
    tech_debt: Optional[int] = None
    stage: Optional[str] = None
    complexity: Optional[str] = None
    loc: Optional[int] = None

    # Cost estimates
    cost_estimate: Optional[Dict[str, Any]] = None

    # Validation
    validation_passed: bool = True
    validation_warnings: List[str] = field(default_factory=list)

    # Metadata
    profile: str = "standard"
    region: str = "ua"
    branch: str = "main"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisRecord":
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


class AuditWorkspace:
    """
    Manages a local audit workspace.

    Usage:
        # Initialize workspace
        ws = AuditWorkspace.init("/path/to/workspace")

        # Or open existing
        ws = AuditWorkspace("/path/to/workspace")

        # Save analysis
        ws.save_analysis(analysis_id, results)

        # Generate report
        ws.generate_report("summary")

        # Get history
        analyses = ws.list_analyses()
    """

    STRUCTURE = {
        ".audit": ["config.json", "session.json", "calibration.json"],
        "analyses": [],
        "reports": [],
        "documents": ["contracts"],
    }

    def __init__(self, workspace_path: str):
        """Open existing workspace."""
        self.path = Path(workspace_path).resolve()
        self.audit_dir = self.path / ".audit"

        if not self.audit_dir.exists():
            raise ValueError(f"Not a valid workspace: {workspace_path}. Run AuditWorkspace.init() first.")

        self.config = self._load_config()
        self._validation_service = None
        logger.info(f"Opened workspace: {self.config.name} ({self.config.workspace_id})")

    @property
    def validation_service(self):
        """Lazy load validation service."""
        if self._validation_service is None:
            try:
                from gateway.api.settings import get_settings_service
                self._validation_service = get_settings_service()
            except ImportError:
                self._validation_service = None
        return self._validation_service

    @classmethod
    def init(
        cls,
        workspace_path: str,
        name: Optional[str] = None,
        **kwargs,
    ) -> "AuditWorkspace":
        """
        Initialize a new workspace.

        Args:
            workspace_path: Path to create workspace
            name: Workspace name (defaults to folder name)
            **kwargs: Additional config options (region, language, client_name, etc.)

        Returns:
            AuditWorkspace instance
        """
        path = Path(workspace_path).resolve()
        audit_dir = path / ".audit"

        if audit_dir.exists():
            logger.info(f"Workspace already exists at {path}")
            return cls(workspace_path)

        # Create structure
        path.mkdir(parents=True, exist_ok=True)
        audit_dir.mkdir()
        (path / "analyses").mkdir()
        (path / "reports").mkdir()
        (path / "documents").mkdir()
        (path / "documents" / "contracts").mkdir()

        # Create config
        now = datetime.now(timezone.utc).isoformat()
        config = WorkspaceConfig(
            workspace_id=str(uuid4())[:8],
            name=name or path.name,
            created_at=now,
            updated_at=now,
            default_region=kwargs.get("region", "ua"),
            default_profile=kwargs.get("profile", "standard"),
            language=kwargs.get("language", "en"),
            client_name=kwargs.get("client_name"),
            contractor_name=kwargs.get("contractor_name"),
            enable_validation=kwargs.get("enable_validation", True),
        )

        # Save config
        with open(audit_dir / "config.json", "w") as f:
            json.dump(config.to_dict(), f, indent=2)

        # Create empty session
        with open(audit_dir / "session.json", "w") as f:
            json.dump({"active_analysis_id": None, "history": []}, f, indent=2)

        # Create empty calibration
        with open(audit_dir / "calibration.json", "w") as f:
            json.dump({"samples": [], "adjustments": {}}, f, indent=2)

        # Create .gitignore
        with open(path / ".gitignore", "w") as f:
            f.write("# Audit workspace\n.audit/session.json\n")

        logger.info(f"Initialized workspace: {config.name} at {path}")
        return cls(workspace_path)

    def _load_config(self) -> WorkspaceConfig:
        """Load workspace config."""
        with open(self.audit_dir / "config.json") as f:
            return WorkspaceConfig.from_dict(json.load(f))

    def _save_config(self) -> None:
        """Save workspace config."""
        self.config.updated_at = datetime.now(timezone.utc).isoformat()
        with open(self.audit_dir / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    # =========================================================================
    # Validation Integration
    # =========================================================================

    def validate_estimate(
        self,
        hours: float,
        loc: int,
        cost: float,
        methodology: str = "unknown"
    ) -> Dict[str, Any]:
        """Validate estimate using settings service."""
        if not self.config.enable_validation:
            return {"valid": True, "warnings": [], "errors": [], "confidence": "unvalidated"}

        if self.validation_service:
            return self.validation_service.validate_estimate_result(hours, loc, cost, methodology)

        # Fallback basic validation
        if hours <= 0 or loc <= 0:
            return {"valid": False, "errors": ["Hours and LOC must be positive"]}

        hrs_per_kloc = hours / (loc / 1000) if loc > 0 else 0
        if hrs_per_kloc < 2 or hrs_per_kloc > 200:
            return {
                "valid": False,
                "errors": [f"Invalid hrs/KLOC: {hrs_per_kloc:.1f} (expected 2-200)"],
                "confidence": "rejected"
            }

        return {"valid": True, "warnings": [], "errors": [], "confidence": "medium"}

    # =========================================================================
    # Analysis Management
    # =========================================================================

    def save_analysis(
        self,
        analysis_id: str,
        results: Dict[str, Any],
        validate: bool = True,
    ) -> Path:
        """
        Save analysis results with optional validation.

        Args:
            analysis_id: Unique analysis ID
            results: Analysis results dict
            validate: Whether to validate cost estimates

        Returns:
            Path to analysis folder
        """
        analysis_dir = self.path / "analyses" / analysis_id
        analysis_dir.mkdir(parents=True, exist_ok=True)

        # Validate cost estimate if present
        validation_result = {"valid": True, "warnings": [], "errors": []}
        if validate and "cost_estimate" in results:
            cost = results["cost_estimate"]
            hours = cost.get("hours", {}).get("typical", 0)
            loc = results.get("metrics", {}).get("loc_total", 0) or results.get("loc", 0)
            cost_value = cost.get("cost", {}).get("typical", 0)

            validation_result = self.validate_estimate(
                hours, loc, cost_value,
                cost.get("methodology", "unknown")
            )

            # Save validation result
            with open(analysis_dir / "validation.json", "w") as f:
                json.dump(validation_result, f, indent=2)

            # Add to results
            results["validation"] = validation_result

        # Save full results
        with open(analysis_dir / "full.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save metrics separately
        if any(k in results for k in ["metrics", "structure", "code"]):
            metrics = {
                "metrics": results.get("metrics", {}),
                "structure": results.get("structure", {}),
                "code": results.get("code", {}),
                "git": results.get("git", {}),
            }
            with open(analysis_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

        # Save scores
        if "scores" in results:
            with open(analysis_dir / "scores.json", "w") as f:
                json.dump(results["scores"], f, indent=2)

        # Save cost estimate
        if "cost_estimate" in results:
            with open(analysis_dir / "cost.json", "w") as f:
                json.dump(results["cost_estimate"], f, indent=2)

        # Create record
        scores = results.get("scores", {})
        record = AnalysisRecord(
            analysis_id=analysis_id,
            repo_path=results.get("repo_path", results.get("source", "")),
            created_at=results.get("timestamp", datetime.now(timezone.utc).isoformat()),
            repo_health=scores.get("repo_health") if isinstance(scores.get("repo_health"), int) else scores.get("repo_health", {}).get("total"),
            tech_debt=scores.get("tech_debt") if isinstance(scores.get("tech_debt"), int) else scores.get("tech_debt", {}).get("total"),
            stage=results.get("classification", {}).get("stage") or scores.get("product_level"),
            complexity=results.get("metrics", {}).get("complexity"),
            loc=results.get("metrics", {}).get("loc_total") or results.get("loc"),
            cost_estimate=results.get("cost_estimate"),
            validation_passed=validation_result.get("valid", True),
            validation_warnings=validation_result.get("warnings", []),
            profile=results.get("profile", "standard"),
            region=results.get("region", self.config.default_region),
        )

        with open(analysis_dir / "record.json", "w") as f:
            json.dump(record.to_dict(), f, indent=2)

        # Update session
        self._update_session(analysis_id)

        logger.info(f"Saved analysis {analysis_id} to {analysis_dir}")
        return analysis_dir

    def get_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis by ID."""
        analysis_dir = self.path / "analyses" / analysis_id
        full_path = analysis_dir / "full.json"

        if not full_path.exists():
            return None

        with open(full_path) as f:
            return json.load(f)

    def list_analyses(self, limit: int = 20) -> List[AnalysisRecord]:
        """List analyses in workspace."""
        analyses = []
        analyses_dir = self.path / "analyses"

        if not analyses_dir.exists():
            return []

        for analysis_dir in sorted(analyses_dir.iterdir(), reverse=True):
            if not analysis_dir.is_dir():
                continue

            record_path = analysis_dir / "record.json"
            if record_path.exists():
                try:
                    with open(record_path) as f:
                        analyses.append(AnalysisRecord.from_dict(json.load(f)))
                except (json.JSONDecodeError, KeyError):
                    continue

            if len(analyses) >= limit:
                break

        return analyses

    def delete_analysis(self, analysis_id: str) -> bool:
        """Delete analysis."""
        analysis_dir = self.path / "analyses" / analysis_id
        if analysis_dir.exists():
            shutil.rmtree(analysis_dir)
            logger.info(f"Deleted analysis {analysis_id}")
            return True
        return False

    # =========================================================================
    # Session Management
    # =========================================================================

    def _update_session(self, analysis_id: str) -> None:
        """Update session with new analysis."""
        session_path = self.audit_dir / "session.json"

        try:
            with open(session_path) as f:
                session = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            session = {"active_analysis_id": None, "history": []}

        session["active_analysis_id"] = analysis_id
        session["history"].insert(0, {
            "analysis_id": analysis_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        session["history"] = session["history"][:50]  # Keep last 50

        with open(session_path, "w") as f:
            json.dump(session, f, indent=2)

    def get_active_analysis(self) -> Optional[Dict[str, Any]]:
        """Get currently active analysis."""
        session_path = self.audit_dir / "session.json"

        try:
            with open(session_path) as f:
                session = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

        if session.get("active_analysis_id"):
            return self.get_analysis(session["active_analysis_id"])
        return None

    def get_session_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get session history."""
        session_path = self.audit_dir / "session.json"

        try:
            with open(session_path) as f:
                session = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

        return session.get("history", [])[:limit]

    # =========================================================================
    # Report Generation
    # =========================================================================

    def generate_report(
        self,
        report_type: str,
        analysis_id: Optional[str] = None,
        **kwargs,
    ) -> Path:
        """
        Generate report from analysis.

        Args:
            report_type: summary, review, compliance, cost_analysis
            analysis_id: Analysis to report on (defaults to active)
            **kwargs: Additional options

        Returns:
            Path to generated report
        """
        # Get analysis
        if analysis_id:
            analysis = self.get_analysis(analysis_id)
        else:
            analysis = self.get_active_analysis()

        if not analysis:
            raise ValueError("No analysis found")

        # Generate content
        generators = {
            "summary": self._gen_summary_report,
            "review": self._gen_review_report,
            "compliance": self._gen_compliance_report,
            "cost_analysis": self._gen_cost_report,
        }

        if report_type not in generators:
            raise ValueError(f"Unknown report type: {report_type}. Available: {list(generators.keys())}")

        content = generators[report_type](analysis, **kwargs)

        # Save report
        report_path = self.path / "reports" / f"{report_type}.md"
        with open(report_path, "w") as f:
            f.write(content)

        logger.info(f"Generated {report_type} report: {report_path}")
        return report_path

    def _gen_summary_report(self, analysis: Dict, **kwargs) -> str:
        """Generate summary report."""
        stage = analysis.get("classification", {}).get("stage") or \
                analysis.get("scores", {}).get("product_level", "Unknown")
        confidence = analysis.get("classification", {}).get("confidence", 0)
        scores = analysis.get("scores", {})
        health = scores.get("repo_health", {})
        debt = scores.get("tech_debt", {})
        metrics = analysis.get("metrics", {})
        cost = analysis.get("cost_estimate", {})
        validation = analysis.get("validation", {})

        health_total = health if isinstance(health, int) else health.get("total", 0)
        debt_total = debt if isinstance(debt, int) else debt.get("total", 0)

        validation_status = ""
        if validation:
            if validation.get("valid"):
                validation_status = f"Validated ({validation.get('confidence', 'high')})"
            else:
                validation_status = f"**Validation Failed**: {', '.join(validation.get('errors', []))}"

        return f"""# Repository Audit Summary

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Workspace:** {self.config.name}
**Region:** {self.config.default_region.upper()}

## Executive Summary

| Metric | Value |
|--------|-------|
| **Stage** | {stage} |
| **Repo Health** | {health_total}/12 |
| **Tech Debt** | {debt_total}/15 |
| **Complexity** | {metrics.get('complexity', 'N/A')} ({metrics.get('loc_total', 0):,} LOC) |
| **Estimated Effort** | {cost.get('hours', {}).get('typical', 0):,} hours |
| **Validation** | {validation_status or 'Not validated'} |

## Cost Estimate

| Metric | Min | Typical | Max |
|--------|-----|---------|-----|
| Hours | {cost.get('hours', {}).get('min', 0):,} | {cost.get('hours', {}).get('typical', 0):,} | {cost.get('hours', {}).get('max', 0):,} |

## Validation Warnings

{chr(10).join(f'- {w}' for w in validation.get('warnings', [])) or '- No warnings'}

---
*Generated by Audit Platform*
"""

    def _gen_review_report(self, analysis: Dict, **kwargs) -> str:
        """Generate detailed review report."""
        summary = self._gen_summary_report(analysis, **kwargs)

        structure = analysis.get("structure", {})
        cost = analysis.get("cost_estimate", {})

        details = f"""
---

## Detailed Analysis

### Repository Structure

| Check | Status |
|-------|--------|
| README | {'✓' if structure.get('has_readme') else '✗'} |
| Tests | {'✓' if structure.get('has_tests') else '✗'} |
| Documentation | {'✓' if structure.get('has_docs') else '✗'} |
| CI/CD | {'✓' if structure.get('has_ci') else '✗'} |
| Docker | {'✓' if structure.get('has_dockerfile') else '✗'} |
| Dependencies | {'✓' if structure.get('has_deps') else '✗'} |

### Cost Breakdown by Activity

| Activity | Hours |
|----------|-------|
{chr(10).join(f"| {act.title()} | {hrs} |" for act, hrs in cost.get('hours_breakdown', {}).items())}

### Regional Cost Comparison

| Region | Typical Cost |
|--------|--------------|
{chr(10).join(f"| {r.upper()} | ${c.get('cost', {}).get('typical', 0):,.0f} |" for r, c in cost.get('cost_by_region', {}).items())}

---
*Generated by Audit Platform*
"""
        return summary + details

    def _gen_compliance_report(self, analysis: Dict, **kwargs) -> str:
        """Generate compliance report."""
        stage = analysis.get("classification", {}).get("stage") or \
                analysis.get("scores", {}).get("product_level", "Unknown")
        scores = analysis.get("scores", {})
        health = scores.get("repo_health", {})
        debt = scores.get("tech_debt", {})
        structure = analysis.get("structure", {})

        health_total = health if isinstance(health, int) else health.get("total", 0)
        debt_total = debt if isinstance(debt, int) else debt.get("total", 0)

        checks = [
            ("README exists", structure.get("has_readme", False)),
            ("Tests exist", structure.get("has_tests", False)),
            ("CI/CD configured", structure.get("has_ci", False)),
            ("Docker support", structure.get("has_dockerfile", False)),
            ("Documentation", structure.get("has_docs", False)),
            ("Dependencies declared", structure.get("has_deps", False)),
        ]

        passed = sum(1 for _, v in checks if v)
        total = len(checks)

        return f"""# Compliance Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Workspace:** {self.config.name}

## Summary

| Metric | Value |
|--------|-------|
| Stage | {stage} |
| Compliance Score | {passed}/{total} ({passed/total*100:.0f}%) |
| Repo Health | {health_total}/12 |
| Tech Debt | {debt_total}/15 |

## Checklist

| Requirement | Status |
|-------------|--------|
{chr(10).join(f"| {name} | {'✓ PASS' if status else '✗ FAIL'} |" for name, status in checks)}

## Recommendations

{chr(10).join(f'- Add {name.lower()}' for name, status in checks if not status) or '- All checks passed'}

---
*Generated by Audit Platform*
"""

    def _gen_cost_report(self, analysis: Dict, **kwargs) -> str:
        """Generate detailed cost analysis report."""
        cost = analysis.get("cost_estimate", {})
        validation = analysis.get("validation", {})
        metrics = analysis.get("metrics", {})

        return f"""# Cost Analysis Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Workspace:** {self.config.name}

## Input Parameters

| Parameter | Value |
|-----------|-------|
| LOC | {metrics.get('loc_total', 0):,} |
| Methodology | {cost.get('methodology', 'COCOMO II')} |
| Region | {self.config.default_region.upper()} |

## Effort Estimate

| Metric | Min | Typical | Max |
|--------|-----|---------|-----|
| Hours | {cost.get('hours', {}).get('min', 0):,} | {cost.get('hours', {}).get('typical', 0):,} | {cost.get('hours', {}).get('max', 0):,} |
| Days | {cost.get('hours', {}).get('min', 0)//8} | {cost.get('hours', {}).get('typical', 0)//8} | {cost.get('hours', {}).get('max', 0)//8} |

## Activity Breakdown

| Activity | Typical Hours | % |
|----------|---------------|---|
{chr(10).join(f"| {act.title()} | {hrs} | {hrs*100//max(cost.get('hours', {}).get('typical', 1), 1)}% |" for act, hrs in cost.get('hours_breakdown', {}).items())}

## Regional Cost Comparison

| Region | Currency | Typical Cost |
|--------|----------|--------------|
{chr(10).join(f"| {r.upper()} | {c.get('currency', 'USD')} | {c.get('symbol', '$')}{c.get('cost', {}).get('typical', 0):,.0f} |" for r, c in cost.get('cost_by_region', {}).items())}

## Validation Status

| Check | Status |
|-------|--------|
| Valid | {'✓' if validation.get('valid', True) else '✗'} |
| Confidence | {validation.get('confidence', 'N/A')} |

### Warnings
{chr(10).join(f'- {w}' for w in validation.get('warnings', [])) or '- None'}

### Errors
{chr(10).join(f'- {e}' for e in validation.get('errors', [])) or '- None'}

---
*Generated by Audit Platform*
"""

    # =========================================================================
    # Document Generation
    # =========================================================================

    def generate_document(
        self,
        document_type: str,
        analysis_id: Optional[str] = None,
        **kwargs,
    ) -> Path:
        """
        Generate business document.

        Args:
            document_type: acceptance_act, invoice
            analysis_id: Analysis to document
            **kwargs: client_name, contractor_name, etc.

        Returns:
            Path to generated document
        """
        # Get analysis
        if analysis_id:
            analysis = self.get_analysis(analysis_id)
        else:
            analysis = self.get_active_analysis()

        if not analysis:
            raise ValueError("No analysis found")

        # Get names
        client = kwargs.get("client_name", self.config.client_name or "Client")
        contractor = kwargs.get("contractor_name", self.config.contractor_name or "Contractor")
        language = kwargs.get("language", self.config.language)

        generators = {
            "acceptance_act": self._gen_acceptance_act,
            "invoice": self._gen_invoice,
        }

        if document_type not in generators:
            raise ValueError(f"Unknown document type: {document_type}")

        content = generators[document_type](analysis, client, contractor, language)

        # Save document
        doc_path = self.path / "documents" / f"{document_type}.md"
        with open(doc_path, "w") as f:
            f.write(content)

        logger.info(f"Generated {document_type}: {doc_path}")
        return doc_path

    def _gen_acceptance_act(self, analysis: Dict, client: str, contractor: str, lang: str) -> str:
        """Generate acceptance act."""
        date = datetime.now().strftime("%Y-%m-%d")
        stage = analysis.get("classification", {}).get("stage") or \
                analysis.get("scores", {}).get("product_level", "Unknown")
        loc = analysis.get("metrics", {}).get("loc_total", 0)
        hours = analysis.get("cost_estimate", {}).get("hours", {}).get("typical", 0)

        if lang == "uk":
            return f"""# АКТ ПРИЙМАННЯ-ПЕРЕДАЧІ

**Дата:** {date}

**Замовник:** {client}
**Виконавець:** {contractor}

## Виконані роботи

| Назва | Результат |
|-------|-----------|
| Аналіз репозиторію | Завершено |
| Стадія продукту | {stage} |
| Обсяг коду | {loc:,} рядків |
| Оцінка трудовитрат | {hours:,} годин |

## Підписи

Замовник: _______________ / {client}

Виконавець: _______________ / {contractor}
"""
        else:
            return f"""# ACCEPTANCE ACT

**Date:** {date}

**Client:** {client}
**Contractor:** {contractor}

## Completed Work

| Item | Result |
|------|--------|
| Repository Analysis | Completed |
| Product Stage | {stage} |
| Code Volume | {loc:,} LOC |
| Effort Estimate | {hours:,} hours |

## Signatures

Client: _______________ / {client}

Contractor: _______________ / {contractor}
"""

    def _gen_invoice(self, analysis: Dict, client: str, contractor: str, lang: str) -> str:
        """Generate invoice."""
        date = datetime.now().strftime("%Y-%m-%d")
        cost = analysis.get("cost_estimate", {})
        hours = cost.get("hours", {}).get("typical", 0)
        amount = cost.get("cost", {}).get("typical", 0)
        currency = cost.get("cost", {}).get("currency", "EUR")

        rate = amount / max(hours, 1)

        return f"""# INVOICE

**Date:** {date}
**Invoice #:** INV-{datetime.now().strftime('%Y%m%d')}-001

**From:** {contractor}
**To:** {client}

## Services

| Description | Hours | Rate | Amount |
|-------------|-------|------|--------|
| Repository Audit & Analysis | {hours:,} | {currency} {rate:.0f}/hr | {currency} {amount:,.0f} |

**Total: {currency} {amount:,.0f}**

Payment due within 30 days.
"""

    # =========================================================================
    # Calibration
    # =========================================================================

    def add_calibration_feedback(
        self,
        analysis_id: str,
        actual_hours: Optional[float] = None,
        actual_cost: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Add actual hours/cost for calibration."""
        cal_path = self.audit_dir / "calibration.json"

        try:
            with open(cal_path) as f:
                calibration = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            calibration = {"samples": [], "adjustments": {}}

        # Get predicted values
        analysis = self.get_analysis(analysis_id)
        if not analysis:
            raise ValueError(f"Analysis not found: {analysis_id}")

        predicted_hours = analysis.get("cost_estimate", {}).get("hours", {}).get("typical", 0)
        predicted_cost = analysis.get("cost_estimate", {}).get("cost", {}).get("typical", 0)

        # Calculate errors
        hours_error = None
        cost_error = None

        if actual_hours and predicted_hours:
            hours_error = ((predicted_hours - actual_hours) / actual_hours) * 100

        if actual_cost and predicted_cost:
            cost_error = ((predicted_cost - actual_cost) / actual_cost) * 100

        sample = {
            "analysis_id": analysis_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "predicted_hours": predicted_hours,
            "predicted_cost": predicted_cost,
            "actual_hours": actual_hours,
            "actual_cost": actual_cost,
            "hours_error_pct": hours_error,
            "cost_error_pct": cost_error,
        }

        calibration["samples"].append(sample)

        # Recalculate adjustments if enough samples
        if len(calibration["samples"]) >= 3:
            self._recalculate_calibration(calibration)

        with open(cal_path, "w") as f:
            json.dump(calibration, f, indent=2)

        logger.info(f"Added calibration feedback for {analysis_id}")
        return sample

    def _recalculate_calibration(self, calibration: Dict) -> None:
        """Recalculate calibration adjustments."""
        samples = [s for s in calibration["samples"] if s.get("actual_hours")]

        if len(samples) < 3:
            return

        # Calculate mean adjustment ratio
        ratios = [
            s["actual_hours"] / s["predicted_hours"]
            for s in samples
            if s["predicted_hours"] > 0
        ]

        if ratios:
            calibration["adjustments"]["hours_multiplier"] = sum(ratios) / len(ratios)

        # Calculate MAPE
        errors = [abs(s.get("hours_error_pct", 0)) for s in samples if s.get("hours_error_pct")]
        if errors:
            calibration["adjustments"]["mape"] = sum(errors) / len(errors)

    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get calibration statistics."""
        cal_path = self.audit_dir / "calibration.json"

        try:
            with open(cal_path) as f:
                calibration = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"sample_count": 0, "adjustments": {}, "recent_samples": []}

        return {
            "sample_count": len(calibration["samples"]),
            "adjustments": calibration["adjustments"],
            "recent_samples": calibration["samples"][-5:],
        }

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def update_config(self, **kwargs) -> None:
        """Update workspace configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self._save_config()

    def get_status(self) -> Dict[str, Any]:
        """Get workspace status."""
        analyses = self.list_analyses(limit=5)
        session = self.get_session_history(limit=5)
        calibration = self.get_calibration_stats()

        analyses_dir = self.path / "analyses"
        analyses_count = len(list(analyses_dir.iterdir())) if analyses_dir.exists() else 0

        return {
            "workspace_id": self.config.workspace_id,
            "name": self.config.name,
            "path": str(self.path),
            "config": self.config.to_dict(),
            "analyses_count": analyses_count,
            "recent_analyses": [a.to_dict() for a in analyses],
            "session_history": session,
            "calibration": calibration,
        }

    def cleanup(self, keep_last: int = 10) -> int:
        """Cleanup old analyses, keep last N."""
        analyses_dir = self.path / "analyses"

        if not analyses_dir.exists():
            return 0

        all_analyses = sorted(analyses_dir.iterdir(), reverse=True)

        deleted = 0
        for analysis_dir in all_analyses[keep_last:]:
            if analysis_dir.is_dir():
                shutil.rmtree(analysis_dir)
                deleted += 1

        logger.info(f"Cleaned up {deleted} old analyses")
        return deleted
