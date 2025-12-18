"""
Workspace Initializer - Creates complete audit workspace structure.

Creates a working directory with:
- Claude memory system
- Analysis storage
- Reports & documents
- Configuration

Usage:
    python -m core.workspace.initializer /path/to/workspace --name "My Project"
"""
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


def create_workspace(
    path: str,
    name: Optional[str] = None,
    region: str = "ua",
    language: str = "en",
    client_name: Optional[str] = None,
    contractor_name: Optional[str] = None,
) -> Path:
    """
    Create a complete audit workspace.

    Structure:
    {workspace}/
    ├── .audit/                     # Audit system files
    │   ├── config.json             # Workspace configuration
    │   ├── session.json            # Current session state
    │   └── calibration.json        # Estimation calibration data
    │
    ├── .claude/                    # Claude memory system
    │   ├── CLAUDE.md               # Instructions for Claude
    │   ├── memory.json             # Working memory (facts)
    │   ├── decisions.json          # Architectural decisions
    │   ├── learnings.json          # Learning from mistakes
    │   ├── context.json            # Current context
    │   ├── sessions/               # Session history
    │   └── decisions/              # Decision markdown files
    │
    ├── analyses/                   # Analysis results
    │   └── {analysis_id}/
    │       ├── full.json           # Complete results
    │       ├── metrics.json        # Code metrics
    │       ├── scores.json         # Calculated scores
    │       ├── cost.json           # Cost estimates
    │       └── validation.json     # Validation results
    │
    ├── reports/                    # Generated reports
    │   ├── summary.md
    │   ├── review.md
    │   ├── compliance.md
    │   └── cost_analysis.md
    │
    ├── documents/                  # Business documents
    │   ├── contracts/              # Uploaded contracts
    │   ├── policies/               # Uploaded policies
    │   ├── acceptance_act.md
    │   └── invoice.md
    │
    ├── .gitignore                  # Git ignore rules
    └── README.md                   # Workspace readme
    """
    workspace_path = Path(path).resolve()
    workspace_name = name or workspace_path.name
    workspace_id = str(uuid4())[:8]
    now = datetime.now(timezone.utc).isoformat()

    logger.info(f"Creating workspace: {workspace_name} at {workspace_path}")

    # Create directory structure
    workspace_path.mkdir(parents=True, exist_ok=True)

    # .audit/
    audit_dir = workspace_path / ".audit"
    audit_dir.mkdir(exist_ok=True)

    # .claude/
    claude_dir = workspace_path / ".claude"
    claude_dir.mkdir(exist_ok=True)
    (claude_dir / "sessions").mkdir(exist_ok=True)
    (claude_dir / "decisions").mkdir(exist_ok=True)

    # analyses/
    (workspace_path / "analyses").mkdir(exist_ok=True)

    # reports/
    (workspace_path / "reports").mkdir(exist_ok=True)

    # documents/
    documents_dir = workspace_path / "documents"
    documents_dir.mkdir(exist_ok=True)
    (documents_dir / "contracts").mkdir(exist_ok=True)
    (documents_dir / "policies").mkdir(exist_ok=True)

    # =========================================================================
    # Create config files
    # =========================================================================

    # .audit/config.json
    config = {
        "workspace_id": workspace_id,
        "name": workspace_name,
        "created_at": now,
        "updated_at": now,
        "default_region": region,
        "default_profile": "standard",
        "language": language,
        "client_name": client_name,
        "contractor_name": contractor_name,
        "repo_health_threshold": 8,
        "tech_debt_threshold": 10,
        "enable_validation": True,
        "strict_validation": False,
    }
    (audit_dir / "config.json").write_text(json.dumps(config, indent=2))

    # .audit/session.json
    session = {
        "active_analysis_id": None,
        "history": [],
        "created_at": now,
    }
    (audit_dir / "session.json").write_text(json.dumps(session, indent=2))

    # .audit/calibration.json
    calibration = {
        "samples": [],
        "adjustments": {},
        "created_at": now,
    }
    (audit_dir / "calibration.json").write_text(json.dumps(calibration, indent=2))

    # =========================================================================
    # Create Claude memory files
    # =========================================================================

    # .claude/CLAUDE.md
    claude_md = f"""# Claude Instructions for {workspace_name}

## Workspace Overview
- **ID:** {workspace_id}
- **Created:** {now[:10]}
- **Region:** {region.upper()}
- **Language:** {language}

## Memory System

Your persistent memory is stored in this `.claude/` directory. Use it to:
- Remember important facts about the project
- Track architectural decisions
- Learn from mistakes and corrections
- Maintain context between sessions

### Files:
- `memory.json` - Working memory (facts, observations)
- `decisions.json` - Architectural Decision Records
- `learnings.json` - Lessons learned from errors
- `context.json` - Current session context
- `sessions/` - Historical session data

## How to Use

### At Session Start
1. Read `context.json` to understand current state
2. Check `learnings.json` to avoid past mistakes
3. Review recent `decisions.json` for context

### During Session
1. Store important facts with `store_memory()`
2. Record decisions with `record_decision()`
3. Update context with `set_context()`

### At Session End
1. Save session with `save_session()`
2. Update context summary
3. Record any learnings

## Audit Workflow

Standard workflow for repository audits:

```
1. quick_scan     → Fast overview (files, LOC, languages)
2. detect_type    → Project type, framework detection
3. check_quality  → Health, debt, security analysis
4. estimate_cost  → COCOMO + 8 methodologies
5. full_audit     → Complete analysis + compliance
```

## Estimation Methods

Available methodologies (all formulas hardcoded):
- COCOMO II Modern: `Effort = 0.5 × (KLOC)^0.85 × EAF`
- Gartner Standard
- IEEE 1063
- Microsoft Standard
- Google Guidelines
- PMI Standard
- SEI SLIM (10K+ LOC)
- Function Points

## Regional Rates

{region.upper()} rates are configured by default.
Available regions: UA, UA_Compliance, PL, EU, DE, UK, US, IN

## Validation (Anti-Hallucination)

All estimates are validated against bounds:
- Rates: $5-300/hr
- Hours/KLOC: 2-200
- PERT ratio: max 10x

---

## Project Notes

<!-- Add project-specific notes below -->



---
*Auto-generated by audit-platform workspace initializer*
"""
    (claude_dir / "CLAUDE.md").write_text(claude_md)

    # .claude/memory.json
    memory = {
        "entries": [
            {
                "id": "init001",
                "type": "fact",
                "content": f"Workspace '{workspace_name}' initialized on {now[:10]}",
                "metadata": {"source": "initializer"},
                "created_at": now,
                "access_count": 0,
            }
        ]
    }
    (claude_dir / "memory.json").write_text(json.dumps(memory, indent=2))

    # .claude/decisions.json
    decisions = {"decisions": []}
    (claude_dir / "decisions.json").write_text(json.dumps(decisions, indent=2))

    # .claude/learnings.json
    learnings = {"learnings": []}
    (claude_dir / "learnings.json").write_text(json.dumps(learnings, indent=2))

    # .claude/context.json
    context = {
        "workspace_id": workspace_id,
        "workspace_name": workspace_name,
        "current_topic": None,
        "active_analysis_id": None,
        "last_action": "workspace_initialized",
        "updated_at": now,
    }
    (claude_dir / "context.json").write_text(json.dumps(context, indent=2))

    # =========================================================================
    # Create .gitignore
    # =========================================================================

    gitignore = """# Audit Workspace

# Session data (contains conversation history)
.audit/session.json
.claude/sessions/

# Local cache
.cache/
__pycache__/

# Temporary files
*.tmp
*.temp

# OS files
.DS_Store
Thumbs.db

# IDE
.idea/
.vscode/
*.swp

# Keep structure
!.gitkeep
"""
    (workspace_path / ".gitignore").write_text(gitignore)

    # =========================================================================
    # Create README.md
    # =========================================================================

    readme = f"""# {workspace_name}

Audit workspace created by audit-platform.

## Structure

```
{workspace_name}/
├── .audit/          # Audit configuration & state
├── .claude/         # Claude memory system
├── analyses/        # Analysis results
├── reports/         # Generated reports
└── documents/       # Business documents
```

## Quick Start

### 1. Run Analysis
```bash
# Using CLI
python run.py audit /path/to/repo --task full_audit

# Using API
curl -X POST http://localhost:8080/api/audit \\
  -H "Content-Type: application/json" \\
  -d '{{"repo_url": "https://github.com/user/repo"}}'
```

### 2. Generate Report
```bash
# Summary report
curl "http://localhost:8080/api/workspace/report?path={workspace_path}&type=summary"
```

### 3. View Results
- Check `analyses/` for raw results
- Check `reports/` for formatted reports

## Configuration

Edit `.audit/config.json` to change:
- Default region
- Validation thresholds
- Client/contractor names

## Claude Memory

Claude's persistent memory is in `.claude/`:
- Facts and observations
- Architectural decisions
- Learnings from mistakes

---
*Created: {now[:10]}*
*ID: {workspace_id}*
"""
    (workspace_path / "README.md").write_text(readme)

    # Create .gitkeep files
    for dir_name in ["analyses", "reports"]:
        (workspace_path / dir_name / ".gitkeep").touch()

    logger.info(f"Workspace created: {workspace_path}")

    return workspace_path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Initialize an audit workspace"
    )
    parser.add_argument("path", help="Path to create workspace")
    parser.add_argument("--name", help="Workspace name")
    parser.add_argument("--region", default="ua", help="Default region (ua, eu, us, etc.)")
    parser.add_argument("--language", default="en", help="Language (en, uk)")
    parser.add_argument("--client", help="Client name")
    parser.add_argument("--contractor", help="Contractor name")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    workspace = create_workspace(
        path=args.path,
        name=args.name,
        region=args.region,
        language=args.language,
        client_name=args.client,
        contractor_name=args.contractor,
    )

    print(f"\n✓ Workspace created: {workspace}")
    print(f"\nStructure:")
    print(f"  {workspace}/")
    print(f"  ├── .audit/        (config, session, calibration)")
    print(f"  ├── .claude/       (memory, decisions, learnings)")
    print(f"  ├── analyses/      (analysis results)")
    print(f"  ├── reports/       (generated reports)")
    print(f"  └── documents/     (contracts, policies)")
    print(f"\nNext steps:")
    print(f"  1. Edit .claude/CLAUDE.md to add project-specific instructions")
    print(f"  2. Run an audit on your repository")
    print(f"  3. Generate reports")


if __name__ == "__main__":
    main()
