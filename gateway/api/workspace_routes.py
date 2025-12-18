"""
Workspace API Routes for Audit Platform

Provides REST endpoints for workspace management:
- Workspace initialization and status
- Analysis storage and retrieval
- Report generation
- Document generation
- Calibration feedback
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.workspace import AuditWorkspace, WorkspaceConfig, ClaudeMemory, create_workspace as init_workspace_structure


# =============================================================================
# MODELS
# =============================================================================

class InitWorkspaceRequest(BaseModel):
    """Request to initialize workspace."""
    path: str
    name: Optional[str] = None
    region: str = "ua"
    language: str = "en"
    client_name: Optional[str] = None
    contractor_name: Optional[str] = None


class SaveAnalysisRequest(BaseModel):
    """Request to save analysis."""
    analysis_id: str
    results: Dict[str, Any]
    validate: bool = True


class GenerateReportRequest(BaseModel):
    """Request to generate report."""
    report_type: str = "summary"
    analysis_id: Optional[str] = None


class GenerateDocumentRequest(BaseModel):
    """Request to generate document."""
    document_type: str = "acceptance_act"
    analysis_id: Optional[str] = None
    client_name: Optional[str] = None
    contractor_name: Optional[str] = None
    language: Optional[str] = None


class CalibrationFeedbackRequest(BaseModel):
    """Request to add calibration feedback."""
    analysis_id: str
    actual_hours: Optional[float] = None
    actual_cost: Optional[float] = None


class UpdateConfigRequest(BaseModel):
    """Request to update workspace config."""
    default_region: Optional[str] = None
    default_profile: Optional[str] = None
    language: Optional[str] = None
    client_name: Optional[str] = None
    contractor_name: Optional[str] = None
    repo_health_threshold: Optional[int] = None
    tech_debt_threshold: Optional[int] = None
    enable_validation: Optional[bool] = None


# =============================================================================
# WORKSPACE MANAGER
# =============================================================================

# Store active workspaces
_active_workspaces: Dict[str, AuditWorkspace] = {}


def get_workspace(workspace_path: str) -> AuditWorkspace:
    """Get or load workspace."""
    path = str(Path(workspace_path).resolve())

    if path not in _active_workspaces:
        try:
            _active_workspaces[path] = AuditWorkspace(workspace_path)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    return _active_workspaces[path]


def clear_workspace_cache(workspace_path: str):
    """Clear cached workspace."""
    path = str(Path(workspace_path).resolve())
    if path in _active_workspaces:
        del _active_workspaces[path]


# =============================================================================
# API ROUTER
# =============================================================================

router = APIRouter(prefix="/api/workspace", tags=["workspace"])


# --- Workspace Management ---

@router.post("/init")
async def init_workspace(request: InitWorkspaceRequest):
    """Initialize a new workspace."""
    try:
        ws = AuditWorkspace.init(
            request.path,
            name=request.name,
            region=request.region,
            language=request.language,
            client_name=request.client_name,
            contractor_name=request.contractor_name,
        )

        # Cache it
        path = str(Path(request.path).resolve())
        _active_workspaces[path] = ws

        return {
            "message": "Workspace initialized",
            "workspace_id": ws.config.workspace_id,
            "path": str(ws.path),
            "config": ws.config.to_dict(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/status")
async def get_workspace_status(path: str):
    """Get workspace status."""
    ws = get_workspace(path)
    return ws.get_status()


@router.put("/config")
async def update_workspace_config(path: str, request: UpdateConfigRequest):
    """Update workspace configuration."""
    ws = get_workspace(path)

    updates = {k: v for k, v in request.dict().items() if v is not None}
    if updates:
        ws.update_config(**updates)

    return {
        "message": "Config updated",
        "config": ws.config.to_dict(),
    }


# --- Analysis Management ---

@router.post("/analysis")
async def save_analysis(path: str, request: SaveAnalysisRequest):
    """Save analysis results to workspace."""
    ws = get_workspace(path)

    try:
        analysis_path = ws.save_analysis(
            request.analysis_id,
            request.results,
            validate=request.validate,
        )

        return {
            "message": "Analysis saved",
            "analysis_id": request.analysis_id,
            "path": str(analysis_path),
            "validation": request.results.get("validation", {}),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/analysis/{analysis_id}")
async def get_analysis(path: str, analysis_id: str):
    """Get analysis by ID."""
    ws = get_workspace(path)
    analysis = ws.get_analysis(analysis_id)

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return analysis


@router.get("/analyses")
async def list_analyses(path: str, limit: int = 20):
    """List analyses in workspace."""
    ws = get_workspace(path)
    analyses = ws.list_analyses(limit)

    return {
        "count": len(analyses),
        "analyses": [a.to_dict() for a in analyses],
    }


@router.delete("/analysis/{analysis_id}")
async def delete_analysis(path: str, analysis_id: str):
    """Delete an analysis."""
    ws = get_workspace(path)

    if not ws.delete_analysis(analysis_id):
        raise HTTPException(status_code=404, detail="Analysis not found")

    return {"message": "Analysis deleted"}


@router.get("/analysis/active")
async def get_active_analysis(path: str):
    """Get currently active analysis."""
    ws = get_workspace(path)
    analysis = ws.get_active_analysis()

    if not analysis:
        return {"active": False, "analysis": None}

    return {"active": True, "analysis": analysis}


# --- Reports ---

@router.post("/report")
async def generate_report(path: str, request: GenerateReportRequest):
    """Generate a report."""
    ws = get_workspace(path)

    try:
        report_path = ws.generate_report(
            request.report_type,
            analysis_id=request.analysis_id,
        )

        # Read content
        content = report_path.read_text()

        return {
            "message": "Report generated",
            "type": request.report_type,
            "path": str(report_path),
            "content": content,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/reports")
async def list_reports(path: str):
    """List available reports."""
    ws = get_workspace(path)
    reports_dir = ws.path / "reports"

    if not reports_dir.exists():
        return {"reports": []}

    reports = []
    for report_file in reports_dir.glob("*.md"):
        reports.append({
            "name": report_file.stem,
            "path": str(report_file),
            "size": report_file.stat().st_size,
            "modified": datetime.fromtimestamp(report_file.stat().st_mtime).isoformat(),
        })

    return {"reports": reports}


# --- Documents ---

@router.post("/document")
async def generate_document(path: str, request: GenerateDocumentRequest):
    """Generate a business document."""
    ws = get_workspace(path)

    try:
        doc_path = ws.generate_document(
            request.document_type,
            analysis_id=request.analysis_id,
            client_name=request.client_name,
            contractor_name=request.contractor_name,
            language=request.language,
        )

        content = doc_path.read_text()

        return {
            "message": "Document generated",
            "type": request.document_type,
            "path": str(doc_path),
            "content": content,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- Calibration ---

@router.post("/calibration")
async def add_calibration(path: str, request: CalibrationFeedbackRequest):
    """Add calibration feedback."""
    ws = get_workspace(path)

    try:
        sample = ws.add_calibration_feedback(
            request.analysis_id,
            actual_hours=request.actual_hours,
            actual_cost=request.actual_cost,
        )

        return {
            "message": "Calibration feedback added",
            "sample": sample,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/calibration")
async def get_calibration_stats(path: str):
    """Get calibration statistics."""
    ws = get_workspace(path)
    return ws.get_calibration_stats()


# --- Session ---

@router.get("/session/history")
async def get_session_history(path: str, limit: int = 10):
    """Get session history."""
    ws = get_workspace(path)
    history = ws.get_session_history(limit)
    return {"history": history}


# --- Cleanup ---

@router.post("/cleanup")
async def cleanup_workspace(path: str, keep_last: int = 10):
    """Cleanup old analyses."""
    ws = get_workspace(path)
    deleted = ws.cleanup(keep_last)

    return {
        "message": f"Cleaned up {deleted} old analyses",
        "deleted_count": deleted,
    }


# =============================================================================
# MEMORY ROUTES (Claude Memory System)
# =============================================================================

# Store active memory instances
_memory_instances: Dict[str, ClaudeMemory] = {}


def get_memory(workspace_path: str) -> ClaudeMemory:
    """Get or create memory instance for workspace."""
    path = str(Path(workspace_path).resolve())

    if path not in _memory_instances:
        _memory_instances[path] = ClaudeMemory(workspace_path)

    return _memory_instances[path]


# --- Memory ---

@router.post("/memory/store")
async def store_memory(
    path: str,
    content: str,
    type: str = "fact",
    metadata: Optional[Dict[str, Any]] = None
):
    """Store a memory entry."""
    memory = get_memory(path)
    entry = memory.store_memory(content, type, **(metadata or {}))
    return {"message": "Memory stored", "entry": entry.to_dict()}


@router.get("/memory/recall")
async def recall_memory(
    path: str,
    query: Optional[str] = None,
    type: Optional[str] = None,
    limit: int = 10
):
    """Recall memories."""
    memory = get_memory(path)
    entries = memory.recall(query, type, limit)
    return {"entries": [e.to_dict() for e in entries]}


@router.delete("/memory/{memory_id}")
async def forget_memory(path: str, memory_id: str):
    """Delete a memory entry."""
    memory = get_memory(path)
    if not memory.forget(memory_id):
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"message": "Memory deleted"}


# --- Decisions ---

@router.post("/memory/decision")
async def record_decision(
    path: str,
    title: str,
    context: str,
    decision: str,
    consequences: Optional[List[str]] = None,
    alternatives: Optional[List[str]] = None
):
    """Record an architectural decision."""
    memory = get_memory(path)
    dec = memory.record_decision(
        title=title,
        context=context,
        decision=decision,
        consequences=consequences,
        alternatives=alternatives,
    )
    return {"message": "Decision recorded", "decision": dec.to_dict()}


@router.get("/memory/decisions")
async def get_decisions(
    path: str,
    status: Optional[str] = None,
    limit: int = 20
):
    """Get recorded decisions."""
    memory = get_memory(path)
    decisions = memory.get_decisions(status, limit)
    return {"decisions": [d.to_dict() for d in decisions]}


# --- Learnings ---

@router.post("/memory/learning")
async def record_learning(
    path: str,
    what_happened: str,
    what_learned: str,
    correction: Optional[str] = None,
    pattern: Optional[str] = None
):
    """Record a learning event."""
    memory = get_memory(path)
    event = memory.record_learning(
        what_happened=what_happened,
        what_learned=what_learned,
        correction=correction,
        pattern=pattern,
    )
    return {"message": "Learning recorded", "learning": event.to_dict()}


@router.get("/memory/learnings")
async def get_learnings(path: str, limit: int = 20):
    """Get learning events."""
    memory = get_memory(path)
    learnings = memory.get_learnings(limit)
    return {"learnings": [l.to_dict() for l in learnings]}


# --- Context ---

@router.get("/memory/context")
async def get_context(path: str, key: Optional[str] = None):
    """Get current context."""
    memory = get_memory(path)
    ctx = memory.get_context(key)
    return {"context": ctx}


@router.put("/memory/context")
async def update_context(path: str, updates: Dict[str, Any]):
    """Update context values."""
    memory = get_memory(path)
    memory.update_context(**updates)
    return {"message": "Context updated", "context": memory.get_context()}


@router.delete("/memory/context")
async def clear_context(path: str):
    """Clear current context."""
    memory = get_memory(path)
    memory.clear_context()
    return {"message": "Context cleared"}


# --- Sessions ---

@router.post("/memory/session/save")
async def save_memory_session(
    path: str,
    session_id: str,
    messages: List[Dict[str, Any]],
    summary: Optional[str] = None
):
    """Save a session."""
    memory = get_memory(path)
    memory.save_session(session_id, messages, summary)
    return {"message": "Session saved", "session_id": session_id}


@router.get("/memory/session/{session_id}")
async def load_memory_session(path: str, session_id: str):
    """Load a session."""
    memory = get_memory(path)
    session = memory.load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.get("/memory/sessions")
async def list_memory_sessions(path: str, limit: int = 10):
    """List sessions."""
    memory = get_memory(path)
    sessions = memory.list_sessions(limit)
    return {"sessions": sessions}


# --- Export/Stats ---

@router.get("/memory/stats")
async def get_memory_stats(path: str):
    """Get memory statistics."""
    memory = get_memory(path)
    return memory.get_stats()


@router.get("/memory/export")
async def export_memory(path: str):
    """Export all memory data."""
    memory = get_memory(path)
    return memory.export_all()


@router.post("/memory/import")
async def import_memory(path: str, data: Dict[str, Any]):
    """Import memory data."""
    memory = get_memory(path)
    memory.import_all(data)
    return {"message": "Memory imported"}


@router.get("/memory/context-prompt")
async def get_context_prompt(path: str):
    """Get context prompt for LLM."""
    memory = get_memory(path)
    prompt = memory.build_context_prompt()
    return {"prompt": prompt}


# --- CLAUDE.md ---

@router.get("/claude-instructions")
async def get_claude_instructions(path: str):
    """Get CLAUDE.md contents."""
    claude_md = Path(path) / ".claude" / "CLAUDE.md"
    if not claude_md.exists():
        raise HTTPException(status_code=404, detail="CLAUDE.md not found")
    return {"content": claude_md.read_text()}


@router.put("/claude-instructions")
async def update_claude_instructions(path: str, content: str):
    """Update CLAUDE.md contents."""
    claude_md = Path(path) / ".claude" / "CLAUDE.md"
    claude_md.parent.mkdir(parents=True, exist_ok=True)
    claude_md.write_text(content)
    return {"message": "CLAUDE.md updated"}
