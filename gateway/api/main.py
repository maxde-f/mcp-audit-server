"""
REST API Gateway for Audit Platform
Provides HTTP API for web UI and external integrations
"""
import asyncio
import uuid
from typing import Optional
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.engine import get_engine
from gateway.api.settings import router as settings_router
from gateway.api.integrations import router as integrations_router
from gateway.api.workspace_routes import router as workspace_router


app = FastAPI(
    title="Audit Platform API",
    description="Repository audit and analysis platform with estimation, validation, and workspace management",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(settings_router)
app.include_router(integrations_router)
app.include_router(workspace_router)

# Store for active analyses
active_analyses = {}
websocket_connections = {}


# Request/Response models
class AuditRequest(BaseModel):
    repo_url: str
    branch: str = "main"
    profile: str = "default"


class MetricExplainRequest(BaseModel):
    metric_name: str
    value: float


class RecommendationsRequest(BaseModel):
    repo_health: int
    tech_debt: int
    security_score: int = 0
    target_level: Optional[str] = "Internal Tool"


# Endpoints
@app.get("/health")
async def health():
    return {"status": "ok", "service": "audit-platform"}


@app.get("/api/workflows")
async def list_workflows():
    """List available workflows"""
    engine = get_engine()
    return {
        "workflows": [
            {
                "name": name,
                "description": wf.get("description", ""),
                "stages": len(wf.get("stages", []))
            }
            for name, wf in engine.workflows.items()
        ]
    }


@app.post("/api/audit")
async def start_audit(request: AuditRequest):
    """Start a new repository audit"""
    analysis_id = str(uuid.uuid4())[:8]
    engine = get_engine()

    # Store analysis state
    active_analyses[analysis_id] = {
        "status": "running",
        "progress": 0,
        "current_stage": "initializing",
        "request": request.dict()
    }

    # Run audit in background
    asyncio.create_task(_run_audit(analysis_id, request, engine))

    return {
        "analysis_id": analysis_id,
        "status": "started",
        "message": "Audit started. Connect to WebSocket for progress updates."
    }


async def _run_audit(analysis_id: str, request: AuditRequest, engine):
    """Background task to run audit"""

    async def progress_callback(progress: dict):
        # Update stored state
        active_analyses[analysis_id].update({
            "progress": progress.get("progress", 0),
            "current_stage": progress.get("stage", ""),
            "stage_name": progress.get("stage_name", "")
        })
        # Notify WebSocket clients
        if analysis_id in websocket_connections:
            ws = websocket_connections[analysis_id]
            try:
                await ws.send_json(progress)
            except:
                pass

    try:
        result = await engine.run_workflow(
            workflow_name="audit",
            inputs={
                "repo_url": request.repo_url,
                "branch": request.branch,
                "profile": request.profile
            },
            analysis_id=analysis_id,
            progress_callback=progress_callback
        )

        active_analyses[analysis_id].update({
            "status": "completed",
            "progress": 100,
            "result": result
        })

    except Exception as e:
        active_analyses[analysis_id].update({
            "status": "failed",
            "error": str(e)
        })


@app.get("/api/audit/{analysis_id}")
async def get_audit_status(analysis_id: str):
    """Get audit status and results"""
    if analysis_id not in active_analyses:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return active_analyses[analysis_id]


@app.get("/api/audit/{analysis_id}/report")
async def get_audit_report(analysis_id: str):
    """Get formatted audit report"""
    if analysis_id not in active_analyses:
        raise HTTPException(status_code=404, detail="Analysis not found")

    analysis = active_analyses[analysis_id]
    if analysis["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed")

    result = analysis.get("result", {})
    scores = result.get("scores", {})

    return {
        "analysis_id": analysis_id,
        "summary": {
            "repo_health": scores.get("repo_health", 0),
            "repo_health_max": 12,
            "tech_debt": scores.get("tech_debt", 0),
            "tech_debt_max": 15,
            "security_score": scores.get("security_score", 0),
            "security_max": 3,
            "product_level": scores.get("product_level", "Unknown"),
            "overall_readiness": scores.get("overall_readiness", 0)
        },
        "stages": result.get("stages", {}),
        "generated_at": result.get("outputs", {}).get("timestamp")
    }


@app.websocket("/api/ws/audit/{analysis_id}")
async def websocket_progress(websocket: WebSocket, analysis_id: str):
    """WebSocket endpoint for real-time progress updates"""
    await websocket.accept()
    websocket_connections[analysis_id] = websocket

    try:
        while True:
            # Send current state
            if analysis_id in active_analyses:
                await websocket.send_json(active_analyses[analysis_id])

                if active_analyses[analysis_id]["status"] in ["completed", "failed"]:
                    break

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        pass
    finally:
        if analysis_id in websocket_connections:
            del websocket_connections[analysis_id]


@app.post("/api/explain/metric")
async def explain_metric(request: MetricExplainRequest):
    """Explain a metric in business terms"""
    engine = get_engine()
    return engine.get_metric_explanation(request.metric_name, request.value)


@app.get("/api/explain/level/{level_name}")
async def explain_level(level_name: str):
    """Explain a product level"""
    engine = get_engine()
    return engine.get_product_level_info(level_name)


@app.post("/api/recommendations")
async def get_recommendations(request: RecommendationsRequest):
    """Get prioritized recommendations"""
    # Use MCP server logic
    from gateway.mcp.server import MCPServer
    mcp = MCPServer()
    return mcp._get_recommendations(request.dict())


@app.get("/api/rules")
async def get_scoring_rules():
    """Get current scoring rules"""
    engine = get_engine()
    return engine.rules.get("scoring", {})


@app.get("/api/knowledge/metrics")
async def get_metrics_knowledge():
    """Get metrics knowledge base"""
    engine = get_engine()
    return engine.knowledge.get("metrics", {})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
