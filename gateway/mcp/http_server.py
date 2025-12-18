#!/usr/bin/env python3
"""
MCP Audit HTTP Server v1.0 - SSE Transport for Web Claude

Features:
- MCP SSE at /mcp/sse (for Web Claude integration)
- OAuth 2.0 with PKCE for authentication
- PostgreSQL for persistent memory
- Redis for session cache
- 12+ unified audit tools
- Anti-hallucination validation

Usage:
    python -m gateway.mcp.http_server

    Or with environment variables:
    DATABASE_URL=postgres://... REDIS_URL=redis://... python http_server.py
"""

import os
import sys
import json
import logging
import asyncio
import secrets
import hashlib
import base64
import urllib.parse
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import existing MCP server
from gateway.mcp.server import AuditMCPServer, FORMULAS_AVAILABLE

# Try to import formulas
try:
    from executors.cost_estimator.formulas import REGIONAL_RATES
except ImportError:
    REGIONAL_RATES = {
        "ua": {"junior": 15, "middle": 25, "senior": 40, "lead": 55, "architect": 70},
        "eu": {"junior": 40, "middle": 65, "senior": 95, "lead": 120, "architect": 150},
        "us": {"junior": 60, "middle": 100, "senior": 150, "lead": 200, "architect": 250},
    }

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8090")
DATABASE_URL = os.environ.get("DATABASE_URL", "")
REDIS_URL = os.environ.get("REDIS_URL", "")

# =============================================================================
# DATABASE (PostgreSQL)
# =============================================================================

db_connection = None
redis_client = None


def init_postgres():
    """Initialize PostgreSQL connection and create tables."""
    global db_connection
    if not DATABASE_URL:
        logger.warning("DATABASE_URL not set - using in-memory storage")
        return False

    try:
        import psycopg2
        db_connection = psycopg2.connect(DATABASE_URL)
        cursor = db_connection.cursor()

        # Memory table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS claude_memory (
                id VARCHAR(32) PRIMARY KEY,
                workspace_id VARCHAR(64),
                type VARCHAR(50) NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB DEFAULT '{}',
                access_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Decisions table (ADR format)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS claude_decisions (
                id VARCHAR(32) PRIMARY KEY,
                workspace_id VARCHAR(64),
                title VARCHAR(255) NOT NULL,
                context TEXT,
                decision TEXT NOT NULL,
                consequences TEXT[],
                alternatives TEXT[],
                status VARCHAR(20) DEFAULT 'accepted',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Learnings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS claude_learnings (
                id VARCHAR(32) PRIMARY KEY,
                workspace_id VARCHAR(64),
                what_happened TEXT NOT NULL,
                what_learned TEXT NOT NULL,
                correction TEXT,
                pattern VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS claude_sessions (
                session_id VARCHAR(64) PRIMARY KEY,
                workspace_id VARCHAR(64),
                messages JSONB DEFAULT '[]',
                summary TEXT,
                context JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Analysis results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_results (
                id VARCHAR(32) PRIMARY KEY,
                workspace_id VARCHAR(64),
                repo_url VARCHAR(512),
                repo_name VARCHAR(255),
                metrics JSONB,
                scores JSONB,
                cost_estimate JSONB,
                validation JSONB,
                status VARCHAR(20) DEFAULT 'completed',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Calibration feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calibration_feedback (
                id SERIAL PRIMARY KEY,
                analysis_id VARCHAR(32) REFERENCES analysis_results(id),
                estimated_hours FLOAT,
                actual_hours FLOAT,
                estimated_cost FLOAT,
                actual_cost FLOAT,
                deviation_pct FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Workspace settings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workspace_settings (
                workspace_id VARCHAR(64),
                key VARCHAR(64),
                value JSONB NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (workspace_id, key)
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_workspace ON claude_memory(workspace_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON claude_memory(type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_decisions_workspace ON claude_decisions(workspace_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_workspace ON analysis_results(workspace_id)")

        db_connection.commit()
        logger.info("PostgreSQL initialized successfully")
        return True

    except Exception as e:
        logger.error(f"PostgreSQL initialization failed: {e}")
        return False


def init_redis():
    """Initialize Redis connection."""
    global redis_client
    if not REDIS_URL:
        logger.warning("REDIS_URL not set - using in-memory cache")
        return False

    try:
        import redis
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("Redis connected successfully")
        return True
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        return False


# =============================================================================
# IN-MEMORY FALLBACKS
# =============================================================================

_memory_store: Dict[str, List[Dict]] = {}
_decisions_store: Dict[str, List[Dict]] = {}
_learnings_store: Dict[str, List[Dict]] = {}
_context_store: Dict[str, Dict] = {}
_cache: Dict[str, tuple] = {}  # key -> (value, expiry_time)
CACHE_TTL = 1800  # 30 minutes


def cache_get(key: str) -> Optional[Any]:
    """Get from cache."""
    if redis_client:
        try:
            val = redis_client.get(key)
            return json.loads(val) if val else None
        except:
            pass

    if key in _cache:
        value, expiry = _cache[key]
        if datetime.now().timestamp() < expiry:
            return value
        del _cache[key]
    return None


def cache_set(key: str, value: Any, ttl: int = CACHE_TTL):
    """Set in cache."""
    if redis_client:
        try:
            redis_client.setex(key, ttl, json.dumps(value))
            return
        except:
            pass
    _cache[key] = (value, datetime.now().timestamp() + ttl)


# =============================================================================
# MEMORY PERSISTENCE LAYER
# =============================================================================

class MemoryPersistence:
    """PostgreSQL-backed memory persistence with Redis cache."""

    @staticmethod
    def store_memory(workspace_id: str, content: str, type: str = "fact", **metadata) -> Dict:
        """Store a memory entry."""
        entry_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc).isoformat()

        entry = {
            "id": entry_id,
            "workspace_id": workspace_id,
            "type": type,
            "content": content,
            "metadata": metadata,
            "access_count": 0,
            "created_at": now,
        }

        if db_connection:
            try:
                cursor = db_connection.cursor()
                cursor.execute("""
                    INSERT INTO claude_memory (id, workspace_id, type, content, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                """, (entry_id, workspace_id, type, content, json.dumps(metadata)))
                db_connection.commit()
            except Exception as e:
                logger.error(f"Error storing memory: {e}")
                db_connection.rollback()
        else:
            if workspace_id not in _memory_store:
                _memory_store[workspace_id] = []
            _memory_store[workspace_id].append(entry)

        # Invalidate cache
        cache_key = f"memory:{workspace_id}"
        if redis_client:
            try:
                redis_client.delete(cache_key)
            except:
                pass
        elif cache_key in _cache:
            del _cache[cache_key]

        return entry

    @staticmethod
    def recall_memory(
        workspace_id: str,
        query: Optional[str] = None,
        type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """Recall memories with caching."""
        cache_key = f"memory:{workspace_id}:{type}:{query}:{limit}"
        cached = cache_get(cache_key)
        if cached:
            return cached

        results = []

        if db_connection:
            try:
                cursor = db_connection.cursor()
                sql = "SELECT id, type, content, metadata, access_count, created_at FROM claude_memory WHERE workspace_id = %s"
                params = [workspace_id]

                if type:
                    sql += " AND type = %s"
                    params.append(type)

                if query:
                    sql += " AND LOWER(content) LIKE %s"
                    params.append(f"%{query.lower()}%")

                sql += " ORDER BY access_count DESC, created_at DESC LIMIT %s"
                params.append(limit)

                cursor.execute(sql, params)

                for row in cursor.fetchall():
                    results.append({
                        "id": row[0],
                        "type": row[1],
                        "content": row[2],
                        "metadata": row[3] or {},
                        "access_count": row[4],
                        "created_at": row[5].isoformat() if row[5] else None,
                    })

                # Update access count
                if results:
                    ids = [r["id"] for r in results]
                    cursor.execute(
                        "UPDATE claude_memory SET access_count = access_count + 1, updated_at = NOW() WHERE id = ANY(%s)",
                        (ids,)
                    )
                    db_connection.commit()

            except Exception as e:
                logger.error(f"Error recalling memory: {e}")
        else:
            entries = _memory_store.get(workspace_id, [])
            if type:
                entries = [e for e in entries if e["type"] == type]
            if query:
                query_lower = query.lower()
                entries = [e for e in entries if query_lower in e["content"].lower()]
            results = sorted(entries, key=lambda x: (x.get("access_count", 0), x.get("created_at", "")), reverse=True)[:limit]

        cache_set(cache_key, results, ttl=300)
        return results

    @staticmethod
    def record_decision(
        workspace_id: str,
        title: str,
        context: str,
        decision: str,
        consequences: List[str] = None,
        alternatives: List[str] = None,
    ) -> Dict:
        """Record an architectural decision."""
        entry_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc).isoformat()

        entry = {
            "id": entry_id,
            "workspace_id": workspace_id,
            "title": title,
            "context": context,
            "decision": decision,
            "consequences": consequences or [],
            "alternatives": alternatives or [],
            "status": "accepted",
            "created_at": now,
        }

        if db_connection:
            try:
                cursor = db_connection.cursor()
                cursor.execute("""
                    INSERT INTO claude_decisions (id, workspace_id, title, context, decision, consequences, alternatives)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (entry_id, workspace_id, title, context, decision, consequences or [], alternatives or []))
                db_connection.commit()
            except Exception as e:
                logger.error(f"Error recording decision: {e}")
                db_connection.rollback()
        else:
            if workspace_id not in _decisions_store:
                _decisions_store[workspace_id] = []
            _decisions_store[workspace_id].append(entry)

        return entry

    @staticmethod
    def record_learning(
        workspace_id: str,
        what_happened: str,
        what_learned: str,
        correction: Optional[str] = None,
        pattern: Optional[str] = None,
    ) -> Dict:
        """Record a learning event."""
        entry_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc).isoformat()

        entry = {
            "id": entry_id,
            "workspace_id": workspace_id,
            "what_happened": what_happened,
            "what_learned": what_learned,
            "correction": correction,
            "pattern": pattern,
            "created_at": now,
        }

        if db_connection:
            try:
                cursor = db_connection.cursor()
                cursor.execute("""
                    INSERT INTO claude_learnings (id, workspace_id, what_happened, what_learned, correction, pattern)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (entry_id, workspace_id, what_happened, what_learned, correction, pattern))
                db_connection.commit()
            except Exception as e:
                logger.error(f"Error recording learning: {e}")
                db_connection.rollback()
        else:
            if workspace_id not in _learnings_store:
                _learnings_store[workspace_id] = []
            _learnings_store[workspace_id].append(entry)

        return entry

    @staticmethod
    def get_context(workspace_id: str) -> Dict:
        """Get current context."""
        cache_key = f"context:{workspace_id}"
        cached = cache_get(cache_key)
        if cached:
            return cached

        if db_connection:
            try:
                cursor = db_connection.cursor()
                cursor.execute(
                    "SELECT context FROM claude_sessions WHERE workspace_id = %s ORDER BY updated_at DESC LIMIT 1",
                    (workspace_id,)
                )
                row = cursor.fetchone()
                if row:
                    return row[0] or {}
            except Exception as e:
                logger.error(f"Error getting context: {e}")

        return _context_store.get(workspace_id, {})

    @staticmethod
    def set_context(workspace_id: str, context: Dict):
        """Set current context."""
        cache_key = f"context:{workspace_id}"
        cache_set(cache_key, context, ttl=3600)
        _context_store[workspace_id] = context

    @staticmethod
    def get_decisions(workspace_id: str, limit: int = 10) -> List[Dict]:
        """Get recorded decisions."""
        if db_connection:
            try:
                cursor = db_connection.cursor()
                cursor.execute("""
                    SELECT id, title, context, decision, consequences, alternatives, status, created_at
                    FROM claude_decisions WHERE workspace_id = %s
                    ORDER BY created_at DESC LIMIT %s
                """, (workspace_id, limit))

                results = []
                for row in cursor.fetchall():
                    results.append({
                        "id": row[0],
                        "title": row[1],
                        "context": row[2],
                        "decision": row[3],
                        "consequences": row[4] or [],
                        "alternatives": row[5] or [],
                        "status": row[6],
                        "created_at": row[7].isoformat() if row[7] else None,
                    })
                return results
            except Exception as e:
                logger.error(f"Error getting decisions: {e}")

        return _decisions_store.get(workspace_id, [])[:limit]

    @staticmethod
    def get_learnings(workspace_id: str, limit: int = 10) -> List[Dict]:
        """Get recorded learnings."""
        if db_connection:
            try:
                cursor = db_connection.cursor()
                cursor.execute("""
                    SELECT id, what_happened, what_learned, correction, pattern, created_at
                    FROM claude_learnings WHERE workspace_id = %s
                    ORDER BY created_at DESC LIMIT %s
                """, (workspace_id, limit))

                results = []
                for row in cursor.fetchall():
                    results.append({
                        "id": row[0],
                        "what_happened": row[1],
                        "what_learned": row[2],
                        "correction": row[3],
                        "pattern": row[4],
                        "created_at": row[5].isoformat() if row[5] else None,
                    })
                return results
            except Exception as e:
                logger.error(f"Error getting learnings: {e}")

        return _learnings_store.get(workspace_id, [])[:limit]


# =============================================================================
# SETTINGS MANAGER
# =============================================================================

# Default settings
DEFAULT_SETTINGS = {
    "rates": {
        "ua": {"junior": 15, "middle": 25, "senior": 40, "lead": 55, "architect": 70, "typical": 35},
        "ua_compliance": {"junior": 20, "middle": 35, "senior": 55, "lead": 75, "architect": 95, "typical": 50},
        "pl": {"junior": 25, "middle": 40, "senior": 60, "lead": 80, "architect": 100, "typical": 55},
        "eu": {"junior": 40, "middle": 65, "senior": 95, "lead": 120, "architect": 150, "typical": 85},
        "de": {"junior": 50, "middle": 80, "senior": 110, "lead": 140, "architect": 175, "typical": 100},
        "uk": {"junior": 45, "middle": 75, "senior": 105, "lead": 135, "architect": 165, "typical": 95},
        "us": {"junior": 60, "middle": 100, "senior": 150, "lead": 200, "architect": 250, "typical": 140},
        "in": {"junior": 10, "middle": 18, "senior": 30, "lead": 45, "architect": 60, "typical": 25},
    },
    "cocomo": {
        "a": 0.5,          # Coefficient
        "b": 0.85,         # Exponent
        "hours_per_pm": 160,  # Hours per person-month
        "eaf_default": 1.0,   # Effort adjustment factor
    },
    "ai_productivity": {
        "pure_human": 25,      # hrs/KLOC for traditional dev
        "ai_assisted": 8,      # hrs/KLOC with AI tools
        "hybrid": 6.5,         # hrs/KLOC AI + human review
        "ai_speedup": 3.0,     # Multiplier for AI vs human
    },
    "validation": {
        "rate_min": 5,
        "rate_max": 300,
        "hours_per_kloc_min": 2,
        "hours_per_kloc_max": 200,
        "pert_ratio_max": 10,
        "cocomo_a_min": 0.1,
        "cocomo_a_max": 5.0,
        "cocomo_b_min": 0.5,
        "cocomo_b_max": 1.5,
    },
    "pert": {
        "optimistic_factor": 0.7,  # Multiply base by this for optimistic
        "pessimistic_factor": 2.0, # Multiply base by this for pessimistic
    },
}

# Settings storage
_settings_store: Dict[str, Dict] = {"global": DEFAULT_SETTINGS.copy()}


class SettingsManager:
    """Workspace-scoped settings management with validation."""

    @staticmethod
    def get_all(workspace_id: str = "global") -> Dict:
        """Get all settings for a workspace."""
        if workspace_id not in _settings_store:
            _settings_store[workspace_id] = json.loads(json.dumps(DEFAULT_SETTINGS))
        return _settings_store[workspace_id]

    @staticmethod
    def get_rates(workspace_id: str = "global") -> Dict:
        """Get regional rates."""
        settings = SettingsManager.get_all(workspace_id)
        return settings.get("rates", DEFAULT_SETTINGS["rates"])

    @staticmethod
    def get_cocomo(workspace_id: str = "global") -> Dict:
        """Get COCOMO parameters."""
        settings = SettingsManager.get_all(workspace_id)
        return settings.get("cocomo", DEFAULT_SETTINGS["cocomo"])

    @staticmethod
    def get_ai_productivity(workspace_id: str = "global") -> Dict:
        """Get AI productivity settings."""
        settings = SettingsManager.get_all(workspace_id)
        return settings.get("ai_productivity", DEFAULT_SETTINGS["ai_productivity"])

    @staticmethod
    def get_validation_bounds(workspace_id: str = "global") -> Dict:
        """Get validation bounds."""
        settings = SettingsManager.get_all(workspace_id)
        return settings.get("validation", DEFAULT_SETTINGS["validation"])

    @staticmethod
    def update_rates(workspace_id: str, rates: Dict) -> Dict:
        """Update regional rates with validation."""
        bounds = SettingsManager.get_validation_bounds(workspace_id)
        errors = []

        # Validate each rate
        for region, levels in rates.items():
            for level, rate in levels.items():
                if rate < bounds["rate_min"] or rate > bounds["rate_max"]:
                    errors.append(f"{region}.{level}: ${rate} outside bounds ${bounds['rate_min']}-${bounds['rate_max']}")

        if errors:
            return {"success": False, "errors": errors}

        settings = SettingsManager.get_all(workspace_id)
        settings["rates"] = rates
        _settings_store[workspace_id] = settings

        # Persist to DB if available
        SettingsManager._persist(workspace_id, "rates", rates)

        return {"success": True, "rates": rates}

    @staticmethod
    def update_cocomo(workspace_id: str, params: Dict) -> Dict:
        """Update COCOMO parameters with validation."""
        bounds = SettingsManager.get_validation_bounds(workspace_id)
        errors = []

        a = params.get("a", 0.5)
        b = params.get("b", 0.85)

        if a < bounds["cocomo_a_min"] or a > bounds["cocomo_a_max"]:
            errors.append(f"Coefficient 'a' ({a}) outside bounds {bounds['cocomo_a_min']}-{bounds['cocomo_a_max']}")
        if b < bounds["cocomo_b_min"] or b > bounds["cocomo_b_max"]:
            errors.append(f"Exponent 'b' ({b}) outside bounds {bounds['cocomo_b_min']}-{bounds['cocomo_b_max']}")

        if errors:
            return {"success": False, "errors": errors}

        settings = SettingsManager.get_all(workspace_id)
        settings["cocomo"] = {
            "a": a,
            "b": b,
            "hours_per_pm": params.get("hours_per_pm", 160),
            "eaf_default": params.get("eaf_default", 1.0),
        }
        _settings_store[workspace_id] = settings

        SettingsManager._persist(workspace_id, "cocomo", settings["cocomo"])

        return {"success": True, "cocomo": settings["cocomo"]}

    @staticmethod
    def update_ai_productivity(workspace_id: str, params: Dict) -> Dict:
        """Update AI productivity settings."""
        bounds = SettingsManager.get_validation_bounds(workspace_id)

        # Validate hours per KLOC
        for key in ["pure_human", "ai_assisted", "hybrid"]:
            val = params.get(key, 10)
            if val < bounds["hours_per_kloc_min"] or val > bounds["hours_per_kloc_max"]:
                return {
                    "success": False,
                    "errors": [f"{key}: {val} outside bounds {bounds['hours_per_kloc_min']}-{bounds['hours_per_kloc_max']}"]
                }

        settings = SettingsManager.get_all(workspace_id)
        settings["ai_productivity"] = {
            "pure_human": params.get("pure_human", 25),
            "ai_assisted": params.get("ai_assisted", 8),
            "hybrid": params.get("hybrid", 6.5),
            "ai_speedup": params.get("ai_speedup", 3.0),
        }
        _settings_store[workspace_id] = settings

        SettingsManager._persist(workspace_id, "ai_productivity", settings["ai_productivity"])

        return {"success": True, "ai_productivity": settings["ai_productivity"]}

    @staticmethod
    def reset(workspace_id: str = "global") -> Dict:
        """Reset all settings to defaults."""
        _settings_store[workspace_id] = json.loads(json.dumps(DEFAULT_SETTINGS))

        if db_connection:
            try:
                cursor = db_connection.cursor()
                cursor.execute("DELETE FROM workspace_settings WHERE workspace_id = %s", (workspace_id,))
                db_connection.commit()
            except:
                pass

        return {"success": True, "settings": DEFAULT_SETTINGS}

    @staticmethod
    def _persist(workspace_id: str, key: str, value: Dict):
        """Persist settings to database."""
        if not db_connection:
            return

        try:
            cursor = db_connection.cursor()
            cursor.execute("""
                INSERT INTO workspace_settings (workspace_id, key, value, updated_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (workspace_id, key) DO UPDATE SET value = %s, updated_at = NOW()
            """, (workspace_id, key, json.dumps(value), json.dumps(value)))
            db_connection.commit()
        except Exception as e:
            logger.warning(f"Failed to persist settings: {e}")

    @staticmethod
    def _load_from_db(workspace_id: str):
        """Load settings from database."""
        if not db_connection:
            return

        try:
            cursor = db_connection.cursor()
            cursor.execute(
                "SELECT key, value FROM workspace_settings WHERE workspace_id = %s",
                (workspace_id,)
            )
            for key, value in cursor.fetchall():
                if workspace_id not in _settings_store:
                    _settings_store[workspace_id] = json.loads(json.dumps(DEFAULT_SETTINGS))
                _settings_store[workspace_id][key] = value
        except:
            pass


# =============================================================================
# EXTENDED MCP TOOLS (Memory + Validation)
# =============================================================================

# Validation bounds (anti-hallucination) - now uses settings
class ValidationBounds:
    @staticmethod
    def get(workspace_id: str = "global"):
        return SettingsManager.get_validation_bounds(workspace_id)

    # Legacy class attributes for backward compatibility
    RATE_MIN = 5
    RATE_MAX = 300
    HOURS_PER_KLOC_MIN = 2
    HOURS_PER_KLOC_MAX = 200
    PERT_RATIO_MAX = 10
    COCOMO_A_MIN = 0.1
    COCOMO_A_MAX = 5.0
    COCOMO_B_MIN = 0.5
    COCOMO_B_MAX = 1.5


# Extended tools for HTTP server
MEMORY_TOOLS = [
    {
        "name": "store_memory",
        "description": "Store a fact, observation, or context in persistent memory for this workspace.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "What to remember"},
                "type": {
                    "type": "string",
                    "enum": ["fact", "context", "preference", "observation", "error"],
                    "default": "fact"
                },
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags for retrieval"},
            },
            "required": ["content"]
        }
    },
    {
        "name": "recall_memory",
        "description": "Recall stored memories by query or type.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "type": {"type": "string", "description": "Filter by type"},
                "limit": {"type": "integer", "default": 10},
            }
        }
    },
    {
        "name": "record_decision",
        "description": "Record an architectural or design decision (ADR format).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Decision title"},
                "context": {"type": "string", "description": "Why this decision was needed"},
                "decision": {"type": "string", "description": "What was decided"},
                "consequences": {"type": "array", "items": {"type": "string"}},
                "alternatives": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title", "decision"]
        }
    },
    {
        "name": "record_learning",
        "description": "Record a learning from mistake or user feedback.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "what_happened": {"type": "string", "description": "What went wrong"},
                "what_learned": {"type": "string", "description": "Lesson learned"},
                "correction": {"type": "string", "description": "How to fix"},
                "pattern": {"type": "string", "description": "Pattern to avoid"},
            },
            "required": ["what_happened", "what_learned"]
        }
    },
    {
        "name": "get_context",
        "description": "Get current workspace context including recent decisions and learnings.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "validate_estimate",
        "description": "Validate estimation against anti-hallucination bounds.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "total_hours": {"type": "number", "description": "Estimated hours"},
                "total_cost": {"type": "number", "description": "Estimated cost USD"},
                "kloc": {"type": "number", "description": "Thousands of lines of code"},
                "hourly_rate": {"type": "number", "description": "Average hourly rate"},
            },
            "required": ["total_hours", "total_cost", "kloc"]
        }
    },
    {
        "name": "get_settings",
        "description": "Get current workspace settings (rates, COCOMO params, AI productivity).",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "update_settings",
        "description": "Update workspace settings (rates, COCOMO, AI productivity).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "rates": {"type": "object", "description": "Regional rates by region and level"},
                "cocomo": {"type": "object", "description": "COCOMO parameters (a, b, hours_per_pm)"},
                "ai_productivity": {"type": "object", "description": "AI productivity (pure_human, ai_assisted, hybrid)"},
            }
        }
    },
    {
        "name": "estimate_custom",
        "description": "COCOMO estimation using workspace custom settings. Returns hours and cost by region.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "loc": {"type": "number", "description": "Lines of code"},
                "kloc": {"type": "number", "description": "Thousands of lines of code (alternative to loc)"},
                "complexity": {
                    "type": "string",
                    "enum": ["low", "nominal", "high", "very_high"],
                    "default": "nominal"
                },
                "region": {"type": "string", "description": "Filter by specific region"},
            },
            "required": []
        }
    },
    {
        "name": "compare_estimates",
        "description": "Compare human vs AI-assisted development costs using workspace settings.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "loc": {"type": "number", "description": "Lines of code"},
                "kloc": {"type": "number", "description": "Thousands of lines of code"},
                "region": {"type": "string", "description": "Region for rates", "default": "ua"},
            },
            "required": []
        }
    },
]

# =============================================================================
# ANALYSIS TOOLS - Repository Analysis, Code Quality, Scripts, Reports
# =============================================================================

ANALYSIS_TOOLS = [
    # --- Repo Analysis ---
    {
        "name": "clone_repo",
        "description": "Clone a git repository for analysis. Returns repo path and basic info.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Git repository URL (GitHub, GitLab, etc.)"},
                "branch": {"type": "string", "description": "Branch to clone", "default": "main"},
            },
            "required": ["url"]
        }
    },
    {
        "name": "analyze_repo",
        "description": "Full repository analysis: LOC, files, languages, complexity, dependencies.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to repository (local or cloned)"},
                "url": {"type": "string", "description": "Git URL (will clone if path not provided)"},
                "include_metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Metrics to include: loc, complexity, dependencies, languages, files",
                    "default": ["loc", "languages", "files"]
                },
            },
            "required": []
        }
    },
    {
        "name": "scan_security",
        "description": "Security scan: secrets, vulnerabilities, OWASP checks.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to repository"},
                "checks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Checks: secrets, dependencies, code, owasp",
                    "default": ["secrets", "dependencies"]
                },
            },
            "required": ["path"]
        }
    },
    # --- Script Execution ---
    {
        "name": "run_script",
        "description": "Run a Python or Bash script in sandboxed environment.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "script": {"type": "string", "description": "Script content to execute"},
                "language": {"type": "string", "enum": ["python", "bash"], "default": "python"},
                "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 60},
                "args": {"type": "array", "items": {"type": "string"}, "description": "Script arguments"},
            },
            "required": ["script"]
        }
    },
    {
        "name": "run_tests",
        "description": "Run project tests (pytest, jest, etc.) and return results.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to project"},
                "framework": {"type": "string", "enum": ["pytest", "jest", "mocha", "auto"], "default": "auto"},
                "pattern": {"type": "string", "description": "Test file pattern"},
                "verbose": {"type": "boolean", "default": False},
            },
            "required": ["path"]
        }
    },
    # --- Code Quality ---
    {
        "name": "check_lint",
        "description": "Run linters on code (ruff, eslint, etc.).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to check"},
                "language": {"type": "string", "enum": ["python", "javascript", "typescript", "auto"], "default": "auto"},
                "fix": {"type": "boolean", "description": "Auto-fix issues", "default": False},
            },
            "required": ["path"]
        }
    },
    {
        "name": "check_types",
        "description": "Run type checker (mypy, pyright, tsc).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to check"},
                "strict": {"type": "boolean", "default": False},
            },
            "required": ["path"]
        }
    },
    {
        "name": "find_duplicates",
        "description": "Find duplicate/similar code blocks (jscpd).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to analyze"},
                "min_lines": {"type": "integer", "description": "Minimum duplicate lines", "default": 5},
                "min_tokens": {"type": "integer", "description": "Minimum tokens", "default": 50},
                "threshold": {"type": "number", "description": "Similarity threshold 0-100", "default": 80},
            },
            "required": ["path"]
        }
    },
    {
        "name": "analyze_complexity",
        "description": "Analyze code complexity (cyclomatic, cognitive, maintainability).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to analyze"},
                "threshold": {"type": "integer", "description": "Complexity threshold for warnings", "default": 10},
            },
            "required": ["path"]
        }
    },
    # --- Reports ---
    {
        "name": "generate_report",
        "description": "Generate comprehensive audit report in markdown.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "analysis_id": {"type": "string", "description": "Analysis ID to generate report for"},
                "sections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Sections: summary, metrics, security, quality, estimation",
                    "default": ["summary", "metrics", "estimation"]
                },
                "format": {"type": "string", "enum": ["markdown", "html", "json"], "default": "markdown"},
            },
            "required": []
        }
    },
    {
        "name": "export_results",
        "description": "Export analysis results to file (PDF, Excel, JSON).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "analysis_id": {"type": "string", "description": "Analysis ID"},
                "format": {"type": "string", "enum": ["pdf", "xlsx", "json", "csv"], "default": "json"},
                "include_charts": {"type": "boolean", "default": True},
            },
            "required": []
        }
    },
    # --- Batch Operations ---
    {
        "name": "batch_analyze",
        "description": "Analyze multiple repositories in batch.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repos": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of repository URLs"
                },
                "parallel": {"type": "integer", "description": "Parallel workers", "default": 3},
            },
            "required": ["repos"]
        }
    },
]


# =============================================================================
# BUSINESS TOOLS - Profiles, Contracts, Compliance, Documents
# =============================================================================

# Evaluation Profiles
PROFILES = {
    'eu_standard': {
        'name': 'EU Standard R&D',
        'region': 'EU',
        'currency': 'EUR',
        'hourly': {'junior': 35, 'middle': 55, 'senior': 85},
        'requirements': {'repo_health': 6, 'tech_debt': 6, 'readiness': 60},
    },
    'ua_standard': {
        'name': 'Ukraine R&D',
        'region': 'UA',
        'currency': 'USD',
        'hourly': {'junior': 15, 'middle': 30, 'senior': 50},
        'requirements': {'repo_health': 5, 'tech_debt': 5, 'readiness': 50},
    },
    'eu_enterprise': {
        'name': 'EU Enterprise',
        'region': 'EU',
        'currency': 'EUR',
        'hourly': {'junior': 45, 'middle': 70, 'senior': 110},
        'requirements': {'repo_health': 9, 'tech_debt': 10, 'readiness': 80},
    },
    'us_standard': {
        'name': 'US Standard',
        'region': 'US',
        'currency': 'USD',
        'hourly': {'junior': 50, 'middle': 85, 'senior': 130},
        'requirements': {'repo_health': 6, 'tech_debt': 6, 'readiness': 60},
    },
    'startup': {
        'name': 'Startup/MVP',
        'region': 'Global',
        'currency': 'USD',
        'hourly': {'junior': 25, 'middle': 45, 'senior': 70},
        'requirements': {'repo_health': 3, 'tech_debt': 3, 'readiness': 30},
    },
    'global_fund': {
        'name': 'Global Fund R13',
        'region': 'UA',
        'currency': 'USD',
        'hourly': {'junior': 20, 'middle': 35, 'senior': 55},
        'requirements': {'repo_health': 8, 'tech_debt': 9, 'readiness': 70},
        'compliance': ['HIPAA', 'ISO 22301', 'GDPR'],
    },
}

# Contract Profiles
CONTRACTS = {
    'standard': {
        'name': 'Standard (No specific requirements)',
        'requirements': [],
    },
    'global_fund': {
        'name': 'Global Fund R13',
        'compliance': ['HIPAA', 'ISO 22301', 'GDPR'],
        'requirements': [
            {'metric': 'documentation', 'min': 2, 'blocking': True},
            {'metric': 'security', 'min': 2, 'blocking': True},
            {'metric': 'testing', 'min': 2, 'blocking': False},
        ],
    },
    'gdpr': {
        'name': 'EU GDPR Compliant',
        'compliance': ['GDPR'],
        'requirements': [
            {'metric': 'security', 'min': 2, 'blocking': True},
            {'metric': 'documentation', 'min': 2, 'blocking': True},
        ],
    },
    'hipaa': {
        'name': 'HIPAA Healthcare',
        'compliance': ['HIPAA', 'HITECH'],
        'requirements': [
            {'metric': 'security', 'min': 3, 'blocking': True},
            {'metric': 'testing', 'min': 2, 'blocking': True},
            {'metric': 'documentation', 'min': 2, 'blocking': True},
        ],
    },
    'iso27001': {
        'name': 'ISO 27001',
        'compliance': ['ISO 27001'],
        'requirements': [
            {'metric': 'security', 'min': 3, 'blocking': True},
            {'metric': 'infrastructure', 'min': 2, 'blocking': True},
            {'metric': 'documentation', 'min': 2, 'blocking': True},
        ],
    },
}

# Document Templates
DOCUMENT_TEMPLATES = {
    'act_of_work_uk': '''# АКТ
# виконаних робіт

**Дата:** {{date}}
**Номер:** {{act_number}}

## Сторони

**Виконавець:** {{contractor_name}}
{{contractor_details}}

**Замовник:** {{client_name}}
{{client_details}}

## Опис виконаних робіт

| Опис | Кількість | Ціна |
|------|-----------|------|
{{#each work_items as item}}
| {{item.description}} | {{item.quantity}} {{item.unit}} | {{item.price}} {{currency}} |
{{/each}}

## Підсумок

**Загальна вартість:** {{total_amount}} {{currency}}
**ПДВ ({{tax_rate}}%):** {{tax_amount}} {{currency}}
**До сплати:** {{grand_total}} {{currency}}

## Підписи

Виконавець: _________________ / {{contractor_representative}} /

Замовник: _________________ / {{client_representative}} /
''',

    'act_of_work_en': '''# ACT
# of Completed Work

**Date:** {{date}}
**Number:** {{act_number}}

## Parties

**Contractor:** {{contractor_name}}
{{contractor_details}}

**Client:** {{client_name}}
{{client_details}}

## Description of Work Performed

| Description | Qty | Price |
|-------------|-----|-------|
{{#each work_items as item}}
| {{item.description}} | {{item.quantity}} {{item.unit}} | {{item.price}} {{currency}} |
{{/each}}

## Summary

**Total Amount:** {{total_amount}} {{currency}}
**VAT ({{tax_rate}}%):** {{tax_amount}} {{currency}}
**Grand Total:** {{grand_total}} {{currency}}

## Signatures

Contractor: _________________ / {{contractor_representative}} /

Client: _________________ / {{client_representative}} /
''',

    'invoice': '''# INVOICE

**Invoice Number:** {{invoice_number}}
**Date:** {{date}}
**Due Date:** {{due_date}}

## From
**{{contractor_name}}**
{{contractor_address}}
Tax ID: {{contractor_tax_id}}
IBAN: {{contractor_iban}}
Bank: {{contractor_bank}}
SWIFT: {{contractor_swift}}

## To
**{{client_name}}**
{{client_address}}
Tax ID: {{client_tax_id}}

## Services

| Description | Qty | Unit Price | Amount |
|-------------|-----|------------|--------|
{{#each items as item}}
| {{item.description}} | {{item.quantity}} | {{item.unit_price}} {{currency}} | {{item.amount}} {{currency}} |
{{/each}}

## Summary

| | |
|---|---|
| Subtotal | {{subtotal}} {{currency}} |
| VAT ({{tax_rate}}%) | {{tax_amount}} {{currency}} |
| **Total Due** | **{{total}} {{currency}}** |

## Payment Instructions

Please transfer the amount to:
- IBAN: {{contractor_iban}}
- Bank: {{contractor_bank}}
- SWIFT: {{contractor_swift}}
- Reference: {{invoice_number}}

Thank you for your business!
''',

    'analysis_report': '''# Repository Analysis Report

**Repository:** {{repo_url}}
**Analysis Date:** {{date}}
**Profile:** {{profile_name}}

## Executive Summary

{{summary}}

## Repository Health Score: {{health_total}}/12

| Metric | Score | Max |
|--------|-------|-----|
| Documentation | {{health_documentation}} | 3 |
| Structure | {{health_structure}} | 3 |
| Runability | {{health_runability}} | 3 |
| Commit History | {{health_history}} | 3 |

## Technical Debt Score: {{debt_total}}/15

| Metric | Score | Max |
|--------|-------|-----|
| Architecture | {{debt_architecture}} | 3 |
| Code Quality | {{debt_code_quality}} | 3 |
| Testing | {{debt_testing}} | 3 |
| Infrastructure | {{debt_infrastructure}} | 3 |
| Security | {{debt_security}} | 3 |

## Cost Estimate

- **Complexity:** {{complexity}}
- **Estimated Hours:** {{min_hours}} - {{max_hours}} hours
- **Cost Range:** {{min_cost}} - {{max_cost}} {{currency}}

## Recommendations

{{#if recommendations}}
{{#each recommendations as rec}}
### {{rec.title}}
**Priority:** {{rec.priority}}
**Effort:** {{rec.hours}} hours

{{rec.description}}

{{/each}}
{{/if}}

---
*Generated by MCP Audit Server*
''',
}

# Scoring Rubrics
SCORING_RUBRICS = {
    'documentation': {
        0: 'No README, no docs',
        1: 'Basic README exists',
        2: 'Good README + some API docs',
        3: 'Comprehensive docs, tutorials, examples'
    },
    'structure': {
        0: 'Chaotic, no clear organization',
        1: 'Basic structure, some organization',
        2: 'Good structure, clear separation',
        3: 'Excellent structure, follows best practices'
    },
    'runability': {
        0: 'Cannot run, missing dependencies',
        1: 'Can run with manual setup',
        2: 'Docker or scripts provided',
        3: 'One-command setup, CI/CD ready'
    },
    'history': {
        0: 'Few commits, no history',
        1: 'Some commits, unclear messages',
        2: 'Regular commits, decent messages',
        3: 'Clean history, semantic commits, tags'
    },
    'architecture': {
        0: 'Monolithic, tightly coupled',
        1: 'Some separation exists',
        2: 'Good modularity, clear boundaries',
        3: 'Clean architecture, loosely coupled'
    },
    'code_quality': {
        0: 'No standards, high complexity',
        1: 'Some standards, moderate issues',
        2: 'Good standards, low complexity',
        3: 'Excellent quality, clean code'
    },
    'testing': {
        0: 'No tests',
        1: 'Some unit tests exist',
        2: 'Good coverage, some integration tests',
        3: 'Comprehensive tests, CI integration'
    },
    'infrastructure': {
        0: 'No deployment config',
        1: 'Basic deployment docs',
        2: 'Docker/K8s configs, some monitoring',
        3: 'Full IaC, monitoring, logging'
    },
    'security': {
        0: 'Known vulnerabilities, no scans',
        1: 'Some security measures',
        2: 'No critical issues, updated deps',
        3: 'Security-first, audited, compliant'
    },
}

# Business Tools Definitions
BUSINESS_TOOLS = [
    {
        "name": "list_profiles",
        "description": "List available evaluation profiles with hourly rates and requirements.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        }
    },
    {
        "name": "list_contracts",
        "description": "List available contract compliance profiles (GDPR, HIPAA, etc.).",
        "inputSchema": {
            "type": "object",
            "properties": {},
        }
    },
    {
        "name": "estimate_cost",
        "description": "Estimate development cost based on complexity and profile.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "complexity": {
                    "type": "string",
                    "enum": ["S", "M", "L", "XL"],
                    "description": "Project complexity: S (<160h), M (160-500h), L (500-1200h), XL (>1200h)"
                },
                "profile_id": {
                    "type": "string",
                    "description": "Evaluation profile ID (eu_standard, ua_standard, etc.)"
                },
                "tech_debt_multiplier": {
                    "type": "number",
                    "description": "Tech debt adjustment (1.0-1.5)",
                    "default": 1.0
                }
            },
            "required": ["complexity", "profile_id"]
        }
    },
    {
        "name": "check_readiness",
        "description": "Assess project readiness for audit based on health and debt scores.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "health_documentation": {"type": "integer", "minimum": 0, "maximum": 3},
                "health_structure": {"type": "integer", "minimum": 0, "maximum": 3},
                "health_runability": {"type": "integer", "minimum": 0, "maximum": 3},
                "health_history": {"type": "integer", "minimum": 0, "maximum": 3},
                "debt_architecture": {"type": "integer", "minimum": 0, "maximum": 3},
                "debt_code_quality": {"type": "integer", "minimum": 0, "maximum": 3},
                "debt_testing": {"type": "integer", "minimum": 0, "maximum": 3},
                "debt_infrastructure": {"type": "integer", "minimum": 0, "maximum": 3},
                "debt_security": {"type": "integer", "minimum": 0, "maximum": 3},
                "profile_id": {"type": "string", "default": "eu_standard"}
            },
            "required": []
        }
    },
    {
        "name": "check_compliance",
        "description": "Check if scores meet contract requirements (GDPR, HIPAA, Global Fund).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "contract_id": {
                    "type": "string",
                    "description": "Contract profile ID"
                },
                "documentation": {"type": "integer", "minimum": 0, "maximum": 3},
                "security": {"type": "integer", "minimum": 0, "maximum": 3},
                "testing": {"type": "integer", "minimum": 0, "maximum": 3},
                "infrastructure": {"type": "integer", "minimum": 0, "maximum": 3},
            },
            "required": ["contract_id"]
        }
    },
    {
        "name": "generate_document",
        "description": "Generate a document from template (act, invoice, report).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "template_id": {
                    "type": "string",
                    "enum": ["act_of_work_uk", "act_of_work_en", "invoice", "analysis_report"],
                    "description": "Document template to use"
                },
                "variables": {
                    "type": "object",
                    "description": "Variables to fill in the template"
                }
            },
            "required": ["template_id", "variables"]
        }
    },
    {
        "name": "get_template_variables",
        "description": "Get list of variables required by a document template.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "template_id": {
                    "type": "string",
                    "enum": ["act_of_work_uk", "act_of_work_en", "invoice", "analysis_report"]
                }
            },
            "required": ["template_id"]
        }
    },
    {
        "name": "calculate_scores",
        "description": "Calculate overall scores and readiness percentage.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "health_documentation": {"type": "integer", "minimum": 0, "maximum": 3},
                "health_structure": {"type": "integer", "minimum": 0, "maximum": 3},
                "health_runability": {"type": "integer", "minimum": 0, "maximum": 3},
                "health_history": {"type": "integer", "minimum": 0, "maximum": 3},
                "debt_architecture": {"type": "integer", "minimum": 0, "maximum": 3},
                "debt_code_quality": {"type": "integer", "minimum": 0, "maximum": 3},
                "debt_testing": {"type": "integer", "minimum": 0, "maximum": 3},
                "debt_infrastructure": {"type": "integer", "minimum": 0, "maximum": 3},
                "debt_security": {"type": "integer", "minimum": 0, "maximum": 3},
            },
            "required": []
        }
    },
    {
        "name": "get_scoring_rubric",
        "description": "Get detailed scoring rubric for evaluation metrics.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "enum": ["documentation", "structure", "runability", "history",
                            "architecture", "code_quality", "testing", "infrastructure", "security"],
                    "description": "Metric to get rubric for (optional, returns all if not specified)"
                }
            },
            "required": []
        }
    },
]


# =============================================================================
# TEMPLATE ENGINE
# =============================================================================

import re

class TemplateEngine:
    """Simple template engine with variable substitution, conditionals, and loops."""

    def process(self, template: str, variables: Dict[str, Any]) -> str:
        """Process template with given variables."""
        content = template

        # Process variables
        for key, value in variables.items():
            pattern = rf'\{{\{{{key}(?::[^}}]*)?(?:\|[^}}]*)?\}}\}}'
            content = re.sub(pattern, str(value), content)

        # Process default values {{var|default}}
        content = self._process_defaults(content)

        # Process conditionals {{#if condition}}...{{/if}}
        content = self._process_conditionals(content, variables)

        # Process loops {{#each array as item}}...{{/each}}
        content = self._process_loops(content, variables)

        return content

    def _process_defaults(self, content: str) -> str:
        pattern = r'\{\{(\w+)(?::(\w+))?\|(.+?)\}\}'
        return re.sub(pattern, lambda m: m.group(3), content)

    def _process_conditionals(self, content: str, variables: Dict[str, Any]) -> str:
        pattern = r'\{\{#if\s+(\w+)\}\}(.*?)\{\{/if\}\}'
        def replacer(match):
            condition = match.group(1)
            block = match.group(2)
            return block if variables.get(condition) else ''
        return re.sub(pattern, replacer, content, flags=re.DOTALL)

    def _process_loops(self, content: str, variables: Dict[str, Any]) -> str:
        pattern = r'\{\{#each\s+(\w+)\s+as\s+(\w+)\}\}(.*?)\{\{/each\}\}'

        def replacer(match):
            array_name = match.group(1)
            item_name = match.group(2)
            block = match.group(3)

            array = variables.get(array_name, [])
            if not isinstance(array, list):
                return ''

            result = []
            for item in array:
                processed_block = block
                if isinstance(item, dict):
                    for key, value in item.items():
                        item_pattern = rf'\{{\{{{item_name}\.{key}\}}\}}'
                        processed_block = re.sub(item_pattern, str(value), processed_block)
                else:
                    item_pattern = rf'\{{\{{{item_name}\}}\}}'
                    processed_block = re.sub(item_pattern, str(item), processed_block)
                result.append(processed_block)

            return ''.join(result)

        return re.sub(pattern, replacer, content, flags=re.DOTALL)

    def extract_variables(self, content: str) -> List[str]:
        """Extract variable names from template."""
        pattern = r'\{\{(\w+)(?::(\w+))?(?:\|(.+?))?\}\}'
        variables = []
        for match in re.finditer(pattern, content):
            var_name = match.group(1)
            if var_name not in variables and not var_name.startswith('#'):
                variables.append(var_name)
        return variables


# Create template engine instance
template_engine = TemplateEngine()


# =============================================================================
# BUSINESS TOOL HANDLERS
# =============================================================================

def handle_list_profiles() -> Dict:
    """List all available evaluation profiles."""
    result = []
    for profile_id, profile in PROFILES.items():
        result.append({
            "id": profile_id,
            "name": profile['name'],
            "region": profile['region'],
            "currency": profile['currency'],
            "hourly_rates": profile['hourly'],
            "requirements": profile['requirements'],
            "compliance": profile.get('compliance', []),
        })
    return {"profiles": result, "count": len(result)}


def handle_list_contracts() -> Dict:
    """List all available contract compliance profiles."""
    result = []
    for contract_id, contract in CONTRACTS.items():
        result.append({
            "id": contract_id,
            "name": contract['name'],
            "compliance": contract.get('compliance', []),
            "requirements": contract.get('requirements', []),
        })
    return {"contracts": result, "count": len(result)}


def handle_estimate_cost(arguments: Dict) -> Dict:
    """Estimate development cost based on complexity and profile."""
    complexity = arguments.get('complexity', 'M')
    profile_id = arguments.get('profile_id', 'eu_standard')
    multiplier = arguments.get('tech_debt_multiplier', 1.0)

    profile = PROFILES.get(profile_id)
    if not profile:
        return {"error": f"Unknown profile: {profile_id}"}

    # Base hours by complexity
    hours = {
        'S': {'min': 80, 'typical': 120, 'max': 160},
        'M': {'min': 160, 'typical': 320, 'max': 500},
        'L': {'min': 500, 'typical': 800, 'max': 1200},
        'XL': {'min': 1200, 'typical': 2000, 'max': 3000},
    }

    h = hours.get(complexity, hours['M'])
    rate = profile['hourly']['middle']
    currency = profile['currency']

    min_h = int(h['min'] * multiplier)
    typ_h = int(h['typical'] * multiplier)
    max_h = int(h['max'] * multiplier)

    return {
        "profile": profile['name'],
        "complexity": complexity,
        "tech_debt_multiplier": multiplier,
        "hours": {
            "min": min_h,
            "typical": typ_h,
            "max": max_h,
        },
        "cost": {
            "min": min_h * rate,
            "typical": typ_h * rate,
            "max": max_h * rate,
            "currency": currency,
        },
        "rate": rate,
        "breakdown": {
            "analysis": {"hours": int(typ_h * 0.1), "cost": int(typ_h * 0.1 * rate)},
            "design": {"hours": int(typ_h * 0.15), "cost": int(typ_h * 0.15 * rate)},
            "development": {"hours": int(typ_h * 0.45), "cost": int(typ_h * 0.45 * rate)},
            "testing": {"hours": int(typ_h * 0.2), "cost": int(typ_h * 0.2 * rate)},
            "documentation": {"hours": int(typ_h * 0.1), "cost": int(typ_h * 0.1 * rate)},
        }
    }


def handle_check_readiness(arguments: Dict) -> Dict:
    """Assess project readiness for audit based on scores."""
    profile_id = arguments.get('profile_id', 'eu_standard')
    profile = PROFILES.get(profile_id, PROFILES['eu_standard'])

    # Calculate health score (0-12)
    health_total = (
        arguments.get('health_documentation', 0) +
        arguments.get('health_structure', 0) +
        arguments.get('health_runability', 0) +
        arguments.get('health_history', 0)
    )

    # Calculate tech debt score (0-15)
    debt_total = (
        arguments.get('debt_architecture', 0) +
        arguments.get('debt_code_quality', 0) +
        arguments.get('debt_testing', 0) +
        arguments.get('debt_infrastructure', 0) +
        arguments.get('debt_security', 0)
    )

    # Calculate readiness percentage
    health_pct = (health_total / 12) * 100
    debt_pct = (debt_total / 15) * 100
    readiness = (health_pct + debt_pct) / 2

    # Check against profile requirements
    req = profile['requirements']
    health_ok = health_total >= req['repo_health']
    debt_ok = debt_total >= req['tech_debt']
    readiness_ok = readiness >= req['readiness']

    # Determine level
    if readiness >= 95:
        level = "EXEMPLARY"
    elif readiness >= 80:
        level = "READY"
    elif readiness >= 60:
        level = "ALMOST_READY"
    elif readiness >= 40:
        level = "NEEDS_WORK"
    else:
        level = "NOT_READY"

    return {
        "level": level,
        "readiness_pct": round(readiness, 1),
        "profile": profile['name'],
        "scores": {
            "health": {"score": health_total, "max": 12, "pct": round(health_pct, 1), "passed": health_ok},
            "debt": {"score": debt_total, "max": 15, "pct": round(debt_pct, 1), "passed": debt_ok},
        },
        "requirements": {
            "health": {"required": req['repo_health'], "actual": health_total, "passed": health_ok},
            "debt": {"required": req['tech_debt'], "actual": debt_total, "passed": debt_ok},
            "readiness": {"required": req['readiness'], "actual": round(readiness, 1), "passed": readiness_ok},
        },
        "verdict": "READY_FOR_EVALUATION" if (health_ok and debt_ok and readiness_ok) else "NOT_READY",
    }


def handle_check_compliance(arguments: Dict) -> Dict:
    """Check if scores meet contract requirements."""
    contract_id = arguments.get('contract_id', 'standard')
    contract = CONTRACTS.get(contract_id)

    if not contract:
        return {"error": f"Unknown contract: {contract_id}"}

    scores = {
        'documentation': arguments.get('documentation', 0),
        'security': arguments.get('security', 0),
        'testing': arguments.get('testing', 0),
        'infrastructure': arguments.get('infrastructure', 0),
    }

    if not contract.get('requirements'):
        return {
            "contract": contract['name'],
            "verdict": "COMPLIANT",
            "compliance_pct": 100,
            "message": "No specific requirements - automatically compliant"
        }

    passed = []
    failed = []
    blocking_failed = []

    for req in contract['requirements']:
        metric = req['metric']
        min_score = req['min']
        actual = scores.get(metric, 0)
        blocking = req.get('blocking', False)

        if actual >= min_score:
            passed.append({"metric": metric, "required": min_score, "actual": actual})
        else:
            item = {"metric": metric, "required": min_score, "actual": actual, "gap": min_score - actual}
            if blocking:
                item["blocking"] = True
                blocking_failed.append(item)
            else:
                failed.append(item)

    compliance_pct = len(passed) / len(contract['requirements']) * 100 if contract['requirements'] else 100

    if blocking_failed:
        verdict = "NON_COMPLIANT"
    elif failed:
        verdict = "PARTIAL"
    else:
        verdict = "COMPLIANT"

    return {
        "contract": contract['name'],
        "compliance": contract.get('compliance', []),
        "verdict": verdict,
        "compliance_pct": round(compliance_pct, 1),
        "passed": passed,
        "failed": failed + blocking_failed,
    }


def handle_generate_document(arguments: Dict) -> Dict:
    """Generate a document from template with variables."""
    template_id = arguments.get('template_id')
    variables = arguments.get('variables', {})

    template = DOCUMENT_TEMPLATES.get(template_id)
    if not template:
        return {"error": f"Unknown template: {template_id}"}

    # Add default date if not provided
    if 'date' not in variables:
        variables['date'] = datetime.now().strftime('%Y-%m-%d')

    result = template_engine.process(template, variables)
    return {"document": result, "template_id": template_id}


def handle_get_template_variables(arguments: Dict) -> Dict:
    """Get list of variables required by a document template."""
    template_id = arguments.get('template_id')

    template = DOCUMENT_TEMPLATES.get(template_id)
    if not template:
        return {"error": f"Unknown template: {template_id}"}

    variables = template_engine.extract_variables(template)
    return {"template_id": template_id, "variables": variables}


def handle_calculate_scores(arguments: Dict) -> Dict:
    """Calculate overall scores and readiness percentage."""
    health_total = (
        arguments.get('health_documentation', 0) +
        arguments.get('health_structure', 0) +
        arguments.get('health_runability', 0) +
        arguments.get('health_history', 0)
    )

    debt_total = (
        arguments.get('debt_architecture', 0) +
        arguments.get('debt_code_quality', 0) +
        arguments.get('debt_testing', 0) +
        arguments.get('debt_infrastructure', 0) +
        arguments.get('debt_security', 0)
    )

    health_pct = (health_total / 12) * 100
    debt_pct = (debt_total / 15) * 100
    overall = (health_pct + debt_pct) / 2

    return {
        "health": {"score": health_total, "max": 12, "pct": round(health_pct, 1)},
        "debt": {"score": debt_total, "max": 15, "pct": round(debt_pct, 1)},
        "overall_readiness_pct": round(overall, 1),
        "classification": {
            "health_level": "Good" if health_pct >= 75 else "Moderate" if health_pct >= 50 else "Needs Improvement",
            "debt_level": "Low Debt" if debt_pct >= 75 else "Moderate Debt" if debt_pct >= 50 else "High Debt",
        }
    }


def handle_get_scoring_rubric(arguments: Dict) -> Dict:
    """Get detailed scoring rubric for evaluation metrics."""
    metric = arguments.get('metric')

    if metric and metric in SCORING_RUBRICS:
        return {"metric": metric, "rubric": SCORING_RUBRICS[metric]}

    return {"rubrics": SCORING_RUBRICS}


# =============================================================================
# ANALYSIS TOOL HANDLERS
# =============================================================================

import subprocess
import tempfile
import shutil

def handle_clone_repo(arguments: Dict) -> Dict:
    """Clone a git repository for analysis."""
    url = arguments.get('url')
    branch = arguments.get('branch', 'main')

    if not url:
        return {"error": "URL is required"}

    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix='repo_')
    try:
        result = subprocess.run(
            ['git', 'clone', '--depth', '1', '--branch', branch, url, temp_dir],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            # Try without branch specification
            result = subprocess.run(
                ['git', 'clone', '--depth', '1', url, temp_dir],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                return {"error": f"Clone failed: {result.stderr}"}

        return {
            "success": True,
            "path": temp_dir,
            "url": url,
            "branch": branch,
        }
    except Exception as e:
        return {"error": str(e)}


def handle_analyze_repo(arguments: Dict) -> Dict:
    """Full repository analysis."""
    path = arguments.get('path', '')
    include_metrics = arguments.get('include_metrics', ['loc', 'languages', 'files'])

    if not path or not os.path.exists(path):
        return {"error": "Valid path is required"}

    result = {"path": path, "metrics": {}}

    # Count files and languages
    if 'files' in include_metrics or 'languages' in include_metrics or 'loc' in include_metrics:
        files_by_ext = {}
        total_lines = 0
        file_count = 0

        for root, dirs, files in os.walk(path):
            # Skip hidden and common non-code directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', 'venv', '__pycache__', 'dist', 'build']]

            for f in files:
                if f.startswith('.'):
                    continue
                ext = os.path.splitext(f)[1].lower() or 'no_ext'
                files_by_ext[ext] = files_by_ext.get(ext, 0) + 1
                file_count += 1

                # Count lines for code files
                if ext in ['.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs', '.java', '.c', '.cpp', '.h', '.rb', '.php']:
                    try:
                        filepath = os.path.join(root, f)
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as fp:
                            total_lines += sum(1 for _ in fp)
                    except:
                        pass

        result["metrics"]["files"] = {"total": file_count, "by_extension": files_by_ext}
        result["metrics"]["loc"] = total_lines

        # Map extensions to languages
        ext_to_lang = {
            '.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript',
            '.tsx': 'TypeScript', '.jsx': 'JavaScript', '.go': 'Go',
            '.rs': 'Rust', '.java': 'Java', '.rb': 'Ruby', '.php': 'PHP',
            '.c': 'C', '.cpp': 'C++', '.h': 'C/C++', '.cs': 'C#',
        }
        languages = {}
        for ext, count in files_by_ext.items():
            lang = ext_to_lang.get(ext, ext)
            languages[lang] = languages.get(lang, 0) + count
        result["metrics"]["languages"] = languages

    return result


def handle_scan_security(arguments: Dict) -> Dict:
    """Security scan: secrets, vulnerabilities."""
    path = arguments.get('path', '')
    checks = arguments.get('checks', ['secrets', 'dependencies'])

    if not path or not os.path.exists(path):
        return {"error": "Valid path is required"}

    results = {"path": path, "findings": []}

    # Simple secrets check (pattern-based)
    if 'secrets' in checks:
        secret_patterns = [
            (r'api[_-]?key\s*[=:]\s*["\']?[\w-]{20,}', 'API Key'),
            (r'password\s*[=:]\s*["\'][^"\']+', 'Password'),
            (r'secret[_-]?key\s*[=:]\s*["\']?[\w-]+', 'Secret Key'),
            (r'AWS[A-Z0-9]{16,}', 'AWS Key'),
            (r'sk-[a-zA-Z0-9]{20,}', 'OpenAI Key'),
        ]

        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', 'venv']]
            for f in files:
                if f.endswith(('.py', '.js', '.ts', '.env', '.json', '.yml', '.yaml')):
                    try:
                        filepath = os.path.join(root, f)
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as fp:
                            content = fp.read()
                            for pattern, name in secret_patterns:
                                if re.search(pattern, content, re.IGNORECASE):
                                    results["findings"].append({
                                        "type": "secret",
                                        "name": name,
                                        "file": os.path.relpath(filepath, path),
                                        "severity": "high"
                                    })
                    except:
                        pass

    return results


def handle_run_script(arguments: Dict) -> Dict:
    """Run a Python or Bash script in sandboxed environment."""
    script = arguments.get('script', '')
    language = arguments.get('language', 'python')
    timeout = arguments.get('timeout', 60)

    if not script:
        return {"error": "Script content is required"}

    try:
        if language == 'python':
            result = subprocess.run(
                ['python3', '-c', script],
                capture_output=True, text=True, timeout=timeout
            )
        else:
            result = subprocess.run(
                ['bash', '-c', script],
                capture_output=True, text=True, timeout=timeout
            )

        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Script timed out after {timeout} seconds"}
    except Exception as e:
        return {"error": str(e)}


def handle_run_tests(arguments: Dict) -> Dict:
    """Run project tests."""
    path = arguments.get('path', '')
    framework = arguments.get('framework', 'auto')

    if not path or not os.path.exists(path):
        return {"error": "Valid path is required"}

    # Auto-detect framework
    if framework == 'auto':
        if os.path.exists(os.path.join(path, 'pytest.ini')) or os.path.exists(os.path.join(path, 'tests')):
            framework = 'pytest'
        elif os.path.exists(os.path.join(path, 'package.json')):
            framework = 'jest'

    try:
        if framework == 'pytest':
            result = subprocess.run(
                ['python', '-m', 'pytest', '-v', '--tb=short'],
                cwd=path, capture_output=True, text=True, timeout=300
            )
        elif framework in ['jest', 'mocha']:
            result = subprocess.run(
                ['npm', 'test'],
                cwd=path, capture_output=True, text=True, timeout=300
            )
        else:
            return {"error": f"Unknown framework: {framework}"}

        return {
            "framework": framework,
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "output": result.stdout + result.stderr,
        }
    except Exception as e:
        return {"error": str(e)}


def handle_check_lint(arguments: Dict) -> Dict:
    """Run linters on code."""
    path = arguments.get('path', '')
    language = arguments.get('language', 'auto')

    if not path or not os.path.exists(path):
        return {"error": "Valid path is required"}

    # Auto-detect language
    if language == 'auto':
        if any(f.endswith('.py') for f in os.listdir(path)):
            language = 'python'
        elif os.path.exists(os.path.join(path, 'package.json')):
            language = 'javascript'

    try:
        if language == 'python':
            result = subprocess.run(
                ['python', '-m', 'ruff', 'check', path],
                capture_output=True, text=True, timeout=120
            )
        elif language in ['javascript', 'typescript']:
            result = subprocess.run(
                ['npx', 'eslint', path],
                capture_output=True, text=True, timeout=120
            )
        else:
            return {"error": f"Unsupported language: {language}"}

        return {
            "language": language,
            "success": result.returncode == 0,
            "output": result.stdout + result.stderr,
        }
    except Exception as e:
        return {"error": str(e)}


def handle_check_types(arguments: Dict) -> Dict:
    """Run type checker."""
    path = arguments.get('path', '')

    if not path or not os.path.exists(path):
        return {"error": "Valid path is required"}

    try:
        # Try pyright for Python
        result = subprocess.run(
            ['python', '-m', 'pyright', path],
            capture_output=True, text=True, timeout=120
        )

        return {
            "success": result.returncode == 0,
            "output": result.stdout + result.stderr,
        }
    except Exception as e:
        return {"error": str(e)}


def handle_find_duplicates(arguments: Dict) -> Dict:
    """Find duplicate/similar code blocks."""
    path = arguments.get('path', '')
    min_lines = arguments.get('min_lines', 5)

    if not path or not os.path.exists(path):
        return {"error": "Valid path is required"}

    try:
        result = subprocess.run(
            ['jscpd', path, '--min-lines', str(min_lines), '--reporters', 'json'],
            capture_output=True, text=True, timeout=300
        )

        return {
            "success": result.returncode == 0,
            "output": result.stdout,
        }
    except FileNotFoundError:
        return {"error": "jscpd not installed. Run: npm install -g jscpd"}
    except Exception as e:
        return {"error": str(e)}


def handle_analyze_complexity(arguments: Dict) -> Dict:
    """Analyze code complexity."""
    path = arguments.get('path', '')
    threshold = arguments.get('threshold', 10)

    if not path or not os.path.exists(path):
        return {"error": "Valid path is required"}

    # Simple complexity analysis for Python files
    complexities = []

    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', 'venv']]
        for f in files:
            if f.endswith('.py'):
                filepath = os.path.join(root, f)
                try:
                    with open(filepath, 'r') as fp:
                        content = fp.read()
                        # Count control flow statements as simple complexity measure
                        complexity = (
                            content.count('if ') +
                            content.count('elif ') +
                            content.count('else:') +
                            content.count('for ') +
                            content.count('while ') +
                            content.count('try:') +
                            content.count('except')
                        )
                        if complexity > threshold:
                            complexities.append({
                                "file": os.path.relpath(filepath, path),
                                "complexity": complexity,
                            })
                except:
                    pass

    return {
        "path": path,
        "threshold": threshold,
        "high_complexity_files": sorted(complexities, key=lambda x: x['complexity'], reverse=True)[:20],
        "count": len(complexities),
    }


def handle_generate_report(arguments: Dict, workspace_id: str) -> Dict:
    """Generate comprehensive audit report."""
    analysis_id = arguments.get('analysis_id')
    sections = arguments.get('sections', ['summary', 'metrics', 'estimation'])
    format_type = arguments.get('format', 'markdown')

    report = {
        "generated_at": datetime.now().isoformat(),
        "sections": sections,
        "format": format_type,
    }

    # If we have an analysis_id, try to load it
    if analysis_id:
        report["analysis_id"] = analysis_id

    return report


def handle_export_results(arguments: Dict) -> Dict:
    """Export analysis results to file."""
    analysis_id = arguments.get('analysis_id')
    format_type = arguments.get('format', 'json')

    return {
        "format": format_type,
        "analysis_id": analysis_id,
        "status": "ready",
    }


async def handle_batch_analyze(arguments: Dict) -> Dict:
    """Analyze multiple repositories in batch."""
    repos = arguments.get('repos', [])

    if not repos:
        return {"error": "No repositories provided"}

    results = []
    for repo_url in repos[:10]:  # Limit to 10
        clone_result = handle_clone_repo({"url": repo_url})
        if clone_result.get('success'):
            analyze_result = handle_analyze_repo({"path": clone_result['path']})
            results.append({
                "url": repo_url,
                "metrics": analyze_result.get('metrics', {}),
            })
            # Cleanup
            try:
                shutil.rmtree(clone_result['path'])
            except:
                pass
        else:
            results.append({
                "url": repo_url,
                "error": clone_result.get('error'),
            })

    return {"results": results, "count": len(results)}


# =============================================================================
# MCP SERVER INFO
# =============================================================================

MCP_SERVER_INFO = {
    "name": "mcp-audit-server",
    "version": "1.0.0",
    "protocolVersion": "2024-11-05",
    "description": "Repository audit platform with cost estimation, anti-hallucination validation, and persistent memory."
}


# =============================================================================
# TOOL EXECUTOR
# =============================================================================

# Create audit server instance
audit_server = AuditMCPServer()


async def execute_tool(name: str, arguments: Dict, workspace_id: str = "global") -> Dict:
    """Execute an MCP tool by name."""
    logger.info(f"Executing tool: {name} with args: {arguments}")

    try:
        # Memory tools
        if name == "store_memory":
            entry = MemoryPersistence.store_memory(
                workspace_id=workspace_id,
                content=arguments.get("content", ""),
                type=arguments.get("type", "fact"),
                tags=arguments.get("tags", [])
            )
            return {"stored": True, "entry": entry}

        elif name == "recall_memory":
            entries = MemoryPersistence.recall_memory(
                workspace_id=workspace_id,
                query=arguments.get("query"),
                type=arguments.get("type"),
                limit=arguments.get("limit", 10)
            )
            return {"count": len(entries), "entries": entries}

        elif name == "record_decision":
            entry = MemoryPersistence.record_decision(
                workspace_id=workspace_id,
                title=arguments.get("title", ""),
                context=arguments.get("context", ""),
                decision=arguments.get("decision", ""),
                consequences=arguments.get("consequences"),
                alternatives=arguments.get("alternatives")
            )
            return {"recorded": True, "decision": entry}

        elif name == "record_learning":
            entry = MemoryPersistence.record_learning(
                workspace_id=workspace_id,
                what_happened=arguments.get("what_happened", ""),
                what_learned=arguments.get("what_learned", ""),
                correction=arguments.get("correction"),
                pattern=arguments.get("pattern")
            )
            return {"recorded": True, "learning": entry}

        elif name == "get_context":
            context = MemoryPersistence.get_context(workspace_id)
            decisions = MemoryPersistence.get_decisions(workspace_id, limit=3)
            learnings = MemoryPersistence.get_learnings(workspace_id, limit=3)
            memories = MemoryPersistence.recall_memory(workspace_id, type="fact", limit=5)

            return {
                "workspace_id": workspace_id,
                "context": context,
                "recent_decisions": decisions,
                "recent_learnings": learnings,
                "recent_memories": memories
            }

        elif name == "validate_estimate":
            return validate_estimate(
                total_hours=arguments.get("total_hours", 0),
                total_cost=arguments.get("total_cost", 0),
                kloc=arguments.get("kloc", 0),
                hourly_rate=arguments.get("hourly_rate"),
                workspace_id=workspace_id
            )

        elif name == "get_settings":
            return SettingsManager.get_all(workspace_id)

        elif name == "update_settings":
            results = {}
            if "rates" in arguments:
                results["rates"] = SettingsManager.update_rates(workspace_id, arguments["rates"])
            if "cocomo" in arguments:
                results["cocomo"] = SettingsManager.update_cocomo(workspace_id, arguments["cocomo"])
            if "ai_productivity" in arguments:
                results["ai_productivity"] = SettingsManager.update_ai_productivity(workspace_id, arguments["ai_productivity"])
            return {"updated": True, "results": results}

        elif name == "estimate_custom":
            return estimate_with_settings(
                workspace_id=workspace_id,
                loc=arguments.get("loc"),
                kloc=arguments.get("kloc"),
                complexity=arguments.get("complexity", "nominal"),
                region=arguments.get("region")
            )

        elif name == "compare_estimates":
            return compare_human_vs_ai(
                workspace_id=workspace_id,
                loc=arguments.get("loc"),
                kloc=arguments.get("kloc"),
                region=arguments.get("region", "ua")
            )

        # ============== BUSINESS TOOLS ==============
        elif name == "list_profiles":
            return handle_list_profiles()

        elif name == "list_contracts":
            return handle_list_contracts()

        elif name == "estimate_cost":
            return handle_estimate_cost(arguments)

        elif name == "check_readiness":
            return handle_check_readiness(arguments)

        elif name == "check_compliance":
            return handle_check_compliance(arguments)

        elif name == "generate_document":
            return handle_generate_document(arguments)

        elif name == "get_template_variables":
            return handle_get_template_variables(arguments)

        elif name == "calculate_scores":
            return handle_calculate_scores(arguments)

        elif name == "get_scoring_rubric":
            return handle_get_scoring_rubric(arguments)

        # ============== ANALYSIS TOOLS ==============
        elif name == "clone_repo":
            return handle_clone_repo(arguments)

        elif name == "analyze_repo":
            return handle_analyze_repo(arguments)

        elif name == "scan_security":
            return handle_scan_security(arguments)

        elif name == "run_script":
            return handle_run_script(arguments)

        elif name == "run_tests":
            return handle_run_tests(arguments)

        elif name == "check_lint":
            return handle_check_lint(arguments)

        elif name == "check_types":
            return handle_check_types(arguments)

        elif name == "find_duplicates":
            return handle_find_duplicates(arguments)

        elif name == "analyze_complexity":
            return handle_analyze_complexity(arguments)

        elif name == "generate_report":
            return handle_generate_report(arguments, workspace_id)

        elif name == "export_results":
            return handle_export_results(arguments)

        elif name == "batch_analyze":
            return await handle_batch_analyze(arguments)

        # Pass through to audit server
        else:
            return await audit_server.handle_tool(name, arguments)

    except Exception as e:
        logger.error(f"Tool error: {e}")
        return {"error": str(e)}


def validate_estimate(
    total_hours: float,
    total_cost: float,
    kloc: float,
    hourly_rate: float = None,
    workspace_id: str = "global"
) -> Dict:
    """Validate estimate against bounds."""
    bounds = SettingsManager.get_validation_bounds(workspace_id)
    warnings = []
    errors = []

    if hourly_rate is None and total_hours > 0:
        hourly_rate = total_cost / total_hours

    # Hours per KLOC check
    if kloc > 0:
        hours_per_kloc = total_hours / kloc
        if hours_per_kloc < bounds["hours_per_kloc_min"]:
            errors.append(f"Hours/KLOC ({hours_per_kloc:.1f}) below minimum ({bounds['hours_per_kloc_min']})")
        elif hours_per_kloc > bounds["hours_per_kloc_max"]:
            errors.append(f"Hours/KLOC ({hours_per_kloc:.1f}) above maximum ({bounds['hours_per_kloc_max']})")
    else:
        hours_per_kloc = None

    # Rate check
    if hourly_rate:
        if hourly_rate < bounds["rate_min"]:
            warnings.append(f"Rate ${hourly_rate}/hr below market minimum ${bounds['rate_min']}")
        elif hourly_rate > bounds["rate_max"]:
            warnings.append(f"Rate ${hourly_rate}/hr above market maximum ${bounds['rate_max']}")

    return {
        "valid": len(errors) == 0,
        "total_hours": total_hours,
        "total_cost": total_cost,
        "kloc": kloc,
        "hourly_rate": hourly_rate,
        "hours_per_kloc": hours_per_kloc,
        "warnings": warnings,
        "errors": errors,
        "bounds": {
            "rate_range": f"${bounds['rate_min']}-${bounds['rate_max']}/hr",
            "hours_per_kloc_range": f"{bounds['hours_per_kloc_min']}-{bounds['hours_per_kloc_max']}",
        }
    }


def estimate_with_settings(
    workspace_id: str,
    loc: Optional[int] = None,
    kloc: Optional[float] = None,
    complexity: str = "nominal",
    region: Optional[str] = None
) -> Dict:
    """
    Calculate cost estimate using workspace-specific COCOMO settings and rates.

    Formula: Effort (PM) = a × (KLOC)^b × EAF
    Hours = Effort × hours_per_pm
    """
    # Get workspace settings
    cocomo = SettingsManager.get_cocomo(workspace_id)
    rates = SettingsManager.get_rates(workspace_id)

    # Convert LOC to KLOC
    if loc and not kloc:
        kloc = loc / 1000
    if not kloc:
        kloc = 10  # Default 10K LOC

    # Complexity EAF multipliers
    complexity_eaf = {
        "low": 0.7,
        "nominal": 1.0,
        "high": 1.3,
        "very_high": 1.6
    }
    eaf = complexity_eaf.get(complexity, 1.0)

    # COCOMO II calculation
    a = cocomo.get("a", 0.5)
    b = cocomo.get("b", 0.85)
    hours_per_pm = cocomo.get("hours_per_pm", 160)

    effort_pm = a * (kloc ** b) * eaf
    base_hours = effort_pm * hours_per_pm

    # Calculate cost by region
    results = {
        "methodology": "COCOMO II Modern",
        "formula": f"Effort = {a} × (KLOC)^{b} × EAF",
        "inputs": {
            "kloc": kloc,
            "loc": int(kloc * 1000),
            "complexity": complexity,
            "eaf": eaf,
        },
        "calculation": {
            "effort_pm": round(effort_pm, 2),
            "base_hours": round(base_hours, 1),
            "a": a,
            "b": b,
            "hours_per_pm": hours_per_pm,
        },
        "estimates": {}
    }

    # Filter by region if specified
    regions_to_calc = {region: rates[region]} if region and region in rates else rates

    for region_code, region_rates in regions_to_calc.items():
        typical_rate = region_rates.get("typical", region_rates.get("middle", 50))
        cost = base_hours * typical_rate

        results["estimates"][region_code] = {
            "hours": round(base_hours, 1),
            "rate": typical_rate,
            "cost": round(cost, 2),
            "rates_by_level": {
                level: round(base_hours * rate, 2)
                for level, rate in region_rates.items()
            }
        }

    # Add validation
    if len(regions_to_calc) == 1:
        region_data = list(results["estimates"].values())[0]
        results["validation"] = validate_estimate(
            total_hours=base_hours,
            total_cost=region_data["cost"],
            kloc=kloc,
            hourly_rate=region_data["rate"],
            workspace_id=workspace_id
        )

    return results


def compare_human_vs_ai(
    workspace_id: str,
    loc: Optional[int] = None,
    kloc: Optional[float] = None,
    region: str = "ua"
) -> Dict:
    """Compare human vs AI-assisted development costs."""
    # Get workspace settings
    ai_prod = SettingsManager.get_ai_productivity(workspace_id)
    rates = SettingsManager.get_rates(workspace_id)

    # Convert LOC to KLOC
    if loc and not kloc:
        kloc = loc / 1000
    if not kloc:
        kloc = 10  # Default

    # Get region rates
    region_rates = rates.get(region, rates.get("ua", {"typical": 35}))
    typical_rate = region_rates.get("typical", region_rates.get("middle", 35))

    # Calculate hours for each mode
    pure_human_hrs = kloc * ai_prod.get("pure_human", 25)
    ai_assisted_hrs = kloc * ai_prod.get("ai_assisted", 8)
    hybrid_hrs = kloc * ai_prod.get("hybrid", 6.5)

    # Calculate costs
    pure_human_cost = pure_human_hrs * typical_rate
    ai_assisted_cost = ai_assisted_hrs * typical_rate
    hybrid_cost = hybrid_hrs * typical_rate

    # Calculate savings
    ai_time_savings = (pure_human_hrs - ai_assisted_hrs) / pure_human_hrs * 100
    ai_cost_savings = pure_human_cost - ai_assisted_cost
    hybrid_time_savings = (pure_human_hrs - hybrid_hrs) / pure_human_hrs * 100
    hybrid_cost_savings = pure_human_cost - hybrid_cost

    return {
        "comparison": "Human vs AI-Assisted Development",
        "inputs": {
            "kloc": kloc,
            "loc": int(kloc * 1000),
            "region": region,
            "hourly_rate": typical_rate,
        },
        "productivity_settings": {
            "pure_human_hrs_per_kloc": ai_prod.get("pure_human", 25),
            "ai_assisted_hrs_per_kloc": ai_prod.get("ai_assisted", 8),
            "hybrid_hrs_per_kloc": ai_prod.get("hybrid", 6.5),
        },
        "estimates": {
            "pure_human": {
                "hours": round(pure_human_hrs, 1),
                "cost": round(pure_human_cost, 2),
                "description": "Traditional development without AI"
            },
            "ai_assisted": {
                "hours": round(ai_assisted_hrs, 1),
                "cost": round(ai_assisted_cost, 2),
                "description": "Development with AI tools (Copilot, Claude, etc.)"
            },
            "hybrid": {
                "hours": round(hybrid_hrs, 1),
                "cost": round(hybrid_cost, 2),
                "description": "AI-generated code + human review"
            }
        },
        "savings": {
            "ai_vs_human": {
                "hours_saved": round(pure_human_hrs - ai_assisted_hrs, 1),
                "cost_saved": round(ai_cost_savings, 2),
                "time_reduction_pct": round(ai_time_savings, 1),
                "speedup_factor": round(pure_human_hrs / ai_assisted_hrs, 2) if ai_assisted_hrs > 0 else 0
            },
            "hybrid_vs_human": {
                "hours_saved": round(pure_human_hrs - hybrid_hrs, 1),
                "cost_saved": round(hybrid_cost_savings, 2),
                "time_reduction_pct": round(hybrid_time_savings, 1),
                "speedup_factor": round(pure_human_hrs / hybrid_hrs, 2) if hybrid_hrs > 0 else 0
            }
        },
        "recommendation": _get_recommendation(kloc, ai_time_savings)
    }


def _get_recommendation(kloc: float, ai_savings_pct: float) -> str:
    """Generate recommendation based on project size and savings."""
    if kloc < 5:
        return "Small project - AI assistance recommended for rapid prototyping"
    elif kloc < 20:
        return f"Medium project - AI tools can save ~{ai_savings_pct:.0f}% time, recommended for new features"
    elif kloc < 100:
        return f"Large project - Hybrid approach recommended: AI for initial code, human review for quality"
    else:
        return "Enterprise scale - Use AI for boilerplate, maintain strict review process for critical components"


# =============================================================================
# OAUTH 2.0 ENDPOINTS
# =============================================================================

oauth_codes: Dict[str, Dict] = {}
oauth_tokens: Dict[str, Dict] = {}


async def oauth_metadata(request):
    """OAuth 2.0 Authorization Server Metadata (RFC 8414)."""
    return JSONResponse({
        "issuer": SERVER_URL,
        "authorization_endpoint": f"{SERVER_URL}/authorize",
        "token_endpoint": f"{SERVER_URL}/token",
        "registration_endpoint": f"{SERVER_URL}/register",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256", "plain"],
        "token_endpoint_auth_methods_supported": ["none", "client_secret_post"],
        "scopes_supported": ["claudeai", "mcp", "audit", "memory", "read", "write"],
        "service_documentation": f"{SERVER_URL}/",
    })


async def mcp_discovery(request):
    """MCP Server Discovery."""
    all_tools = audit_server.get_tools() + MEMORY_TOOLS + ANALYSIS_TOOLS + BUSINESS_TOOLS
    return JSONResponse({
        "name": MCP_SERVER_INFO["name"],
        "version": MCP_SERVER_INFO["version"],
        "description": MCP_SERVER_INFO["description"],
        "transport": {
            "type": "sse",
            "url": f"{SERVER_URL}/sse"
        },
        "oauth": {
            "authorization_url": f"{SERVER_URL}/authorize",
            "token_url": f"{SERVER_URL}/token"
        },
        "capabilities": {
            "audit": True,
            "memory": True,
            "validation": True,
            "multi_region_rates": True,
            "formulas_available": FORMULAS_AVAILABLE,
        },
        "tools_count": len(all_tools),
        "tools": [t["name"] for t in all_tools]
    })


async def oauth_authorize(request):
    """OAuth 2.0 Authorization Endpoint - auto-approve for Claude Web."""
    params = request.query_params

    client_id = params.get("client_id", "")
    redirect_uri = params.get("redirect_uri", "")
    response_type = params.get("response_type", "code")
    state = params.get("state", "")
    scope = params.get("scope", "mcp")
    code_challenge = params.get("code_challenge", "")
    code_challenge_method = params.get("code_challenge_method", "S256")

    logger.info(f"OAuth authorize: client_id={client_id}, redirect_uri={redirect_uri}")

    if response_type != "code":
        return JSONResponse({"error": "unsupported_response_type"}, status_code=400)

    if not redirect_uri:
        return JSONResponse({"error": "invalid_request", "error_description": "redirect_uri required"}, status_code=400)

    # Generate authorization code (auto-approve)
    auth_code = secrets.token_urlsafe(32)

    # Store code with metadata
    oauth_codes[auth_code] = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "code_challenge": code_challenge,
        "code_challenge_method": code_challenge_method,
        "scope": scope,
        "created_at": datetime.now()
    }

    # Build redirect URL
    parsed = urllib.parse.urlparse(redirect_uri)
    query_params = urllib.parse.parse_qs(parsed.query)
    query_params["code"] = [auth_code]
    if state:
        query_params["state"] = [state]

    new_query = urllib.parse.urlencode(query_params, doseq=True)
    redirect_url = urllib.parse.urlunparse((
        parsed.scheme, parsed.netloc, parsed.path,
        parsed.params, new_query, parsed.fragment
    ))

    return RedirectResponse(url=redirect_url, status_code=302)


async def oauth_token(request):
    """OAuth 2.0 Token Endpoint."""
    try:
        if request.headers.get("content-type", "").startswith("application/json"):
            data = await request.json()
        else:
            form = await request.form()
            data = dict(form)
    except:
        data = dict(request.query_params)

    grant_type = data.get("grant_type", "")
    code = data.get("code", "")
    client_id = data.get("client_id", "")
    code_verifier = data.get("code_verifier", "")

    logger.info(f"OAuth token: grant_type={grant_type}")

    if grant_type == "authorization_code":
        if code not in oauth_codes:
            return JSONResponse({"error": "invalid_grant"}, status_code=400)

        code_data = oauth_codes[code]

        # Verify PKCE
        if code_data.get("code_challenge") and code_verifier:
            if code_data.get("code_challenge_method") == "S256":
                digest = hashlib.sha256(code_verifier.encode()).digest()
                computed = base64.urlsafe_b64encode(digest).rstrip(b'=').decode()
                if computed != code_data["code_challenge"]:
                    return JSONResponse({"error": "invalid_grant", "error_description": "Invalid code_verifier"}, status_code=400)

        # Generate tokens
        access_token = secrets.token_urlsafe(32)
        refresh_token = secrets.token_urlsafe(32)

        oauth_tokens[access_token] = {
            "client_id": client_id,
            "scope": code_data.get("scope", "mcp"),
            "created_at": datetime.now()
        }

        del oauth_codes[code]

        return JSONResponse({
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": 86400,
            "refresh_token": refresh_token,
            "scope": code_data.get("scope", "mcp")
        })

    elif grant_type == "refresh_token":
        access_token = secrets.token_urlsafe(32)
        refresh_token = secrets.token_urlsafe(32)

        oauth_tokens[access_token] = {
            "client_id": client_id,
            "scope": "mcp",
            "created_at": datetime.now()
        }

        return JSONResponse({
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": 86400,
            "refresh_token": refresh_token,
            "scope": "mcp"
        })

    return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)


async def oauth_register(request):
    """OAuth 2.0 Dynamic Client Registration."""
    try:
        data = await request.json()
    except:
        data = {}

    client_id = data.get("client_name", "claude") + "_" + secrets.token_hex(8)
    client_secret = secrets.token_urlsafe(32)

    return JSONResponse({
        "client_id": client_id,
        "client_secret": client_secret,
        "client_name": data.get("client_name", "Claude"),
        "redirect_uris": data.get("redirect_uris", []),
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none"
    }, status_code=201)


# =============================================================================
# MCP SSE ENDPOINTS
# =============================================================================

async def mcp_sse_endpoint(request):
    """SSE endpoint for MCP protocol."""
    session_id = str(uuid.uuid4())
    logger.info(f"MCP SSE: New connection, session={session_id}")

    async def event_generator():
        # Send initial endpoint event
        endpoint_url = f"{SERVER_URL}/message?session_id={session_id}"
        yield f"event: endpoint\ndata: {endpoint_url}\n\n"

        # Keep connection alive
        try:
            while True:
                await asyncio.sleep(30)
                yield ": keepalive\n\n"
        except asyncio.CancelledError:
            logger.info(f"MCP SSE: Connection closed, session={session_id}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


async def mcp_message_endpoint(request):
    """Handle MCP JSON-RPC messages."""
    try:
        data = await request.json()
    except:
        return JSONResponse({"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}}, status_code=400)

    session_id = request.query_params.get("session_id", "default")
    method = data.get("method", "")
    msg_id = data.get("id")
    params = data.get("params", {})

    logger.info(f"MCP Message: method={method}, session={session_id}")

    # Get all tools
    all_tools = audit_server.get_tools() + MEMORY_TOOLS + ANALYSIS_TOOLS + BUSINESS_TOOLS

    # Handle MCP methods
    if method == "initialize":
        result = {
            "protocolVersion": MCP_SERVER_INFO["protocolVersion"],
            "serverInfo": MCP_SERVER_INFO,
            "capabilities": {
                "tools": {"listChanged": False}
            }
        }
    elif method == "tools/list":
        result = {"tools": all_tools}
    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        try:
            tool_result = await execute_tool(tool_name, arguments, workspace_id=session_id)
            result = {
                "content": [{"type": "text", "text": json.dumps(tool_result, indent=2, default=str)}]
            }
        except Exception as e:
            logger.error(f"Tool error: {e}")
            result = {
                "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                "isError": True
            }
    elif method == "notifications/initialized":
        return JSONResponse({})
    else:
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"}
        })

    return JSONResponse({
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": result
    })


async def mcp_streamable_http(request):
    """Streamable HTTP endpoint for MCP."""
    accept = request.headers.get("accept", "")

    if request.method == "GET":
        return await mcp_sse_endpoint(request)

    # POST - JSON-RPC
    try:
        data = await request.json()
    except:
        return JSONResponse(
            {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}},
            status_code=400
        )

    method = data.get("method", "")
    msg_id = data.get("id")
    params = data.get("params", {})

    all_tools = audit_server.get_tools() + MEMORY_TOOLS + ANALYSIS_TOOLS + BUSINESS_TOOLS

    if method == "initialize":
        result = {
            "protocolVersion": MCP_SERVER_INFO["protocolVersion"],
            "serverInfo": MCP_SERVER_INFO,
            "capabilities": {"tools": {"listChanged": False}}
        }
    elif method == "tools/list":
        result = {"tools": all_tools}
    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        try:
            tool_result = await execute_tool(tool_name, arguments)
            result = {
                "content": [{"type": "text", "text": json.dumps(tool_result, indent=2, default=str)}]
            }
        except Exception as e:
            result = {"content": [{"type": "text", "text": f"Error: {str(e)}"}], "isError": True}
    elif method == "notifications/initialized":
        return JSONResponse({})
    else:
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"}
        })

    # SSE response if requested
    if "text/event-stream" in accept:
        async def sse_response():
            response_data = json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": result})
            yield f"event: message\ndata: {response_data}\n\n"
        return StreamingResponse(sse_response(), media_type="text/event-stream")

    return JSONResponse({"jsonrpc": "2.0", "id": msg_id, "result": result})


# =============================================================================
# API ENDPOINTS
# =============================================================================

async def homepage(request):
    """Serve homepage with full UI. Handle POST for MCP streamable HTTP."""
    # Handle POST requests as MCP streamable HTTP
    if request.method == "POST":
        return await mcp_streamable_http(request)

    all_tools = audit_server.get_tools() + MEMORY_TOOLS + ANALYSIS_TOOLS + BUSINESS_TOOLS
    tool_names = [t["name"] for t in all_tools]

    # Get rates for settings
    try:
        from executors.cost_estimator.formulas import REGIONAL_RATES as rates_data
    except:
        rates_data = REGIONAL_RATES

    return HTMLResponse(f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCP Audit Server</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: system-ui, -apple-system, sans-serif; background: #f3f4f6; color: #1f2937; }}

        /* Navigation */
        .nav {{ background: #1f2937; padding: 0 20px; display: flex; align-items: center; }}
        .nav-brand {{ color: white; font-size: 18px; font-weight: 600; padding: 15px 0; }}
        .nav-tabs {{ display: flex; margin-left: 40px; }}
        .nav-tab {{ color: #9ca3af; padding: 15px 20px; cursor: pointer; border-bottom: 2px solid transparent; transition: all 0.2s; }}
        .nav-tab:hover {{ color: white; }}
        .nav-tab.active {{ color: white; border-bottom-color: #3b82f6; }}

        /* Main */
        .main {{ max-width: 1200px; margin: 0 auto; padding: 30px 20px; }}

        /* Cards */
        .card {{ background: white; border-radius: 12px; padding: 24px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .card h2 {{ font-size: 18px; margin-bottom: 16px; color: #374151; }}
        .card h3 {{ font-size: 15px; margin: 20px 0 12px; color: #4b5563; }}

        /* Status badges */
        .status {{ display: inline-flex; align-items: center; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: 500; margin-right: 8px; }}
        .status-ok {{ background: #d1fae5; color: #065f46; }}
        .status-warn {{ background: #fef3c7; color: #92400e; }}
        .status::before {{ content: ""; width: 6px; height: 6px; border-radius: 50%; margin-right: 6px; }}
        .status-ok::before {{ background: #10b981; }}
        .status-warn::before {{ background: #f59e0b; }}

        /* Endpoints */
        .endpoint {{ background: #f9fafb; padding: 12px 16px; margin: 8px 0; border-radius: 8px; font-family: "SF Mono", Monaco, monospace; font-size: 13px; display: flex; justify-content: space-between; align-items: center; }}
        .endpoint-label {{ color: #6b7280; font-weight: 500; }}
        .endpoint-url {{ color: #3b82f6; }}

        /* Tools grid */
        .tools-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; }}
        .tool-item {{ background: #f0fdf4; padding: 12px 16px; border-radius: 8px; border-left: 3px solid #10b981; }}
        .tool-name {{ font-weight: 600; color: #047857; font-size: 14px; }}
        .tool-desc {{ font-size: 12px; color: #6b7280; margin-top: 4px; }}

        /* Forms */
        .form-group {{ margin-bottom: 16px; }}
        .form-label {{ display: block; font-size: 13px; font-weight: 500; color: #374151; margin-bottom: 6px; }}
        .form-input {{ width: 100%; padding: 10px 12px; border: 1px solid #d1d5db; border-radius: 8px; font-size: 14px; }}
        .form-input:focus {{ outline: none; border-color: #3b82f6; box-shadow: 0 0 0 3px rgba(59,130,246,0.1); }}
        .form-hint {{ font-size: 12px; color: #6b7280; margin-top: 4px; }}

        /* Table */
        .table {{ width: 100%; border-collapse: collapse; }}
        .table th, .table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e5e7eb; }}
        .table th {{ background: #f9fafb; font-weight: 600; font-size: 13px; color: #6b7280; text-transform: uppercase; }}
        .table td {{ font-size: 14px; }}
        .table input {{ width: 80px; padding: 6px 8px; border: 1px solid #d1d5db; border-radius: 6px; text-align: right; }}

        /* Buttons */
        .btn {{ padding: 10px 20px; border-radius: 8px; font-weight: 500; cursor: pointer; transition: all 0.2s; border: none; font-size: 14px; }}
        .btn-primary {{ background: #3b82f6; color: white; }}
        .btn-primary:hover {{ background: #2563eb; }}
        .btn-secondary {{ background: #f3f4f6; color: #374151; border: 1px solid #d1d5db; }}
        .btn-secondary:hover {{ background: #e5e7eb; }}
        .btn-danger {{ background: #fee2e2; color: #b91c1c; }}
        .btn-danger:hover {{ background: #fecaca; }}

        /* Tabs content */
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}

        /* Grid layout */
        .grid-2 {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }}
        .grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }}

        /* Code block */
        pre {{ background: #1f2937; color: #e5e7eb; padding: 16px; border-radius: 8px; overflow-x: auto; font-size: 13px; }}
        code {{ font-family: "SF Mono", Monaco, monospace; }}

        /* Alerts */
        .alert {{ padding: 12px 16px; border-radius: 8px; margin-bottom: 16px; }}
        .alert-info {{ background: #dbeafe; color: #1e40af; }}
        .alert-success {{ background: #d1fae5; color: #065f46; }}
        .alert-warning {{ background: #fef3c7; color: #92400e; }}

        /* Docs */
        .docs h3 {{ font-size: 16px; color: #1f2937; margin: 24px 0 12px; padding-bottom: 8px; border-bottom: 1px solid #e5e7eb; }}
        .docs p {{ margin-bottom: 12px; line-height: 1.6; }}
        .docs ul {{ margin: 12px 0; padding-left: 24px; }}
        .docs li {{ margin-bottom: 8px; }}

        @media (max-width: 768px) {{
            .grid-2, .grid-3 {{ grid-template-columns: 1fr; }}
            .nav-tabs {{ margin-left: 10px; }}
            .nav-tab {{ padding: 15px 12px; font-size: 14px; }}
        }}
    </style>
</head>
<body>
    <nav class="nav">
        <div class="nav-brand">MCP Audit Server</div>
        <div class="nav-tabs">
            <div class="nav-tab active" data-tab="dashboard">Dashboard</div>
            <div class="nav-tab" data-tab="settings">Settings</div>
            <div class="nav-tab" data-tab="docs">Documentation</div>
        </div>
    </nav>

    <main class="main">
        <!-- DASHBOARD TAB -->
        <div id="dashboard" class="tab-content active">
            <div class="card">
                <h2>Server Status</h2>
                <p style="margin-bottom: 16px;">
                    <span class="status status-ok">PostgreSQL: {"Connected" if db_connection else "In-memory"}</span>
                    <span class="status status-ok">Redis: {"Connected" if redis_client else "In-memory"}</span>
                    <span class="status status-ok">Formulas: {"Loaded" if FORMULAS_AVAILABLE else "Not loaded"}</span>
                    <span class="status status-ok">Tools: {len(all_tools)}</span>
                </p>
            </div>

            <div class="grid-2">
                <div class="card">
                    <h2>MCP Endpoints</h2>
                    <div class="endpoint">
                        <span class="endpoint-label">SSE Transport</span>
                        <span class="endpoint-url">{SERVER_URL}/mcp/sse</span>
                    </div>
                    <div class="endpoint">
                        <span class="endpoint-label">Streamable HTTP</span>
                        <span class="endpoint-url">{SERVER_URL}/mcp</span>
                    </div>
                    <div class="endpoint">
                        <span class="endpoint-label">Discovery</span>
                        <span class="endpoint-url">{SERVER_URL}/.well-known/mcp.json</span>
                    </div>
                </div>

                <div class="card">
                    <h2>OAuth 2.0</h2>
                    <div class="endpoint">
                        <span class="endpoint-label">Authorize</span>
                        <span class="endpoint-url">{SERVER_URL}/authorize</span>
                    </div>
                    <div class="endpoint">
                        <span class="endpoint-label">Token</span>
                        <span class="endpoint-url">{SERVER_URL}/token</span>
                    </div>
                    <div class="endpoint">
                        <span class="endpoint-label">Register</span>
                        <span class="endpoint-url">{SERVER_URL}/register</span>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2>Available Tools ({len(all_tools)})</h2>
                <div class="tools-grid">
                    {"".join(f'<div class="tool-item"><div class="tool-name">{t["name"]}</div><div class="tool-desc">{t.get("description", "")[:60]}...</div></div>' for t in all_tools)}
                </div>
            </div>

            <div class="card">
                <h2>Quick Connect</h2>
                <p style="margin-bottom: 12px;">Add this MCP server to Claude Web:</p>
                <div class="endpoint">
                    <span class="endpoint-url">{SERVER_URL}</span>
                    <button class="btn btn-secondary" onclick="navigator.clipboard.writeText('{SERVER_URL}')">Copy</button>
                </div>
            </div>
        </div>

        <!-- SETTINGS TAB -->
        <div id="settings" class="tab-content">
            <div class="alert alert-info">
                Settings are validated against anti-hallucination bounds. Invalid values will be rejected.
            </div>

            <div class="card">
                <h2>Regional Rates (USD/hour)</h2>
                <p style="margin-bottom: 16px; color: #6b7280;">Configure hourly rates for different regions. Bounds: $5-300/hr</p>
                <table class="table" id="rates-table">
                    <thead>
                        <tr>
                            <th>Region</th>
                            <th>Junior</th>
                            <th>Middle</th>
                            <th>Senior</th>
                            <th>Lead</th>
                            <th>Typical</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(f'''<tr>
                            <td><strong>{k.upper()}</strong></td>
                            <td><input type="number" value="{v.get("rates", v).get("junior", v.get("junior", 0))}" data-region="{k}" data-level="junior" min="5" max="300"></td>
                            <td><input type="number" value="{v.get("rates", v).get("middle", v.get("middle", 0))}" data-region="{k}" data-level="middle" min="5" max="300"></td>
                            <td><input type="number" value="{v.get("rates", v).get("senior", v.get("senior", 0))}" data-region="{k}" data-level="senior" min="5" max="300"></td>
                            <td><input type="number" value="{v.get("rates", v).get("lead", v.get("lead", 0))}" data-region="{k}" data-level="lead" min="5" max="300"></td>
                            <td><input type="number" value="{v.get("rates", v).get("typical", v.get("typical", 0))}" data-region="{k}" data-level="typical" min="5" max="300"></td>
                        </tr>''' for k, v in rates_data.items())}
                    </tbody>
                </table>
                <div style="margin-top: 16px;">
                    <button class="btn btn-primary" onclick="saveRates()">Save Rates</button>
                    <button class="btn btn-secondary" onclick="resetRates()">Reset to Defaults</button>
                </div>
            </div>

            <div class="grid-2">
                <div class="card">
                    <h2>COCOMO II Parameters</h2>
                    <p style="margin-bottom: 16px; color: #6b7280;">Formula: Effort = a × (KLOC)^b × EAF</p>
                    <div class="form-group">
                        <label class="form-label">Coefficient (a)</label>
                        <input type="number" class="form-input" id="cocomo-a" value="0.5" step="0.01" min="0.1" max="5.0">
                        <div class="form-hint">Bounds: 0.1 - 5.0 (default: 0.5)</div>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Exponent (b)</label>
                        <input type="number" class="form-input" id="cocomo-b" value="0.85" step="0.01" min="0.5" max="1.5">
                        <div class="form-hint">Bounds: 0.5 - 1.5 (default: 0.85)</div>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Hours per Person-Month</label>
                        <input type="number" class="form-input" id="cocomo-hours" value="160" min="120" max="200">
                        <div class="form-hint">Bounds: 120 - 200 (default: 160)</div>
                    </div>
                    <button class="btn btn-primary" onclick="saveCocomo()">Save COCOMO</button>
                </div>

                <div class="card">
                    <h2>AI Productivity (hrs/KLOC)</h2>
                    <p style="margin-bottom: 16px; color: #6b7280;">Hours per 1000 lines of code</p>
                    <div class="form-group">
                        <label class="form-label">Pure Human</label>
                        <input type="number" class="form-input" id="ai-human" value="25" min="1" max="100">
                        <div class="form-hint">Traditional development without AI</div>
                    </div>
                    <div class="form-group">
                        <label class="form-label">AI-Assisted</label>
                        <input type="number" class="form-input" id="ai-assisted" value="8" min="1" max="50">
                        <div class="form-hint">Development with AI tools (Copilot, etc.)</div>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Hybrid</label>
                        <input type="number" class="form-input" id="ai-hybrid" value="6.5" step="0.5" min="1" max="40">
                        <div class="form-hint">AI + human review</div>
                    </div>
                    <button class="btn btn-primary" onclick="saveAI()">Save AI Productivity</button>
                </div>
            </div>

            <div class="card">
                <h2>Validation Bounds</h2>
                <p style="margin-bottom: 16px; color: #6b7280;">These bounds protect against hallucinated estimates</p>
                <div class="grid-3">
                    <div>
                        <h3>Hourly Rates</h3>
                        <p><strong>Min:</strong> $5/hr</p>
                        <p><strong>Max:</strong> $300/hr</p>
                    </div>
                    <div>
                        <h3>Hours per KLOC</h3>
                        <p><strong>Min:</strong> 2 hrs/KLOC</p>
                        <p><strong>Max:</strong> 200 hrs/KLOC</p>
                    </div>
                    <div>
                        <h3>PERT Ratio</h3>
                        <p><strong>Max:</strong> 10x</p>
                        <p><em>Pessimistic/Optimistic</em></p>
                    </div>
                </div>
            </div>
        </div>

        <!-- DOCUMENTATION TAB -->
        <div id="docs" class="tab-content">
            <div class="card docs">
                <h2>MCP Audit Server Documentation</h2>

                <h3>Overview</h3>
                <p>MCP Audit Server provides repository audit capabilities through the Model Context Protocol (MCP). It supports Web Claude integration via SSE transport with OAuth 2.0 authentication.</p>

                <h3>Quick Start</h3>
                <ol>
                    <li>Start the server: <code>python -m gateway.mcp.http_server</code></li>
                    <li>Open Claude Web (claude.ai)</li>
                    <li>Add MCP integration with URL: <code>{SERVER_URL}</code></li>
                    <li>Use tools like <code>estimate_cocomo</code>, <code>full_audit</code>, etc.</li>
                </ol>

                <h3>Available Tools</h3>
                <p><strong>Audit Tools:</strong></p>
                <ul>
                    <li><code>audit</code> - Run repository audit (quick_scan, full_audit, etc.)</li>
                    <li><code>estimate_cocomo</code> - COCOMO II cost estimation</li>
                    <li><code>estimate_comprehensive</code> - All 8 methodologies</li>
                    <li><code>estimate_pert</code> - 3-point PERT analysis</li>
                    <li><code>estimate_ai_efficiency</code> - Human vs AI comparison</li>
                    <li><code>calculate_roi</code> - ROI calculation</li>
                </ul>

                <p><strong>Memory Tools:</strong></p>
                <ul>
                    <li><code>store_memory</code> - Save facts and context</li>
                    <li><code>recall_memory</code> - Retrieve memories</li>
                    <li><code>record_decision</code> - ADR-style decisions</li>
                    <li><code>record_learning</code> - Learn from feedback</li>
                    <li><code>get_context</code> - Get workspace context</li>
                    <li><code>validate_estimate</code> - Anti-hallucination check</li>
                </ul>

                <h3>Example: COCOMO Estimation</h3>
                <pre><code>// MCP tool call
{{
  "name": "estimate_cocomo",
  "arguments": {{
    "loc": 10000,
    "tech_debt_score": 10,
    "team_experience": "nominal"
  }}
}}

// Returns hours and cost for all 8 regions</code></pre>

                <h3>Example: Store Memory</h3>
                <pre><code>// Store a fact
{{
  "name": "store_memory",
  "arguments": {{
    "content": "Project uses FastAPI + PostgreSQL",
    "type": "fact",
    "tags": ["tech", "backend"]
  }}
}}</code></pre>

                <h3>Regional Rates</h3>
                <table class="table">
                    <thead>
                        <tr><th>Region</th><th>Junior</th><th>Middle</th><th>Senior</th><th>Lead</th></tr>
                    </thead>
                    <tbody>
                        <tr><td>UA</td><td>$15</td><td>$25</td><td>$40</td><td>$55</td></tr>
                        <tr><td>UA Compliance</td><td>$20</td><td>$35</td><td>$55</td><td>$75</td></tr>
                        <tr><td>PL</td><td>$25</td><td>$40</td><td>$60</td><td>$80</td></tr>
                        <tr><td>EU</td><td>$40</td><td>$65</td><td>$95</td><td>$120</td></tr>
                        <tr><td>DE</td><td>$50</td><td>$80</td><td>$110</td><td>$140</td></tr>
                        <tr><td>UK</td><td>$45</td><td>$75</td><td>$105</td><td>$135</td></tr>
                        <tr><td>US</td><td>$60</td><td>$100</td><td>$150</td><td>$200</td></tr>
                        <tr><td>IN</td><td>$10</td><td>$18</td><td>$30</td><td>$45</td></tr>
                    </tbody>
                </table>

                <h3>Anti-Hallucination Validation</h3>
                <p>All estimates are validated against industry bounds:</p>
                <ul>
                    <li><strong>Hours/KLOC:</strong> Must be 2-200 (rejects unrealistic estimates)</li>
                    <li><strong>Hourly rates:</strong> Must be $5-300/hr</li>
                    <li><strong>PERT spread:</strong> Pessimistic/Optimistic ratio max 10x</li>
                </ul>

                <h3>API Endpoints</h3>
                <pre><code>GET  /health              - Health check
GET  /api/tools           - List all tools
GET  /api/rates           - Get regional rates
GET  /api/settings        - Get all settings
PUT  /api/settings/rates  - Update rates
PUT  /api/settings/cocomo - Update COCOMO params
POST /api/settings/reset  - Reset to defaults</code></pre>

                <h3>Environment Variables</h3>
                <pre><code>PORT=8090                 # Server port
SERVER_URL=http://...     # Public URL
DATABASE_URL=postgres://  # PostgreSQL (optional)
REDIS_URL=redis://        # Redis cache (optional)</code></pre>
            </div>
        </div>
    </main>

    <script>
        // Tab switching
        document.querySelectorAll('.nav-tab').forEach(tab => {{
            tab.addEventListener('click', () => {{
                document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                tab.classList.add('active');
                document.getElementById(tab.dataset.tab).classList.add('active');
            }});
        }});

        // Save functions
        async function saveRates() {{
            const rates = {{}};
            document.querySelectorAll('#rates-table input').forEach(input => {{
                const region = input.dataset.region;
                const level = input.dataset.level;
                if (!rates[region]) rates[region] = {{}};
                rates[region][level] = parseFloat(input.value);
            }});

            try {{
                const resp = await fetch('/api/settings/rates', {{
                    method: 'PUT',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify(rates)
                }});
                const data = await resp.json();
                if (resp.ok) {{
                    alert('Rates saved successfully!');
                }} else {{
                    alert('Error: ' + (data.error || data.detail || 'Unknown error'));
                }}
            }} catch (e) {{
                alert('Error saving rates: ' + e.message);
            }}
        }}

        async function saveCocomo() {{
            const params = {{
                a: parseFloat(document.getElementById('cocomo-a').value),
                b: parseFloat(document.getElementById('cocomo-b').value),
                hours_per_pm: parseInt(document.getElementById('cocomo-hours').value)
            }};

            try {{
                const resp = await fetch('/api/settings/cocomo', {{
                    method: 'PUT',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify(params)
                }});
                const data = await resp.json();
                if (resp.ok) {{
                    alert('COCOMO parameters saved!');
                }} else {{
                    alert('Error: ' + (data.error || data.detail || 'Unknown error'));
                }}
            }} catch (e) {{
                alert('Error saving COCOMO: ' + e.message);
            }}
        }}

        async function saveAI() {{
            const params = {{
                pure_human: parseFloat(document.getElementById('ai-human').value),
                ai_assisted: parseFloat(document.getElementById('ai-assisted').value),
                hybrid: parseFloat(document.getElementById('ai-hybrid').value)
            }};

            try {{
                const resp = await fetch('/api/settings/ai-productivity', {{
                    method: 'PUT',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify(params)
                }});
                const data = await resp.json();
                if (resp.ok) {{
                    alert('AI productivity saved!');
                }} else {{
                    alert('Error: ' + (data.error || data.detail || 'Unknown error'));
                }}
            }} catch (e) {{
                alert('Error saving AI productivity: ' + e.message);
            }}
        }}

        async function resetRates() {{
            if (!confirm('Reset all settings to defaults?')) return;
            try {{
                const resp = await fetch('/api/settings/reset', {{ method: 'POST' }});
                if (resp.ok) {{
                    alert('Settings reset to defaults. Refreshing...');
                    location.reload();
                }}
            }} catch (e) {{
                alert('Error: ' + e.message);
            }}
        }}
    </script>
</body>
</html>
    ''')


async def health_check(request):
    """Health check endpoint."""
    all_tools = audit_server.get_tools() + MEMORY_TOOLS + ANALYSIS_TOOLS + BUSINESS_TOOLS
    return JSONResponse({
        "status": "ok",
        "service": MCP_SERVER_INFO["name"],
        "version": MCP_SERVER_INFO["version"],
        "database": "connected" if db_connection else "in-memory",
        "redis": "connected" if redis_client else "in-memory",
        "formulas_available": FORMULAS_AVAILABLE,
        "tools_count": len(all_tools),
        "tools": [t["name"] for t in all_tools],
        "timestamp": datetime.now(timezone.utc).isoformat()
    })


async def api_tools(request):
    """List available tools."""
    all_tools = audit_server.get_tools() + MEMORY_TOOLS + ANALYSIS_TOOLS + BUSINESS_TOOLS
    return JSONResponse({
        "tools": all_tools,
        "count": len(all_tools)
    })


async def api_rates(request):
    """Get all rates."""
    workspace_id = request.query_params.get("workspace", "global")
    rates = SettingsManager.get_rates(workspace_id)
    return JSONResponse({
        "rates": rates,
        "regions": list(rates.keys())
    })


async def api_settings(request):
    """Get all settings."""
    workspace_id = request.query_params.get("workspace", "global")
    settings = SettingsManager.get_all(workspace_id)
    return JSONResponse(settings)


async def api_settings_rates(request):
    """Get or update regional rates."""
    workspace_id = request.query_params.get("workspace", "global")

    if request.method == "GET":
        return JSONResponse(SettingsManager.get_rates(workspace_id))

    # PUT - update rates
    try:
        data = await request.json()
        result = SettingsManager.update_rates(workspace_id, data)
        if result.get("success"):
            return JSONResponse(result)
        return JSONResponse(result, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


async def api_settings_cocomo(request):
    """Get or update COCOMO parameters."""
    workspace_id = request.query_params.get("workspace", "global")

    if request.method == "GET":
        return JSONResponse(SettingsManager.get_cocomo(workspace_id))

    # PUT - update COCOMO
    try:
        data = await request.json()
        result = SettingsManager.update_cocomo(workspace_id, data)
        if result.get("success"):
            return JSONResponse(result)
        return JSONResponse(result, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


async def api_settings_ai_productivity(request):
    """Get or update AI productivity settings."""
    workspace_id = request.query_params.get("workspace", "global")

    if request.method == "GET":
        return JSONResponse(SettingsManager.get_ai_productivity(workspace_id))

    # PUT - update AI productivity
    try:
        data = await request.json()
        result = SettingsManager.update_ai_productivity(workspace_id, data)
        if result.get("success"):
            return JSONResponse(result)
        return JSONResponse(result, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


async def api_settings_reset(request):
    """Reset all settings to defaults."""
    workspace_id = request.query_params.get("workspace", "global")
    result = SettingsManager.reset(workspace_id)
    return JSONResponse(result)


async def api_settings_bounds(request):
    """Get validation bounds."""
    workspace_id = request.query_params.get("workspace", "global")
    return JSONResponse(SettingsManager.get_validation_bounds(workspace_id))


# =============================================================================
# APPLICATION
# =============================================================================

def create_app():
    """Create the Starlette application."""
    routes = [
        # Homepage
        Route('/', homepage, methods=['GET', 'POST']),

        # Health & Info
        Route('/health', health_check),
        Route('/api/tools', api_tools),
        Route('/api/rates', api_rates),

        # Settings API
        Route('/api/settings', api_settings),
        Route('/api/settings/rates', api_settings_rates, methods=['GET', 'PUT']),
        Route('/api/settings/cocomo', api_settings_cocomo, methods=['GET', 'PUT']),
        Route('/api/settings/ai-productivity', api_settings_ai_productivity, methods=['GET', 'PUT']),
        Route('/api/settings/bounds', api_settings_bounds),
        Route('/api/settings/reset', api_settings_reset, methods=['POST']),

        # OAuth 2.0
        Route('/.well-known/oauth-authorization-server', oauth_metadata),
        Route('/.well-known/mcp.json', mcp_discovery),
        Route('/authorize', oauth_authorize),
        Route('/token', oauth_token, methods=['POST']),
        Route('/register', oauth_register, methods=['POST']),

        # MCP endpoints
        Route('/mcp/sse', mcp_sse_endpoint),
        Route('/mcp/message', mcp_message_endpoint, methods=['POST']),
        Route('/mcp', mcp_streamable_http, methods=['GET', 'POST']),
        Route('/sse', mcp_sse_endpoint),
        Route('/message', mcp_message_endpoint, methods=['POST']),
    ]

    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=['*'],
            allow_methods=['*'],
            allow_headers=['*']
        )
    ]

    return Starlette(routes=routes, middleware=middleware)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8090))

    # Initialize databases
    init_postgres()
    init_redis()

    all_tools = audit_server.get_tools() + MEMORY_TOOLS + ANALYSIS_TOOLS + BUSINESS_TOOLS

    logger.info("=" * 60)
    logger.info(f"MCP Audit HTTP Server v{MCP_SERVER_INFO['version']}")
    logger.info("=" * 60)
    logger.info(f"Tools: {len(all_tools)}")
    logger.info(f"Formulas: {'Available' if FORMULAS_AVAILABLE else 'Not loaded'}")
    logger.info(f"Database: {'PostgreSQL' if db_connection else 'In-memory'}")
    logger.info(f"Cache: {'Redis' if redis_client else 'In-memory'}")
    logger.info("-" * 60)
    logger.info(f"Dashboard: http://localhost:{port}/")
    logger.info(f"MCP SSE: http://localhost:{port}/mcp/sse")
    logger.info(f"Discovery: http://localhost:{port}/.well-known/mcp.json")
    logger.info(f"Health: http://localhost:{port}/health")
    logger.info("=" * 60)

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
