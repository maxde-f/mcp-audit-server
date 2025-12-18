"""
AuditSession - Session management for audit workspaces.

Tracks conversation context, active analyses, and provides
resume capability between Claude sessions.
"""
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message in a session."""
    id: str
    role: str  # user, assistant, tool
    content: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(**data)


@dataclass
class AuditSession:
    """
    A single audit session.

    Tracks:
    - Conversation messages
    - Active analysis
    - Tool calls
    - Context for LLM
    """
    session_id: str
    workspace_id: str
    created_at: str
    updated_at: str

    # Active state
    active_analysis_id: Optional[str] = None
    current_topic: Optional[str] = None

    # Messages
    messages: List[Message] = field(default_factory=list)

    # Context
    context: Dict[str, Any] = field(default_factory=dict)

    # Tool history
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["messages"] = [m.to_dict() if isinstance(m, Message) else m for m in self.messages]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditSession":
        messages = [
            Message.from_dict(m) if isinstance(m, dict) else m
            for m in data.get("messages", [])
        ]
        data["messages"] = messages

        # Filter only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    def add_message(
        self,
        role: str,
        content: str,
        **metadata,
    ) -> Message:
        """Add a message to the session."""
        msg = Message(
            id=str(uuid4())[:8],
            role=role,
            content=content,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=metadata,
        )
        self.messages.append(msg)
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return msg

    def add_tool_call(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> None:
        """Record a tool call."""
        self.tool_calls.append({
            "tool_name": tool_name,
            "inputs": inputs,
            "outputs": outputs,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def get_context_for_llm(self, max_messages: int = 20) -> List[Dict[str, str]]:
        """Get messages formatted for LLM API."""
        recent = self.messages[-max_messages:]
        return [
            {"role": m.role, "content": m.content}
            for m in recent
        ]

    def get_summary(self) -> str:
        """Get session summary for context."""
        parts = [f"Session: {self.session_id}"]

        if self.active_analysis_id:
            parts.append(f"Active analysis: {self.active_analysis_id}")

        if self.current_topic:
            parts.append(f"Current topic: {self.current_topic}")

        if self.tool_calls:
            recent_tools = [tc["tool_name"] for tc in self.tool_calls[-5:]]
            parts.append(f"Recent tools: {', '.join(recent_tools)}")

        return " | ".join(parts)


class SessionManager:
    """
    Manages sessions for a workspace.

    Provides:
    - Session creation/loading
    - Persistence to disk
    - Context building for LLM
    """

    def __init__(self, workspace_path: Path):
        self.workspace_path = Path(workspace_path)
        self.sessions_dir = self.workspace_path / ".audit" / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        self._current_session: Optional[AuditSession] = None

    def create_session(
        self,
        workspace_id: str,
        title: Optional[str] = None,
    ) -> AuditSession:
        """Create a new session."""
        now = datetime.now(timezone.utc).isoformat()

        session = AuditSession(
            session_id=str(uuid4())[:8],
            workspace_id=workspace_id,
            created_at=now,
            updated_at=now,
        )

        self._save_session(session)
        self._current_session = session

        logger.info(f"Created session: {session.session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[AuditSession]:
        """Load session by ID."""
        session_path = self.sessions_dir / f"{session_id}.json"

        if not session_path.exists():
            return None

        try:
            with open(session_path) as f:
                return AuditSession.from_dict(json.load(f))
        except (json.JSONDecodeError, KeyError):
            return None

    def get_current_session(self) -> Optional[AuditSession]:
        """Get current active session."""
        if self._current_session:
            return self._current_session

        # Try to load most recent
        sessions = self.list_sessions(limit=1)
        if sessions:
            self._current_session = sessions[0]
            return self._current_session

        return None

    def list_sessions(self, limit: int = 20) -> List[AuditSession]:
        """List recent sessions."""
        sessions = []

        if not self.sessions_dir.exists():
            return []

        for session_file in sorted(self.sessions_dir.glob("*.json"), reverse=True):
            try:
                with open(session_file) as f:
                    sessions.append(AuditSession.from_dict(json.load(f)))
            except (json.JSONDecodeError, KeyError):
                continue

            if len(sessions) >= limit:
                break

        return sessions

    def _save_session(self, session: AuditSession) -> None:
        """Save session to disk."""
        session_path = self.sessions_dir / f"{session.session_id}.json"
        with open(session_path, "w") as f:
            json.dump(session.to_dict(), f, indent=2)

    def save_current_session(self) -> None:
        """Save current session."""
        if self._current_session:
            self._save_session(self._current_session)

    def add_message(
        self,
        role: str,
        content: str,
        session_id: Optional[str] = None,
        **metadata,
    ) -> Optional[Message]:
        """Add message to session."""
        if session_id:
            session = self.get_session(session_id)
        else:
            session = self.get_current_session()

        if not session:
            return None

        msg = session.add_message(role, content, **metadata)
        self._save_session(session)
        return msg

    def set_active_analysis(
        self,
        analysis_id: str,
        session_id: Optional[str] = None,
    ) -> None:
        """Set active analysis for session."""
        if session_id:
            session = self.get_session(session_id)
        else:
            session = self.get_current_session()

        if session:
            session.active_analysis_id = analysis_id
            session.updated_at = datetime.now(timezone.utc).isoformat()
            self._save_session(session)

    def build_llm_context(
        self,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        include_analysis: bool = True,
        max_messages: int = 20,
    ) -> List[Dict[str, str]]:
        """
        Build context for LLM API call.

        Returns messages in OpenAI format.
        """
        if session_id:
            session = self.get_session(session_id)
        else:
            session = self.get_current_session()

        if not session:
            return []

        messages = []

        # Add system prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add context about current state
        context_parts = []
        if session.active_analysis_id:
            context_parts.append(f"Active analysis: {session.active_analysis_id}")
        if session.current_topic:
            context_parts.append(f"Topic: {session.current_topic}")

        if context_parts:
            messages.append({
                "role": "system",
                "content": f"Current context: {' | '.join(context_parts)}"
            })

        # Add recent messages
        messages.extend(session.get_context_for_llm(max_messages))

        return messages

    def get_resume_context(self, session_id: Optional[str] = None) -> str:
        """
        Get context string for resuming a session.

        Useful for MCP tool to restore state.
        """
        if session_id:
            session = self.get_session(session_id)
        else:
            session = self.get_current_session()

        if not session:
            return "No active session"

        parts = [
            f"Session: {session.session_id}",
            f"Created: {session.created_at}",
            f"Messages: {len(session.messages)}",
        ]

        if session.active_analysis_id:
            parts.append(f"Active analysis: {session.active_analysis_id}")

        if session.tool_calls:
            recent = session.tool_calls[-3:]
            tools = [tc["tool_name"] for tc in recent]
            parts.append(f"Recent tools: {', '.join(tools)}")

        # Add last few message summaries
        if session.messages:
            parts.append("\nRecent conversation:")
            for msg in session.messages[-3:]:
                preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                parts.append(f"  [{msg.role}]: {preview}")

        return "\n".join(parts)

    def cleanup_old_sessions(self, keep_last: int = 20) -> int:
        """Remove old session files."""
        if not self.sessions_dir.exists():
            return 0

        sessions = sorted(self.sessions_dir.glob("*.json"), reverse=True)
        deleted = 0

        for session_file in sessions[keep_last:]:
            session_file.unlink()
            deleted += 1

        logger.info(f"Cleaned up {deleted} old sessions")
        return deleted
