"""
Base Executor Interface
All executors inherit from this
"""
from abc import ABC, abstractmethod
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


class BaseExecutor(ABC):
    """Base class for all executors"""

    name: str = "base"

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        logger.info(f"Initialized executor: {self.name}")

    @abstractmethod
    async def run(self, action: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action with given inputs"""
        pass

    async def health_check(self) -> bool:
        """Check if executor is healthy"""
        return True

    def get_capabilities(self) -> list[str]:
        """Return list of supported actions"""
        return []
