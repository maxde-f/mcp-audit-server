"""
Document Loader Executor
Loads and parses contracts and policies from PDF/DOCX files
Based on: repo-auditor/backend/app/services/contract_parser.py
"""
import re
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from base import BaseExecutor

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Requirement:
    """Contract/policy requirement"""
    id: str
    category: str  # quality, security, documentation, testing, etc.
    description: str
    metric: Optional[str] = None  # e.g., "repo_health", "test_coverage"
    threshold: Optional[float] = None  # e.g., 8 (for repo_health >= 8)
    operator: str = ">="  # >=, <=, ==, >
    priority: str = "required"  # required, recommended, optional

    def check(self, value: float) -> bool:
        """Check if value meets requirement"""
        if self.threshold is None:
            return True
        ops = {
            ">=": lambda v, t: v >= t,
            "<=": lambda v, t: v <= t,
            "==": lambda v, t: v == t,
            ">": lambda v, t: v > t,
            "<": lambda v, t: v < t,
        }
        return ops.get(self.operator, lambda v, t: True)(value, self.threshold)


@dataclass
class ParsedContract:
    """Parsed contract data"""
    id: str
    name: str
    type: str  # contract, policy, specification

    # Metadata
    date: Optional[str] = None
    parties: List[str] = field(default_factory=list)

    # Requirements
    requirements: List[Requirement] = field(default_factory=list)

    # Financial
    budget_total: Optional[float] = None
    currency: str = "USD"

    # Summary
    summary: str = ""
    raw_text: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# SAVED TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════

SAVED_POLICIES = {
    "global_fund_r13": ParsedContract(
        id="global_fund_r13",
        name="Global Fund R13 Requirements",
        type="policy",
        requirements=[
            Requirement("gf1", "documentation", "README with setup instructions", "has_readme", 1.0, "==", "required"),
            Requirement("gf2", "testing", "Unit tests present", "has_tests", 1.0, "==", "required"),
            Requirement("gf3", "ci", "CI/CD pipeline", "has_ci", 1.0, "==", "required"),
            Requirement("gf4", "quality", "Minimum repo health", "repo_health", 8, ">=", "required"),
            Requirement("gf5", "security", "No critical vulnerabilities", "security_score", 2, ">=", "required"),
            Requirement("gf6", "documentation", "API documentation", "has_api_docs", 1.0, "==", "recommended"),
        ],
        summary="Global Fund Round 13 technical requirements for software deliverables"
    ),
    "standard": ParsedContract(
        id="standard",
        name="Standard Requirements",
        type="policy",
        requirements=[
            Requirement("std1", "documentation", "README present", "has_readme", 1.0, "==", "required"),
            Requirement("std2", "quality", "Basic repo health", "repo_health", 6, ">=", "required"),
            Requirement("std3", "quality", "Manageable tech debt", "tech_debt", 8, ">=", "required"),
        ],
        summary="Standard baseline requirements"
    ),
    "enterprise": ParsedContract(
        id="enterprise",
        name="Enterprise Requirements",
        type="policy",
        requirements=[
            Requirement("ent1", "documentation", "Full documentation", "has_readme", 1.0, "==", "required"),
            Requirement("ent2", "testing", "Tests required", "has_tests", 1.0, "==", "required"),
            Requirement("ent3", "ci", "CI/CD required", "has_ci", 1.0, "==", "required"),
            Requirement("ent4", "deployment", "Docker required", "has_docker", 1.0, "==", "required"),
            Requirement("ent5", "quality", "High repo health", "repo_health", 10, ">=", "required"),
            Requirement("ent6", "quality", "Low tech debt", "tech_debt", 12, ">=", "required"),
            Requirement("ent7", "security", "Security audit passed", "security_score", 2, ">=", "required"),
        ],
        summary="Enterprise-grade requirements for production systems"
    ),
}


class DocumentLoaderExecutor(BaseExecutor):
    """Executor for loading and parsing documents"""

    name = "document-loader"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.storage_dir = Path(config.get("storage_dir", "/tmp/audit-documents") if config else "/tmp/audit-documents")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    async def run(self, action: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if action == "load":
            return await self.load(**inputs)
        elif action == "parse_file":
            return await self.parse_file(**inputs)
        elif action == "list_saved":
            return await self.list_saved()
        raise ValueError(f"Unknown action: {action}")

    async def load(
        self,
        contract_file: Optional[str] = None,
        policy_file: Optional[str] = None,
        contract_id: Optional[str] = None,
        policy_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Load contract and policy documents.

        Can load from:
        - File path (contract_file, policy_file)
        - Saved ID (contract_id, policy_id)
        """
        contract_parsed = None
        policy_parsed = None

        # Load contract
        if contract_file:
            contract_parsed = await self._parse_document(contract_file, "contract")
        elif contract_id:
            contract_parsed = await self._load_saved(contract_id)

        # Load policy
        if policy_file:
            policy_parsed = await self._parse_document(policy_file, "policy")
        elif policy_id:
            policy_parsed = await self._load_saved(policy_id)

        return {
            "contract_parsed": asdict(contract_parsed) if contract_parsed else None,
            "policy_parsed": asdict(policy_parsed) if policy_parsed else None,
            "contract_summary": contract_parsed.summary if contract_parsed else None,
            "policy_summary": policy_parsed.summary if policy_parsed else None,
        }

    async def parse_file(self, file_path: str, doc_type: str = "contract", **kwargs) -> Dict[str, Any]:
        """Parse a single document file"""
        parsed = await self._parse_document(file_path, doc_type)
        return asdict(parsed) if parsed else {"error": "Failed to parse"}

    async def list_saved(self) -> Dict[str, Any]:
        """List saved policies"""
        return {
            "policies": [
                {"id": k, "name": v.name, "requirements": len(v.requirements)}
                for k, v in SAVED_POLICIES.items()
            ]
        }

    async def _load_saved(self, doc_id: str) -> Optional[ParsedContract]:
        """Load a saved policy by ID"""
        return SAVED_POLICIES.get(doc_id)

    async def _parse_document(self, file_path: str, doc_type: str) -> Optional[ParsedContract]:
        """Parse document from file"""
        path = Path(file_path)

        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        # Extract text based on file type
        text = await self._extract_text(path)
        if not text:
            return None

        # Generate ID
        doc_id = hashlib.md5(text.encode()).hexdigest()[:8]

        # Extract requirements using patterns and LLM
        requirements = await self._extract_requirements(text, doc_type)

        # Extract metadata
        name = path.stem
        summary = text[:500] + "..." if len(text) > 500 else text

        return ParsedContract(
            id=doc_id,
            name=name,
            type=doc_type,
            requirements=requirements,
            summary=summary,
            raw_text=text[:10000]  # Limit stored text
        )

    async def _extract_text(self, path: Path) -> Optional[str]:
        """Extract text from file"""
        suffix = path.suffix.lower()

        if suffix == ".txt":
            return path.read_text(errors="ignore")

        elif suffix == ".pdf":
            return await self._extract_pdf(path)

        elif suffix in [".docx", ".doc"]:
            return await self._extract_docx(path)

        elif suffix == ".json":
            data = json.loads(path.read_text())
            return json.dumps(data, indent=2)

        else:
            logger.warning(f"Unsupported file type: {suffix}")
            return None

    async def _extract_pdf(self, path: Path) -> Optional[str]:
        """Extract text from PDF"""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(path))
            text_parts = [page.get_text() for page in doc]
            doc.close()
            return "\n".join(text_parts)
        except ImportError:
            try:
                import pdfplumber
                with pdfplumber.open(str(path)) as pdf:
                    return "\n".join(page.extract_text() or "" for page in pdf.pages)
            except ImportError:
                logger.error("No PDF library available (install pymupdf or pdfplumber)")
                return None

    async def _extract_docx(self, path: Path) -> Optional[str]:
        """Extract text from DOCX"""
        try:
            from docx import Document
            doc = Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            logger.error("python-docx not installed")
            return None

    async def _extract_requirements(self, text: str, doc_type: str) -> List[Requirement]:
        """Extract requirements from text using patterns"""
        requirements = []
        text_lower = text.lower()

        # Pattern-based extraction
        patterns = [
            # Quality requirements
            (r"(?:must|shall|required).{0,50}(?:documentation|readme)",
             Requirement("doc1", "documentation", "Documentation required", "has_readme", 1.0, "==", "required")),
            (r"(?:must|shall|required).{0,50}(?:test|testing|unit test)",
             Requirement("test1", "testing", "Tests required", "has_tests", 1.0, "==", "required")),
            (r"(?:must|shall|required).{0,50}(?:ci|continuous integration|pipeline)",
             Requirement("ci1", "ci", "CI/CD required", "has_ci", 1.0, "==", "required")),
            (r"(?:must|shall|required).{0,50}(?:docker|container)",
             Requirement("docker1", "deployment", "Containerization required", "has_docker", 1.0, "==", "required")),

            # Score requirements
            (r"(?:repo|repository).{0,20}health.{0,20}(?:>=?|at least|minimum)\s*(\d+)",
             lambda m: Requirement("health1", "quality", f"Repo health >= {m.group(1)}", "repo_health", float(m.group(1)), ">=", "required")),
            (r"(?:tech|technical).{0,10}debt.{0,20}(?:>=?|at least|minimum)\s*(\d+)",
             lambda m: Requirement("debt1", "quality", f"Tech debt >= {m.group(1)}", "tech_debt", float(m.group(1)), ">=", "required")),
            (r"security.{0,20}(?:score|level).{0,20}(?:>=?|at least|minimum)\s*(\d+)",
             lambda m: Requirement("sec1", "security", f"Security >= {m.group(1)}", "security_score", float(m.group(1)), ">=", "required")),
        ]

        for pattern, req_or_func in patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                if callable(req_or_func):
                    requirements.append(req_or_func(match))
                else:
                    requirements.append(req_or_func)

        # Deduplicate by ID
        seen_ids = set()
        unique_reqs = []
        for req in requirements:
            if req.id not in seen_ids:
                seen_ids.add(req.id)
                unique_reqs.append(req)

        return unique_reqs

    def get_capabilities(self) -> list[str]:
        return ["load", "parse_file", "list_saved"]


def create_executor(config: Dict[str, Any] = None) -> DocumentLoaderExecutor:
    return DocumentLoaderExecutor(config)
