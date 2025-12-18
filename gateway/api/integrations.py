"""
Integrations Service for Audit Platform

Provides:
- GitHub API integration (repos, files, branches)
- Google Drive integration (documents, policies)
- Document management (upload, parse, store)
"""
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

logger = logging.getLogger(__name__)

# Storage for uploaded documents
DOCUMENTS_DIR = Path(__file__).parent.parent.parent / "data" / "documents"
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

CONTRACTS_DIR = DOCUMENTS_DIR / "contracts"
CONTRACTS_DIR.mkdir(exist_ok=True)

POLICIES_DIR = DOCUMENTS_DIR / "policies"
POLICIES_DIR.mkdir(exist_ok=True)


# =============================================================================
# MODELS
# =============================================================================

class GitHubRepoRequest(BaseModel):
    """GitHub repository request."""
    url: str
    branch: str = "main"
    token: Optional[str] = None


class GoogleDriveRequest(BaseModel):
    """Google Drive request."""
    file_id: str
    credentials_json: Optional[str] = None


class DocumentMetadata(BaseModel):
    """Document metadata."""
    id: str
    name: str
    doc_type: str  # contract, policy
    uploaded_at: str
    file_path: str
    file_size: int
    parsed: bool = False
    requirements: List[Dict[str, Any]] = Field(default_factory=list)


# =============================================================================
# GITHUB INTEGRATION
# =============================================================================

class GitHubService:
    """GitHub API integration."""

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.api_base = "https://api.github.com"

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        return headers

    async def get_repo_info(self, owner: str, repo: str) -> Dict[str, Any]:
        """Get repository information."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.api_base}/repos/{owner}/{repo}",
                    headers=self._headers()
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "name": data.get("name"),
                        "full_name": data.get("full_name"),
                        "description": data.get("description"),
                        "default_branch": data.get("default_branch"),
                        "language": data.get("language"),
                        "stars": data.get("stargazers_count"),
                        "forks": data.get("forks_count"),
                        "size": data.get("size"),
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                        "clone_url": data.get("clone_url"),
                        "topics": data.get("topics", []),
                    }
                elif resp.status_code == 404:
                    return {"error": "Repository not found"}
                elif resp.status_code == 403:
                    return {"error": "Rate limited or unauthorized"}
                else:
                    return {"error": f"API error: {resp.status_code}"}
        except ImportError:
            return {"error": "httpx not installed. Run: pip install httpx"}
        except Exception as e:
            return {"error": str(e)}

    async def list_branches(self, owner: str, repo: str) -> List[str]:
        """List repository branches."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.api_base}/repos/{owner}/{repo}/branches",
                    headers=self._headers()
                )
                if resp.status_code == 200:
                    return [b["name"] for b in resp.json()]
                return []
        except:
            return []

    async def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        branch: str = "main"
    ) -> Optional[str]:
        """Get file content from repository."""
        try:
            import httpx
            import base64
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.api_base}/repos/{owner}/{repo}/contents/{path}",
                    params={"ref": branch},
                    headers=self._headers()
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("encoding") == "base64":
                        return base64.b64decode(data["content"]).decode("utf-8")
                return None
        except:
            return None

    async def list_user_repos(self, username: Optional[str] = None) -> List[Dict[str, Any]]:
        """List repositories for user."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                if username:
                    url = f"{self.api_base}/users/{username}/repos"
                else:
                    url = f"{self.api_base}/user/repos"

                resp = await client.get(url, headers=self._headers())
                if resp.status_code == 200:
                    return [
                        {
                            "name": r["name"],
                            "full_name": r["full_name"],
                            "description": r.get("description"),
                            "language": r.get("language"),
                            "updated_at": r.get("updated_at"),
                        }
                        for r in resp.json()
                    ]
                return []
        except:
            return []

    def parse_github_url(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse owner and repo from GitHub URL."""
        import re
        patterns = [
            r"github\.com[/:]([^/]+)/([^/\.]+)",
            r"^([^/]+)/([^/]+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1), match.group(2).replace(".git", "")
        return None, None


# =============================================================================
# GOOGLE DRIVE INTEGRATION
# =============================================================================

class GoogleDriveService:
    """Google Drive API integration."""

    def __init__(self, credentials_json: Optional[str] = None):
        self.credentials_json = credentials_json or os.environ.get("GOOGLE_CREDENTIALS_JSON")
        self._service = None

    def _get_service(self):
        """Get Google Drive service (lazy initialization)."""
        if self._service is not None:
            return self._service

        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build

            if self.credentials_json:
                creds = service_account.Credentials.from_service_account_info(
                    json.loads(self.credentials_json),
                    scopes=["https://www.googleapis.com/auth/drive.readonly"]
                )
                self._service = build("drive", "v3", credentials=creds)
                return self._service
            return None
        except ImportError:
            logger.warning("Google API libraries not installed")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize Google Drive: {e}")
            return None

    async def get_file_metadata(self, file_id: str) -> Dict[str, Any]:
        """Get file metadata from Google Drive."""
        service = self._get_service()
        if not service:
            return {"error": "Google Drive not configured"}

        try:
            file = service.files().get(
                fileId=file_id,
                fields="id,name,mimeType,size,createdTime,modifiedTime"
            ).execute()
            return file
        except Exception as e:
            return {"error": str(e)}

    async def download_file(self, file_id: str, destination: Path) -> bool:
        """Download file from Google Drive."""
        service = self._get_service()
        if not service:
            return False

        try:
            from googleapiclient.http import MediaIoBaseDownload
            import io

            request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)

            done = False
            while not done:
                _, done = downloader.next_chunk()

            with open(destination, "wb") as f:
                f.write(fh.getvalue())

            return True
        except Exception as e:
            logger.error(f"Failed to download file: {e}")
            return False

    async def list_files(
        self,
        folder_id: Optional[str] = None,
        mime_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List files in Google Drive."""
        service = self._get_service()
        if not service:
            return []

        try:
            query_parts = []
            if folder_id:
                query_parts.append(f"'{folder_id}' in parents")
            if mime_type:
                query_parts.append(f"mimeType='{mime_type}'")

            query = " and ".join(query_parts) if query_parts else None

            results = service.files().list(
                q=query,
                pageSize=50,
                fields="files(id,name,mimeType,size,modifiedTime)"
            ).execute()

            return results.get("files", [])
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []


# =============================================================================
# DOCUMENT MANAGEMENT
# =============================================================================

class DocumentService:
    """Document management service."""

    def __init__(self):
        self.documents_index = DOCUMENTS_DIR / "index.json"
        self._ensure_index()

    def _ensure_index(self):
        """Ensure index file exists."""
        if not self.documents_index.exists():
            with open(self.documents_index, "w") as f:
                json.dump({"documents": []}, f)

    def _load_index(self) -> Dict[str, Any]:
        """Load documents index."""
        try:
            with open(self.documents_index) as f:
                return json.load(f)
        except:
            return {"documents": []}

    def _save_index(self, index: Dict[str, Any]):
        """Save documents index."""
        with open(self.documents_index, "w") as f:
            json.dump(index, f, indent=2)

    def upload_document(
        self,
        file_content: bytes,
        filename: str,
        doc_type: str = "contract",
    ) -> DocumentMetadata:
        """Upload and store a document."""
        from uuid import uuid4

        doc_id = str(uuid4())[:8]
        now = datetime.now(timezone.utc).isoformat()

        # Determine storage path
        if doc_type == "contract":
            storage_dir = CONTRACTS_DIR
        else:
            storage_dir = POLICIES_DIR

        file_path = storage_dir / f"{doc_id}_{filename}"
        file_path.write_bytes(file_content)

        metadata = DocumentMetadata(
            id=doc_id,
            name=filename,
            doc_type=doc_type,
            uploaded_at=now,
            file_path=str(file_path),
            file_size=len(file_content),
            parsed=False,
            requirements=[],
        )

        # Update index
        index = self._load_index()
        index["documents"].append(asdict(metadata))
        self._save_index(index)

        logger.info(f"Uploaded document: {filename} ({doc_id})")
        return metadata

    def parse_document(self, doc_id: str) -> Dict[str, Any]:
        """Parse document to extract requirements."""
        index = self._load_index()

        doc = None
        for d in index["documents"]:
            if d["id"] == doc_id:
                doc = d
                break

        if not doc:
            return {"error": "Document not found"}

        file_path = Path(doc["file_path"])
        if not file_path.exists():
            return {"error": "Document file not found"}

        # Extract requirements based on file type
        ext = file_path.suffix.lower()
        requirements = []

        try:
            if ext == ".pdf":
                requirements = self._parse_pdf(file_path)
            elif ext in [".docx", ".doc"]:
                requirements = self._parse_docx(file_path)
            elif ext == ".txt":
                requirements = self._parse_text(file_path)
            else:
                return {"error": f"Unsupported file type: {ext}"}

            # Update document metadata
            doc["parsed"] = True
            doc["requirements"] = requirements
            self._save_index(index)

            return {
                "id": doc_id,
                "requirements_count": len(requirements),
                "requirements": requirements,
            }
        except Exception as e:
            return {"error": str(e)}

    def _parse_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse PDF document."""
        try:
            import pypdf
            reader = pypdf.PdfReader(str(file_path))
            text = "\n".join(page.extract_text() for page in reader.pages)
            return self._extract_requirements(text)
        except ImportError:
            logger.warning("pypdf not installed")
            return []

    def _parse_docx(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse DOCX document."""
        try:
            import docx
            doc = docx.Document(str(file_path))
            text = "\n".join(p.text for p in doc.paragraphs)
            return self._extract_requirements(text)
        except ImportError:
            logger.warning("python-docx not installed")
            return []

    def _parse_text(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse text document."""
        text = file_path.read_text()
        return self._extract_requirements(text)

    def _extract_requirements(self, text: str) -> List[Dict[str, Any]]:
        """Extract requirements from text."""
        import re

        requirements = []

        # Look for common requirement patterns
        patterns = [
            # "must have", "should have", "required"
            (r"(?:must|shall|should|required to)\s+(?:have|include|provide|implement)\s+([^\.]+)", "mandatory"),
            # "minimum X%", "at least X"
            (r"(?:minimum|at least)\s+(\d+(?:\.\d+)?)\s*(%|percent|hours?|days?|weeks?)", "threshold"),
            # "coverage of X%"
            (r"(?:test|code)\s*coverage\s*(?:of|:)?\s*(\d+(?:\.\d+)?)\s*%", "coverage"),
            # Security requirements
            (r"(?:security|vulnerability)\s+(?:audit|scan|check)", "security"),
            # Documentation requirements
            (r"(?:documentation|readme|api\s*docs?)", "documentation"),
        ]

        for pattern, category in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                requirements.append({
                    "text": match.group(0).strip(),
                    "category": category,
                    "metric": self._infer_metric(match.group(0), category),
                })

        return requirements

    def _infer_metric(self, text: str, category: str) -> Optional[str]:
        """Infer metric from requirement text."""
        metric_mapping = {
            "coverage": "test_coverage",
            "security": "security_score",
            "documentation": "has_readme",
        }
        return metric_mapping.get(category)

    def list_documents(
        self,
        doc_type: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """List uploaded documents."""
        index = self._load_index()
        documents = index.get("documents", [])

        if doc_type:
            documents = [d for d in documents if d["doc_type"] == doc_type]

        return documents[:limit]

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        index = self._load_index()
        for doc in index.get("documents", []):
            if doc["id"] == doc_id:
                return doc
        return None

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document."""
        index = self._load_index()
        documents = index.get("documents", [])

        for i, doc in enumerate(documents):
            if doc["id"] == doc_id:
                # Delete file
                file_path = Path(doc["file_path"])
                if file_path.exists():
                    file_path.unlink()

                # Remove from index
                del documents[i]
                self._save_index(index)
                return True

        return False


# =============================================================================
# SINGLETON INSTANCES
# =============================================================================

_github_service: Optional[GitHubService] = None
_gdrive_service: Optional[GoogleDriveService] = None
_document_service: Optional[DocumentService] = None


def get_github_service(token: Optional[str] = None) -> GitHubService:
    global _github_service
    if _github_service is None or token:
        _github_service = GitHubService(token)
    return _github_service


def get_gdrive_service(credentials: Optional[str] = None) -> GoogleDriveService:
    global _gdrive_service
    if _gdrive_service is None or credentials:
        _gdrive_service = GoogleDriveService(credentials)
    return _gdrive_service


def get_document_service() -> DocumentService:
    global _document_service
    if _document_service is None:
        _document_service = DocumentService()
    return _document_service


# =============================================================================
# API ROUTER
# =============================================================================

router = APIRouter(prefix="/api/integrations", tags=["integrations"])


# --- GitHub ---

@router.get("/github/repo")
async def get_github_repo(url: str, token: Optional[str] = None):
    """Get GitHub repository info."""
    service = get_github_service(token)
    owner, repo = service.parse_github_url(url)

    if not owner or not repo:
        raise HTTPException(status_code=400, detail="Invalid GitHub URL")

    return await service.get_repo_info(owner, repo)


@router.get("/github/branches")
async def get_github_branches(url: str, token: Optional[str] = None):
    """List GitHub repository branches."""
    service = get_github_service(token)
    owner, repo = service.parse_github_url(url)

    if not owner or not repo:
        raise HTTPException(status_code=400, detail="Invalid GitHub URL")

    branches = await service.list_branches(owner, repo)
    return {"branches": branches}


@router.get("/github/file")
async def get_github_file(
    url: str,
    path: str,
    branch: str = "main",
    token: Optional[str] = None
):
    """Get file content from GitHub."""
    service = get_github_service(token)
    owner, repo = service.parse_github_url(url)

    if not owner or not repo:
        raise HTTPException(status_code=400, detail="Invalid GitHub URL")

    content = await service.get_file_content(owner, repo, path, branch)
    if content is None:
        raise HTTPException(status_code=404, detail="File not found")

    return {"content": content}


@router.get("/github/repos")
async def list_github_repos(username: Optional[str] = None, token: Optional[str] = None):
    """List GitHub repositories."""
    service = get_github_service(token)
    repos = await service.list_user_repos(username)
    return {"repos": repos}


# --- Google Drive ---

@router.get("/gdrive/file/{file_id}")
async def get_gdrive_file(file_id: str):
    """Get Google Drive file metadata."""
    service = get_gdrive_service()
    return await service.get_file_metadata(file_id)


@router.post("/gdrive/download/{file_id}")
async def download_gdrive_file(file_id: str, doc_type: str = "contract"):
    """Download file from Google Drive and store it."""
    gdrive = get_gdrive_service()
    docs = get_document_service()

    # Get metadata
    metadata = await gdrive.get_file_metadata(file_id)
    if "error" in metadata:
        raise HTTPException(status_code=400, detail=metadata["error"])

    # Download to temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)

    success = await gdrive.download_file(file_id, tmp_path)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to download file")

    # Upload to document service
    content = tmp_path.read_bytes()
    tmp_path.unlink()

    doc = docs.upload_document(content, metadata.get("name", "document"), doc_type)
    return {"document": asdict(doc)}


@router.get("/gdrive/list")
async def list_gdrive_files(folder_id: Optional[str] = None):
    """List Google Drive files."""
    service = get_gdrive_service()
    files = await service.list_files(folder_id)
    return {"files": files}


# --- Documents ---

@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    doc_type: str = Form("contract")
):
    """Upload a document."""
    service = get_document_service()
    content = await file.read()
    doc = service.upload_document(content, file.filename, doc_type)
    return {"document": asdict(doc)}


@router.get("/documents")
async def list_documents(doc_type: Optional[str] = None, limit: int = 50):
    """List uploaded documents."""
    service = get_document_service()
    docs = service.list_documents(doc_type, limit)
    return {"documents": docs}


@router.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get document by ID."""
    service = get_document_service()
    doc = service.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.post("/documents/{doc_id}/parse")
async def parse_document(doc_id: str):
    """Parse document to extract requirements."""
    service = get_document_service()
    result = service.parse_document(doc_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document."""
    service = get_document_service()
    if not service.delete_document(doc_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted"}


# --- Policies (pre-defined) ---

@router.get("/policies")
async def list_policies():
    """List available compliance policies."""
    # Built-in policies + uploaded
    builtin = [
        {
            "id": "global_fund_r13",
            "name": "Global Fund R13",
            "description": "Global Fund Round 13 requirements",
            "requirements": [
                {"metric": "has_readme", "threshold": True, "operator": "eq"},
                {"metric": "has_tests", "threshold": True, "operator": "eq"},
                {"metric": "test_coverage", "threshold": 40, "operator": "gte"},
                {"metric": "has_ci", "threshold": True, "operator": "eq"},
                {"metric": "security_score", "threshold": 2, "operator": "gte"},
                {"metric": "has_docker", "threshold": True, "operator": "eq"},
            ],
            "type": "builtin"
        },
        {
            "id": "standard",
            "name": "Standard Requirements",
            "description": "Basic quality requirements",
            "requirements": [
                {"metric": "has_readme", "threshold": True, "operator": "eq"},
                {"metric": "repo_health", "threshold": 6, "operator": "gte"},
                {"metric": "tech_debt", "threshold": 8, "operator": "gte"},
            ],
            "type": "builtin"
        },
        {
            "id": "enterprise",
            "name": "Enterprise Requirements",
            "description": "Enterprise-grade quality requirements",
            "requirements": [
                {"metric": "has_readme", "threshold": True, "operator": "eq"},
                {"metric": "has_tests", "threshold": True, "operator": "eq"},
                {"metric": "test_coverage", "threshold": 60, "operator": "gte"},
                {"metric": "has_ci", "threshold": True, "operator": "eq"},
                {"metric": "has_cd", "threshold": True, "operator": "eq"},
                {"metric": "security_score", "threshold": 3, "operator": "eq"},
                {"metric": "has_docker", "threshold": True, "operator": "eq"},
            ],
            "type": "builtin"
        },
    ]

    # Add uploaded policy documents
    service = get_document_service()
    uploaded = service.list_documents(doc_type="policy")
    for doc in uploaded:
        if doc.get("parsed") and doc.get("requirements"):
            builtin.append({
                "id": doc["id"],
                "name": doc["name"],
                "description": f"Uploaded policy: {doc['name']}",
                "requirements": doc["requirements"],
                "type": "uploaded"
            })

    return {"policies": builtin}


@router.get("/policies/{policy_id}")
async def get_policy(policy_id: str):
    """Get policy by ID."""
    policies = (await list_policies())["policies"]
    for p in policies:
        if p["id"] == policy_id:
            return p
    raise HTTPException(status_code=404, detail="Policy not found")
