"""
LLM Reviewer Executor
Uses LLM for code review and analysis
"""
import asyncio
import httpx
import os
from pathlib import Path
from typing import Any, Dict
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from base import BaseExecutor

logger = logging.getLogger(__name__)


class LLMReviewerExecutor(BaseExecutor):
    """Executor for LLM-based code review"""

    name = "llm-reviewer"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.groq_api_key = os.getenv("GROQ_API_KEY") or (config or {}).get("groq_api_key")
        self.model = (config or {}).get("model", "llama-3.3-70b-versatile")
        self.base_url = "https://api.groq.com/openai/v1"

    async def run(self, action: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Route to appropriate method"""
        if action == "review":
            return await self.review(**inputs)
        elif action == "analyze_readme":
            return await self.analyze_readme(**inputs)
        else:
            raise ValueError(f"Unknown action: {action}")

    async def review(self, path: str, structure: dict = None, security: dict = None, **kwargs) -> Dict[str, Any]:
        """Review code and generate insights"""
        repo_path = Path(path)

        # Read README
        readme_content = ""
        for name in ["README.md", "README.rst", "README.txt", "README"]:
            readme_path = repo_path / name
            if readme_path.exists():
                try:
                    readme_content = readme_path.read_text(errors="ignore")[:5000]
                except:
                    pass
                break

        # Analyze README quality
        readme_quality = await self._analyze_readme(readme_content) if readme_content else 0

        # Collect code samples for architecture analysis
        code_samples = await self._collect_code_samples(repo_path, structure or {})

        # Get architecture notes
        architecture_notes = await self._analyze_architecture(code_samples, structure or {})

        # Generate recommendations
        recommendations = await self._generate_recommendations(
            structure or {},
            security or {},
            readme_quality
        )

        return {
            "readme_quality": readme_quality,
            "architecture_notes": architecture_notes,
            "recommendations": recommendations
        }

    async def analyze_readme(self, content: str, **kwargs) -> Dict[str, Any]:
        """Analyze README quality"""
        score = await self._analyze_readme(content)
        return {"readme_quality": score}

    async def _analyze_readme(self, content: str) -> int:
        """Score README quality (0-10)"""
        if not content:
            return 0

        score = 0

        # Length check
        if len(content) > 500:
            score += 2
        elif len(content) > 200:
            score += 1

        # Section checks
        sections = ["install", "usage", "example", "api", "license", "contribut"]
        for section in sections:
            if section in content.lower():
                score += 1

        # Has code blocks
        if "```" in content:
            score += 1

        return min(score, 10)

    async def _collect_code_samples(self, repo_path: Path, structure: dict) -> str:
        """Collect representative code samples"""
        samples = []
        languages = structure.get("languages", [])

        # Map language to extensions
        ext_map = {
            "Python": ".py",
            "JavaScript": ".js",
            "TypeScript": ".ts",
            "Go": ".go"
        }

        for lang in languages[:2]:  # Limit to 2 languages
            ext = ext_map.get(lang)
            if not ext:
                continue

            files = list(repo_path.rglob(f"*{ext}"))
            # Filter out tests and configs
            files = [f for f in files if "test" not in str(f).lower() and "config" not in str(f).lower()]

            for f in files[:2]:  # 2 files per language
                try:
                    content = f.read_text(errors="ignore")[:2000]
                    samples.append(f"# {f.name}\n{content}")
                except:
                    pass

        return "\n\n---\n\n".join(samples)[:8000]

    async def _analyze_architecture(self, code_samples: str, structure: dict) -> str:
        """Analyze architecture using LLM"""
        if not self.groq_api_key or not code_samples:
            return "Architecture analysis requires LLM API key"

        prompt = f"""Analyze this code and describe the architecture in 2-3 sentences.
Focus on: patterns used, main components, organization quality.

Languages: {', '.join(structure.get('languages', ['Unknown']))}
Has tests: {structure.get('has_tests', False)}
Has CI: {structure.get('has_ci', False)}

Code samples:
{code_samples[:4000]}

Respond with only the architecture description, no preamble."""

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.groq_api_key}"},
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 300,
                        "temperature": 0.3
                    }
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
            return "Unable to analyze architecture"

    async def _generate_recommendations(
        self,
        structure: dict,
        security: dict,
        readme_quality: int
    ) -> list[str]:
        """Generate improvement recommendations"""
        recommendations = []

        # Documentation
        if not structure.get("has_readme"):
            recommendations.append("Add a README with project description and setup instructions")
        elif readme_quality < 5:
            recommendations.append("Improve README with usage examples and API documentation")

        # Testing
        if not structure.get("has_tests"):
            recommendations.append("Add unit tests to ensure code reliability")

        # CI/CD
        if not structure.get("has_ci"):
            recommendations.append("Setup CI pipeline for automated testing")

        # Docker
        if not structure.get("has_docker"):
            recommendations.append("Add Dockerfile for consistent deployment")

        # Security
        if security.get("secrets_found", 0) > 0:
            recommendations.append("CRITICAL: Remove hardcoded secrets and use environment variables")

        if security.get("critical_count", 0) > 0:
            recommendations.append("CRITICAL: Fix critical security vulnerabilities immediately")

        return recommendations[:5]

    def get_capabilities(self) -> list[str]:
        return ["review", "analyze_readme"]


def create_executor(config: Dict[str, Any] = None) -> LLMReviewerExecutor:
    return LLMReviewerExecutor(config)
