"""
Audit Platform Core Engine
Orchestrates workflows using YAML definitions
"""
import yaml
import asyncio
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    stage_id: str
    status: StageStatus
    outputs: dict = field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: float = 0


@dataclass
class WorkflowContext:
    analysis_id: str
    inputs: dict
    stages: dict = field(default_factory=dict)
    current_stage: Optional[str] = None
    progress_callback: Optional[Callable] = None


class AuditEngine:
    """Core engine that executes audit workflows"""

    def __init__(self, base_path: Path = None):
        self.base_path = base_path or Path(__file__).parent
        self.workflows = {}
        self.rules = {}
        self.knowledge = {}
        self.executors = {}
        self._load_configs()

    def _load_configs(self):
        """Load all YAML configurations"""
        # Load workflows
        workflows_path = self.base_path / "workflows"
        if workflows_path.exists():
            for f in workflows_path.glob("*.yaml"):
                with open(f) as fp:
                    self.workflows[f.stem] = yaml.safe_load(fp)

        # Load rules
        rules_path = self.base_path / "rules"
        if rules_path.exists():
            for f in rules_path.glob("*.yaml"):
                with open(f) as fp:
                    self.rules[f.stem] = yaml.safe_load(fp)

        # Load knowledge
        knowledge_path = self.base_path / "knowledge"
        if knowledge_path.exists():
            for f in knowledge_path.glob("*.yaml"):
                with open(f) as fp:
                    self.knowledge[f.stem] = yaml.safe_load(fp)

        logger.info(f"Loaded {len(self.workflows)} workflows, {len(self.rules)} rules, {len(self.knowledge)} knowledge bases")

    def register_executor(self, name: str, executor: Any):
        """Register an executor for a stage type"""
        self.executors[name] = executor
        logger.info(f"Registered executor: {name}")

    async def run_workflow(
        self,
        workflow_name: str,
        inputs: dict,
        analysis_id: str = None,
        progress_callback: Callable = None
    ) -> dict:
        """Execute a workflow by name"""
        if workflow_name not in self.workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}")

        workflow = self.workflows[workflow_name]
        analysis_id = analysis_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        context = WorkflowContext(
            analysis_id=analysis_id,
            inputs=inputs,
            progress_callback=progress_callback
        )

        results = {}
        total_stages = len(workflow.get("stages", []))

        for idx, stage in enumerate(workflow.get("stages", [])):
            stage_id = stage["id"]
            context.current_stage = stage_id

            # Report progress
            if progress_callback:
                await progress_callback({
                    "stage": stage_id,
                    "stage_name": stage["name"],
                    "progress": (idx / total_stages) * 100,
                    "status": "running"
                })

            # Check skip conditions
            if self._should_skip_stage(stage, inputs):
                results[stage_id] = StageResult(
                    stage_id=stage_id,
                    status=StageStatus.SKIPPED
                )
                continue

            # Execute stage
            try:
                result = await self._execute_stage(stage, context)
                results[stage_id] = result
                context.stages[stage_id] = result.outputs
            except Exception as e:
                logger.error(f"Stage {stage_id} failed: {e}")
                results[stage_id] = StageResult(
                    stage_id=stage_id,
                    status=StageStatus.FAILED,
                    error=str(e)
                )
                # Continue or fail based on workflow config
                if not stage.get("continue_on_error", False):
                    break

        # Calculate final scores
        final_scores = self._calculate_scores(context.stages)

        # Report completion
        if progress_callback:
            await progress_callback({
                "stage": "completed",
                "progress": 100,
                "status": "completed",
                "scores": final_scores
            })

        return {
            "analysis_id": analysis_id,
            "stages": {k: v.__dict__ for k, v in results.items()},
            "scores": final_scores,
            "outputs": self._collect_outputs(workflow, context)
        }

    def _should_skip_stage(self, stage: dict, inputs: dict) -> bool:
        """Check if stage should be skipped based on profile"""
        skip_on = stage.get("skip_on_profile")
        if skip_on and inputs.get("profile") == skip_on:
            return True
        return False

    async def _execute_stage(self, stage: dict, context: WorkflowContext) -> StageResult:
        """Execute a single workflow stage"""
        start_time = datetime.now()
        executor_name = stage.get("executor", "core")

        if executor_name not in self.executors:
            # Use mock executor for missing executors
            logger.warning(f"No executor for {executor_name}, using mock")
            outputs = {"mock": True, "stage": stage["id"]}
        else:
            executor = self.executors[executor_name]
            action = stage.get("action", "run")

            # Resolve inputs
            resolved_inputs = self._resolve_inputs(stage.get("inputs", {}), context)

            # Execute
            method = getattr(executor, action, None)
            if method:
                outputs = await method(**resolved_inputs)
            else:
                outputs = await executor.run(action, resolved_inputs)

        duration = (datetime.now() - start_time).total_seconds() * 1000

        return StageResult(
            stage_id=stage["id"],
            status=StageStatus.COMPLETED,
            outputs=outputs,
            duration_ms=duration
        )

    def _resolve_inputs(self, inputs: dict, context: WorkflowContext) -> dict:
        """Resolve template variables in inputs"""
        resolved = {}
        for key, value in inputs.items():
            if isinstance(value, str) and "{{" in value:
                resolved[key] = self._resolve_template(value, context)
            else:
                resolved[key] = value
        return resolved

    def _resolve_template(self, template: str, context: WorkflowContext) -> Any:
        """Resolve a single template string"""
        # Simple template resolution
        # {{ inputs.repo_url }} -> context.inputs["repo_url"]
        # {{ stages.fetch.repo_path }} -> context.stages["fetch"]["repo_path"]
        import re
        match = re.search(r"\{\{\s*(.+?)\s*\}\}", template)
        if not match:
            return template

        path = match.group(1)
        parts = path.split(".")

        if parts[0] == "inputs":
            return context.inputs.get(parts[1])
        elif parts[0] == "stages":
            stage_outputs = context.stages.get(parts[1], {})
            return stage_outputs.get(parts[2]) if len(parts) > 2 else stage_outputs
        elif parts[0] == "context":
            return getattr(context, parts[1], None)

        return template

    def _calculate_scores(self, stage_outputs: dict) -> dict:
        """Calculate scores based on rules"""
        if "scoring" not in self.rules:
            return {}

        rules = self.rules["scoring"]
        scores = {}

        # Collect all metrics from stages
        all_metrics = {}
        for stage_id, outputs in stage_outputs.items():
            if isinstance(outputs, dict):
                all_metrics.update(outputs)

        # Calculate repo_health
        repo_health = 0
        for metric in rules.get("repo_health", {}).get("metrics", []):
            metric_name = metric["name"]
            if metric_name in all_metrics:
                value = all_metrics[metric_name]
                if self._check_condition(metric.get("condition", "true"), value):
                    repo_health += metric.get("points", 0)
        scores["repo_health"] = repo_health

        # Calculate tech_debt
        tech_debt = 0
        for metric in rules.get("tech_debt", {}).get("metrics", []):
            metric_name = metric["name"]
            if metric_name in all_metrics:
                value = all_metrics[metric_name]
                conditions = metric.get("conditions", [])
                for cond in conditions:
                    if self._check_threshold(cond.get("value", ""), value):
                        tech_debt += cond.get("points", 0)
                        break
        scores["tech_debt"] = tech_debt

        # Determine product level
        scores["product_level"] = self._determine_product_level(scores, all_metrics)

        # Calculate overall readiness
        max_health = rules.get("repo_health", {}).get("max_score", 12)
        max_debt = rules.get("tech_debt", {}).get("max_score", 15)
        scores["overall_readiness"] = round(
            (scores["repo_health"] / max_health * 40) +
            (scores["tech_debt"] / max_debt * 40) +
            (all_metrics.get("security_score", 0) / 3 * 20),
            1
        )

        return scores

    def _check_condition(self, condition: str, value: Any) -> bool:
        """Evaluate a simple condition"""
        if "==" in condition:
            return value == eval(condition.split("==")[1].strip())
        return True

    def _check_threshold(self, threshold: str, value: Any) -> bool:
        """Check threshold conditions like '>= 80'"""
        import re
        match = re.match(r"([<>=]+)\s*(\d+)", threshold)
        if not match:
            return False
        op, num = match.groups()
        num = float(num)
        if op == ">=":
            return value >= num
        elif op == "<=":
            return value <= num
        elif op == ">":
            return value > num
        elif op == "<":
            return value < num
        return False

    def _determine_product_level(self, scores: dict, metrics: dict) -> str:
        """Determine product level based on scores"""
        levels = self.rules.get("scoring", {}).get("product_level", {}).get("levels", [])

        for level in reversed(levels):  # Start from highest
            reqs = level.get("requirements", {})
            matches = True

            for key, req in reqs.items():
                if key in scores:
                    if isinstance(req, dict):
                        if not (req.get("min", 0) <= scores[key] <= req.get("max", 100)):
                            matches = False
                            break
                elif key in metrics:
                    if metrics[key] != req:
                        matches = False
                        break

            if matches:
                return level["name"]

        return "R&D Spike"

    def _collect_outputs(self, workflow: dict, context: WorkflowContext) -> dict:
        """Collect final workflow outputs"""
        outputs = {}
        for key, template in workflow.get("outputs", {}).items():
            outputs[key] = self._resolve_template(str(template), context)
        return outputs

    def get_metric_explanation(self, metric_name: str, value: Any) -> dict:
        """Get human-readable explanation for a metric"""
        if "metrics" not in self.knowledge:
            return {"error": "Knowledge base not loaded"}

        metrics = self.knowledge["metrics"].get("metrics", {})
        if metric_name not in metrics:
            return {"error": f"Unknown metric: {metric_name}"}

        metric_info = metrics[metric_name]
        max_val = metric_info.get("max_value", 100)
        percent = (value / max_val) * 100 if max_val else 0

        # Determine status
        if percent >= 80:
            status = "excellent"
        elif percent >= 60:
            status = "good"
        elif percent >= 40:
            status = "acceptable"
        else:
            status = "needs attention"

        return {
            "display_name": metric_info.get("display_name", metric_name),
            "value": value,
            "max_value": max_val,
            "percent": round(percent, 1),
            "status": status,
            "meaning": metric_info.get("business_meaning", ""),
            "analogy": metric_info.get("analogies", [""])[0],
            "impact": metric_info.get("impact", {}).get(status, "")
        }

    def get_product_level_info(self, level_name: str) -> dict:
        """Get info about a product level"""
        if "metrics" not in self.knowledge:
            return {"error": "Knowledge base not loaded"}

        levels = self.knowledge["metrics"].get("product_levels", {})
        return levels.get(level_name, {"error": f"Unknown level: {level_name}"})


# Singleton instance
_engine = None


def get_engine() -> AuditEngine:
    """Get or create the engine singleton"""
    global _engine
    if _engine is None:
        _engine = AuditEngine()
    return _engine
