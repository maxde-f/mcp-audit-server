#!/usr/bin/env python3
"""
Audit Platform CLI Runner
Run audits from command line, output JSON for Claude processing

Usage:
    python -m run --source /path/to/repo --task quick_scan
    python -m run --source https://github.com/user/repo --task full_audit --output results.json
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add core to path
sys.path.insert(0, str(Path(__file__).parent))

from core.engine import AuditEngine


async def run_audit(
    source: str,
    task: str = "quick_scan",
    branch: str = "main",
    contract_id: str = None,
    policy_id: str = None,
    region: str = "EU_UA",
    output_file: str = None,
    verbose: bool = False
) -> dict:
    """Run audit pipeline and return results"""

    # Determine source type
    source_type = "git" if source.startswith(("http://", "https://", "git@")) else "directory"

    # Initialize engine
    engine = AuditEngine(base_path=Path(__file__).parent / "core")

    # Register executors
    await register_executors(engine, verbose)

    # Progress callback
    async def on_progress(progress: dict):
        if verbose:
            stage = progress.get("stage_name", progress.get("stage", ""))
            pct = progress.get("progress", 0)
            print(f"  [{pct:5.1f}%] {stage}", file=sys.stderr)

    # Run workflow
    if verbose:
        print(f"Starting audit: {source}", file=sys.stderr)
        print(f"Task: {task}", file=sys.stderr)

    try:
        results = await engine.run_workflow(
            workflow_name="audit",
            inputs={
                "source_type": source_type,
                "source_path": source,
                "branch": branch,
                "task": task,
                "contract_id": contract_id,
                "policy_id": policy_id,
                "region": region,
            },
            progress_callback=on_progress if verbose else None
        )
    except Exception as e:
        results = {
            "error": str(e),
            "success": False
        }

    # Add metadata
    output = {
        "meta": {
            "source": source,
            "task": task,
            "timestamp": datetime.now().isoformat(),
            "version": "3.1"
        },
        "results": results,
        "success": "error" not in results
    }

    # Output
    if output_file:
        Path(output_file).write_text(json.dumps(output, indent=2, default=str))
        if verbose:
            print(f"Results saved to: {output_file}", file=sys.stderr)

    return output


async def register_executors(engine: AuditEngine, verbose: bool = False):
    """Register all executors with engine"""
    executors_path = Path(__file__).parent / "executors"

    # Import and register each executor
    executor_modules = [
        ("source-loader", "git-analyzer"),  # Use git-analyzer as source-loader for now
        ("scanner", "static-analyzer"),      # Use static-analyzer for scanning
        ("readiness-checker", None),         # TODO
        ("type-detector", None),             # TODO
        ("quality-analyzer", "static-analyzer"),
        ("compliance-checker", "contract-checker"),
        ("cost-estimator", "cost-estimator"),
        ("document-loader", "document-loader"),
        ("report-generator", "report-generator"),
        ("full-auditor", "llm-reviewer"),
    ]

    for executor_name, module_name in executor_modules:
        if module_name:
            try:
                module_path = executors_path / module_name / "executor.py"
                if module_path.exists():
                    # Dynamic import
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    executor = module.create_executor()
                    engine.register_executor(executor_name, executor)

                    if verbose:
                        print(f"  Registered: {executor_name}", file=sys.stderr)
            except Exception as e:
                if verbose:
                    print(f"  Warning: {executor_name} - {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Audit Platform - Repository Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --source /path/to/project --task quick_scan
  %(prog)s --source https://github.com/user/repo --task check_quality
  %(prog)s --source ./myproject --task check_compliance --policy global_fund_r13
  %(prog)s --source ./repo --task full_audit --output report.json

Tasks:
  quick_scan       - Fast scan (files, LOC, languages)
  detect_type      - Detect project type and framework
  check_quality    - Quality analysis (health, debt, security)
  check_compliance - Check contract/policy compliance
  estimate_cost    - Cost and effort estimation
  full_audit       - Complete audit with all checks
        """
    )

    parser.add_argument("--source", "-s", required=True,
                        help="Git URL or local directory path")
    parser.add_argument("--task", "-t", default="quick_scan",
                        choices=["quick_scan", "detect_type", "check_quality",
                                "check_compliance", "estimate_cost", "full_audit"],
                        help="Task to run (default: quick_scan)")
    parser.add_argument("--branch", "-b", default="main",
                        help="Git branch (default: main)")
    parser.add_argument("--contract", "-c",
                        help="Contract ID for compliance check")
    parser.add_argument("--policy", "-p",
                        help="Policy ID (e.g., global_fund_r13, standard, enterprise)")
    parser.add_argument("--region", "-r", default="EU_UA",
                        choices=["US", "EU_PL", "EU_UA"],
                        help="Region for cost estimation (default: EU_UA)")
    parser.add_argument("--output", "-o",
                        help="Output JSON file path")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Output JSON to stdout")

    args = parser.parse_args()

    # Run
    results = asyncio.run(run_audit(
        source=args.source,
        task=args.task,
        branch=args.branch,
        contract_id=args.contract,
        policy_id=args.policy,
        region=args.region,
        output_file=args.output,
        verbose=args.verbose
    ))

    # Output to stdout if requested
    if args.json or not args.output:
        print(json.dumps(results, indent=2, default=str))

    # Exit code
    sys.exit(0 if results.get("success") else 1)


if __name__ == "__main__":
    main()
