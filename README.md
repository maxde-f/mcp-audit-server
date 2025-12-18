# Audit Platform

Clean microservices architecture for repository auditing.

## Architecture

```
audit-platform/
├── core/                    # Brain - Rules & Workflows
│   ├── workflows/          # YAML workflow definitions
│   ├── rules/              # Scoring rules
│   ├── knowledge/          # Metrics knowledge base
│   └── engine.py           # Workflow orchestrator
├── executors/              # Hands - Execution modules
│   ├── git-analyzer/       # Git operations
│   ├── static-analyzer/    # Code analysis
│   ├── security-scanner/   # Security scanning
│   ├── llm-reviewer/       # LLM-based review
│   └── report-generator/   # Report generation
├── gateway/                # API Layer
│   ├── mcp/               # MCP server for Claude
│   └── api/               # REST API for web
└── ui/                    # Web interface
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Add your GROQ_API_KEY
```

### 3. Run the API

```bash
python -m gateway.api.main
# API available at http://localhost:8080
```

### 4. Open Web UI

Open `ui/index.html` in browser or serve with:

```bash
python -m http.server 3000 -d ui
```

## Usage with Claude (MCP)

Add to your Claude config:

```json
{
  "mcpServers": {
    "audit-platform": {
      "command": "python",
      "args": ["-m", "gateway.mcp.server"],
      "cwd": "/path/to/audit-platform"
    }
  }
}
```

Then in Claude:
- "Audit the repository https://github.com/user/repo"
- "Explain what repo_health score means"
- "What level is this project and how to improve it?"

## Available Tools (MCP)

| Tool | Description |
|------|-------------|
| `audit_repository` | Full repository audit |
| `explain_metric` | Explain metrics for non-technical users |
| `explain_product_level` | Explain product classification |
| `get_recommendations` | Get improvement recommendations |
| `compare_with_contract` | Compare with requirements |

## Scoring

### Repository Health (0-12)
- README (+2)
- License (+1)
- Tests (+2)
- CI/CD (+2)
- Docker (+1)
- Active commits (+2)
- Multiple contributors (+2)

### Technical Debt (0-15)
- Low complexity (+3)
- Low duplication (+3)
- Few lint issues (+3)
- Updated dependencies (+3)
- Good test coverage (+3)

### Product Levels

| Level | Health | Debt | Description |
|-------|--------|------|-------------|
| R&D Spike | 0-3 | 0-5 | Experimental |
| Prototype | 4-6 | 4-8 | Working demo |
| Internal Tool | 6-8 | 7-10 | Team-ready |
| Platform Module | 8-10 | 10-13 | Integration-ready |
| Near-Product | 10-12 | 12-15 | Production-ready |

## API Endpoints

### REST API (port 8080)

```
GET  /health                    # Health check
GET  /api/workflows             # List workflows
POST /api/audit                 # Start audit
GET  /api/audit/{id}            # Get status
GET  /api/audit/{id}/report     # Get report
WS   /api/ws/audit/{id}         # Progress updates
POST /api/explain/metric        # Explain metric
GET  /api/explain/level/{name}  # Explain level
POST /api/recommendations       # Get recommendations
```
