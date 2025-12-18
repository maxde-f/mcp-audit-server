# MCP HTTP Server for Web Claude

HTTP-сервер з SSE транспортом для інтеграції з Web версією Claude.

## Можливості

- **MCP SSE Transport** - для Web Claude інтеграції
- **OAuth 2.0 з PKCE** - безпечна аутентифікація
- **PostgreSQL** - персистентна пам'ять (decisions, learnings, facts)
- **Redis** - кешування сесій
- **23 інструменти** - audit + estimation + memory + validation

## Швидкий старт

### 1. Запуск локально

```bash
cd /Users/maksymdemchenko/audit-platform
python3 -m gateway.mcp.http_server
```

Сервер буде доступний на http://localhost:8090

### 2. З PostgreSQL та Redis

```bash
DATABASE_URL=postgres://user:pass@localhost:5432/audit \
REDIS_URL=redis://localhost:6379 \
python3 -m gateway.mcp.http_server
```

### 3. Deployment (Railway/Render)

```bash
# Встановіть змінні оточення:
DATABASE_URL=postgres://...
REDIS_URL=redis://...
SERVER_URL=https://your-domain.railway.app
PORT=8090
```

## Endpoints

### MCP Protocol

| Endpoint | Метод | Опис |
|----------|-------|------|
| `/mcp/sse` | GET | SSE transport для Claude |
| `/mcp` | GET/POST | Streamable HTTP |
| `/mcp/message` | POST | JSON-RPC повідомлення |
| `/.well-known/mcp.json` | GET | MCP Discovery |

### OAuth 2.0

| Endpoint | Метод | Опис |
|----------|-------|------|
| `/authorize` | GET | Authorization endpoint |
| `/token` | POST | Token endpoint |
| `/register` | POST | Dynamic client registration |
| `/.well-known/oauth-authorization-server` | GET | OAuth metadata |

### API

| Endpoint | Метод | Опис |
|----------|-------|------|
| `/` | GET | Dashboard |
| `/health` | GET | Health check |
| `/api/tools` | GET | Список інструментів |
| `/api/rates` | GET | Регіональні ставки |

## Інструменти (23)

### Audit Tools (17)

| Інструмент | Опис |
|------------|------|
| `audit` | Повний аудит репозиторію |
| `estimate_cocomo` | COCOMO II Modern оцінка |
| `estimate_comprehensive` | Всі 8 методологій |
| `estimate_methodology` | Одна методологія |
| `list_methodologies` | Список методологій |
| `estimate_pert` | PERT 3-point аналіз |
| `estimate_ai_efficiency` | Порівняння Human vs AI |
| `calculate_roi` | ROI калькулятор |
| `get_regional_costs` | Вартість по 8 регіонах |
| `get_formulas` | Всі формули |
| `get_constants` | Всі константи |
| `upload_document` | Завантаження документів |
| `list_policies` | Список політик |
| `explain_metric` | Пояснення метрики |
| `explain_product_level` | Пояснення рівня продукту |
| `get_recommendations` | Рекомендації |
| `load_results` | Завантаження результатів |

### Memory Tools (6)

| Інструмент | Опис |
|------------|------|
| `store_memory` | Зберегти факт/контекст |
| `recall_memory` | Пригадати пам'ять |
| `record_decision` | Записати рішення (ADR) |
| `record_learning` | Записати урок |
| `get_context` | Отримати контекст |
| `validate_estimate` | Валідація оцінки |

## Пам'ять (Memory System)

### Типи записів

- **fact** - факти про проект
- **context** - контекстна інформація
- **preference** - налаштування користувача
- **observation** - спостереження
- **error** - помилки

### Приклади

```python
# Store memory
await execute_tool("store_memory", {
    "content": "Project uses FastAPI + PostgreSQL",
    "type": "fact",
    "tags": ["tech", "backend"]
})

# Recall memory
await execute_tool("recall_memory", {
    "query": "FastAPI",
    "type": "fact",
    "limit": 5
})

# Record decision
await execute_tool("record_decision", {
    "title": "Use PostgreSQL instead of MongoDB",
    "context": "Need relational data with complex queries",
    "decision": "PostgreSQL with pg_trgm for search",
    "consequences": ["Better query performance", "ACID compliance"],
    "alternatives": ["MongoDB", "SQLite"]
})

# Record learning
await execute_tool("record_learning", {
    "what_happened": "Estimate was 2x actual hours",
    "what_learned": "Include 30% buffer for legacy code",
    "pattern": "legacy_code_buffer"
})
```

## Anti-Hallucination Validation

### Bounds

| Параметр | Мінімум | Максимум |
|----------|---------|----------|
| Hourly Rate | $5 | $300 |
| Hours/KLOC | 2 | 200 |
| PERT Ratio | - | 10x |

### Приклад валідації

```python
result = await execute_tool("validate_estimate", {
    "total_hours": 500,
    "total_cost": 17500,
    "kloc": 10,
    "hourly_rate": 35
})

# Результат:
{
    "valid": True,
    "hours_per_kloc": 50,  # OK: 2-200
    "hourly_rate": 35,     # OK: $5-300
    "warnings": [],
    "errors": []
}
```

## Регіональні ставки

| Регіон | Junior | Middle | Senior | Lead | Architect |
|--------|--------|--------|--------|------|-----------|
| UA | $15 | $25 | $40 | $55 | $70 |
| UA_Compliance | $20 | $35 | $55 | $75 | $95 |
| PL | $25 | $40 | $60 | $80 | $100 |
| EU | $40 | $65 | $95 | $120 | $150 |
| DE | $50 | $80 | $110 | $140 | $175 |
| UK | $45 | $75 | $105 | $135 | $165 |
| US | $60 | $100 | $150 | $200 | $250 |
| IN | $10 | $18 | $30 | $45 | $60 |

## Database Schema

### PostgreSQL Tables

```sql
-- Memory
CREATE TABLE claude_memory (
    id VARCHAR(32) PRIMARY KEY,
    workspace_id VARCHAR(64),
    type VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    access_count INTEGER DEFAULT 0,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Decisions (ADR format)
CREATE TABLE claude_decisions (
    id VARCHAR(32) PRIMARY KEY,
    workspace_id VARCHAR(64),
    title VARCHAR(255) NOT NULL,
    context TEXT,
    decision TEXT NOT NULL,
    consequences TEXT[],
    alternatives TEXT[],
    status VARCHAR(20) DEFAULT 'accepted',
    created_at TIMESTAMP
);

-- Learnings
CREATE TABLE claude_learnings (
    id VARCHAR(32) PRIMARY KEY,
    workspace_id VARCHAR(64),
    what_happened TEXT NOT NULL,
    what_learned TEXT NOT NULL,
    correction TEXT,
    pattern VARCHAR(255),
    created_at TIMESTAMP
);

-- Sessions
CREATE TABLE claude_sessions (
    session_id VARCHAR(64) PRIMARY KEY,
    workspace_id VARCHAR(64),
    messages JSONB DEFAULT '[]',
    summary TEXT,
    context JSONB DEFAULT '{}',
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Analysis results
CREATE TABLE analysis_results (
    id VARCHAR(32) PRIMARY KEY,
    workspace_id VARCHAR(64),
    repo_url VARCHAR(512),
    repo_name VARCHAR(255),
    metrics JSONB,
    scores JSONB,
    cost_estimate JSONB,
    validation JSONB,
    status VARCHAR(20),
    created_at TIMESTAMP
);
```

## Підключення до Claude Web

1. Перейдіть на claude.ai
2. Відкрийте Settings → Integrations
3. Додайте MCP Server:
   - URL: `https://your-domain.railway.app`
   - Transport: SSE
4. Claude автоматично пройде OAuth авторизацію

## Архітектура

```
┌─────────────────────────────────────────────────────────────┐
│                      Claude Web                              │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │  SSE     │    │  OAuth   │    │  Tools   │             │
│  │ Connect  │───▶│  Auth    │───▶│  Call    │             │
│  └──────────┘    └──────────┘    └──────────┘             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│               MCP HTTP Server (Starlette)                    │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │  /mcp/   │    │  OAuth   │    │  Tool    │             │
│  │   sse    │    │  2.0     │    │ Executor │             │
│  └──────────┘    └──────────┘    └──────────┘             │
│        │               │               │                    │
│        ▼               ▼               ▼                    │
│  ┌─────────────────────────────────────────┐               │
│  │           Memory Persistence             │               │
│  └─────────────────────────────────────────┘               │
│        │               │                                    │
│        ▼               ▼                                    │
│  ┌──────────┐    ┌──────────┐                             │
│  │PostgreSQL│    │  Redis   │                             │
│  │ (memory) │    │ (cache)  │                             │
│  └──────────┘    └──────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

## Змінні оточення

| Змінна | Опис | Default |
|--------|------|---------|
| `PORT` | Порт сервера | 8090 |
| `SERVER_URL` | URL сервера | http://localhost:8090 |
| `DATABASE_URL` | PostgreSQL URL | (in-memory) |
| `REDIS_URL` | Redis URL | (in-memory) |

## Тестування

```bash
# Health check
curl http://localhost:8090/health

# Discovery
curl http://localhost:8090/.well-known/mcp.json

# Tools list
curl http://localhost:8090/api/tools

# MCP initialize
curl -X POST http://localhost:8090/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize"}'

# Tools list via MCP
curl -X POST http://localhost:8090/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'

# Call tool
curl -X POST http://localhost:8090/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"store_memory","arguments":{"content":"Test fact","type":"fact"}}}'
```
