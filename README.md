# Multi-Agent LLM Orchestration System

A containerized, production-grade multi-agent pipeline with a self-improving evaluation loop, dynamic tool orchestration, and adversarial robustness testing.

---

## Setup & Run

### 1 · Clone the repo

```bash
git clone https://github.com/shrutisingh004/multi-agent-llm-orchestration-system.git
cd multi-agent-system
```

### 2. Set your Gemini API key (free at https://aistudio.google.com/app/apikey)
```bash
cp .env.example .env
# Then open .env and put your real Gemini API key
```

### 3. Start everything
```bash
docker compose up --build
```

### 4. API is live at `http://localhost:8000`
### 5. Log viewer (pgAdmin) at http://localhost:5050 (admin@admin.com / admin)
```

No manual steps after `docker compose up`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI (port 8000)                   │
│  POST /query  GET /trace/:id  GET /eval/latest              │
│  POST /rewrite/approve        POST /eval/rerun              │
└──────────────────────┬──────────────────────────────────────┘
                       │ SSE stream / REST
┌──────────────────────▼──────────────────────────────────────┐
│                     Orchestrator                             │
│  - Calls Gemini to build a routing plan (not hardcoded)     │
│  - Mediates ALL inter-agent communication via SharedContext  │
│  - Checks context budget before each agent invocation       │
│  - Logs all routing decisions with justification            │
└───┬──────────┬───────────┬──────────────┬───────────────────┘
    │          │           │              │
    ▼          ▼           ▼              ▼
┌───────┐ ┌────────┐ ┌─────────┐ ┌──────────┐
│Decomp │ │Retriev.│ │Critique │ │Synthesis │
│Agent  │ │Agent   │ │Agent    │ │Agent     │
└───┬───┘ └────┬───┘ └────┬────┘ └────┬─────┘
    │          │           │           │
    └──────────┴───────────┴───────────┘
              SharedContext (schema-defined)
              Agents NEVER call each other directly

Tools (called by agents via orchestrator):
  ┌─────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
  │ web_search  │ │code_execution│ │ data_lookup  │ │self_reflection│
  │ (stub)      │ │ (sandboxed)  │ │ (NL→SQL stub)│ │(contradiction)│
  └─────────────┘ └──────────────┘ └──────────────┘ └──────────────┘

Persistence:
  PostgreSQL ← all events, tool calls, eval runs, rewrites
  Redis      ← async job queue for background worker

Self-Improving Loop:
  eval run → meta-agent proposes prompt rewrite → human reviews
  → if approved, re-eval on failed cases → delta logged
```

---

## Agents

### Orchestrator
**Decision boundary**: Calls Gemini to build a routing plan JSON at runtime. Does NOT use a hardcoded sequence — it can skip agents for simple queries. Mediates all handoffs via SharedContext. Triggers CompressionAgent when any agent's remaining budget drops below 500 tokens.

### Decomposition Agent
**Decision boundary**: Breaks ambiguous queries into typed sub-tasks with dependency graphs. Emits `SubTask` objects into SharedContext. Dependent sub-tasks are not executed until their dependencies resolve. Falls back to a single generic task if LLM returns no structure.

### Retrieval Agent
**Decision boundary**: Uses the `web_search` tool to fetch at least 2 chunks, then performs multi-hop reasoning across them. Single-hop answers are architecturally prevented — the agent is prompted to cite at minimum 2 chunks. Citations map to specific `chunk_id`s stored in SharedContext.

### Critique Agent
**Decision boundary**: Reviews all prior agent outputs claim-by-claim (not the whole output). Assigns a `confidence` float and a `flagged` boolean per claim. Only flags specific text spans it disagrees with. Does NOT produce a summary judgment.

### Synthesis Agent
**Decision boundary**: Merges all outputs, resolves contradictions flagged by the Critique Agent, and produces a final answer with a provenance map. First calls the `self_reflection` tool to check for internal contradictions in session history before synthesizing.

### Compression Agent
**Decision boundary**: Invoked automatically when context budget is low. Compresses only conversational filler (lossy); preserves all structured data (tool outputs, scores, JSON) verbatim.

### Meta-Agent
**Decision boundary**: After every eval run, reads failure cases, identifies the worst-performing dimension, and proposes a prompt rewrite for the responsible agent. Does NOT auto-apply — requires human approval via the `/rewrite/approve` endpoint.

---

## Tools

All tools have a defined failure contract: `{"data": ..., "error": "timeout|empty|malformed_input|...", "latency_ms": ...}`

| Tool | Description | Failure modes |
|------|-------------|---------------|
| `web_search` | Returns structured results with URLs and relevance scores | timeout, empty, malformed_input |
| `code_execution` | Runs Python in a restricted sandbox, returns stdout/stderr/exit_code | timeout, malformed_input, runtime_error |
| `structured_data_lookup` | NL → query against in-memory stub DB | malformed_input, empty |
| `self_reflection` | Scans session history for contradictions | malformed_input, empty |

Tool calls are logged with: input, output, latency_ms, attempt number, accepted boolean. Retry logic is implemented in code (not in prompts), up to 2 retries per call.

---

## Evaluation

15 test cases across 3 categories:

- **Straightforward (5)**: Known correct answers; baseline scoring
- **Ambiguous (5)**: Underspecified inputs; tests decomposition quality
- **Adversarial (5)**: Prompt injections, false-premise queries; tests robustness

Scoring dimensions (all custom, no third-party eval framework):

| Dimension | What it measures |
|-----------|-----------------|
| `correctness` | Keyword match against expected answers |
| `citation_accuracy` | Retrieval agent cited valid chunk IDs |
| `contradiction_resolution` | Synthesis resolved flagged claims |
| `tool_efficiency` | Penalizes unnecessary/rejected tool calls |
| `budget_compliance` | Context budget policy violations |
| `critique_agreement` | How often critique agrees with final output |
| `adversarial_resistance` | (adversarial only) Injection blocked / false premise corrected |

Every eval run stores: exact prompts, exact tool calls, exact outputs, scores + justifications, timestamp. Re-running on the same inputs produces a diff-able output.

---

## Self-Improving Loop

1. `POST /eval/rerun` → runs all 15 test cases
2. Meta-agent identifies worst-performing dimension
3. Meta-agent proposes a prompt rewrite with structured diff + justification
4. Rewrite stored as `pending` in `prompt_rewrites` table
5. Human reviews via `POST /rewrite/approve` (`{"rewrite_id": 1, "decision": "approve"}`
6. If approved: prompt updated in memory; `POST /eval/rerun` with `failed_ids` runs only previously-failed cases
7. Delta in scores stored in `prompt_rewrites.delta`

Everything is auditable: every proposal, decision, and performance delta has a timestamp and is queryable.

---

## API Reference

### `POST /query`
Submit a query. Returns an SSE stream with real-time agent activity.

```bash
curl -N -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What causes inflation?"}'
```

SSE events include: `routing`, `agent_start`, `agent_done`, `policy_violations`, `done`, `complete`.

### `GET /trace/{job_id}`
Full execution trace: agent events, tool calls, routing decisions, in order.

### `GET /eval/latest`
Latest eval run summary: by category, by dimension, pending rewrites.

### `POST /rewrite/approve`
```json
{"rewrite_id": 1, "decision": "approve"}
```

### `POST /eval/rerun`
```json
{"failed_ids": ["s1", "adv2"]}   // empty array = run all 15
```

---

## Known Limitations

1. **Web search is a stub**: Returns fake results. In production, replace `app/tools/tools.py::web_search` with a real search API (Brave, SerpAPI, etc.).

2. **Code execution sandbox is limited**: The restricted `exec()` approach is basic. Production would use Docker-in-Docker or a proper sandbox like Firecracker.

3. **Retrieval is not vector-based**: No embedding model or vector DB. Retrieval relies on the web_search stub. Real RAG would use FAISS or pgvector.

4. **Meta-agent prompt rewrite quality depends on Gemini**: The proposed rewrites may not always be improvements. The human approval gate is essential.

5. **Token counting is approximate**: Uses `len(text) / 4` heuristic instead of tiktoken (to avoid model-specific tokenizers). Actual token usage may differ by ±20%.

6. **No authentication**: API endpoints are open. Production would require API keys or OAuth.

7. **Self-improving loop is single-session**: Applied prompt rewrites live in memory. Restarting the API resets to default prompts (rewrites are in DB but not auto-loaded on boot).

---

## What I'd Build Next

- Real vector retrieval (pgvector + embeddings)
- Authentic web search integration
- Persistent prompt loading from DB on startup
- Agent-level caching (avoid re-running identical sub-tasks)
- A simple dashboard UI for eval results and rewrite review
- Proper async job queue with status polling fallback

---

## AI Collaboration Disclosure

This project was built with AI assistance (Claude). The AI was used for:
- Boilerplate scaffolding (DB models, FastAPI structure)
- Drafting agent prompt templates
- Code review suggestions

All architectural decisions, agent design, failure-mode contracts, and scoring logic were designed and validated manually.
