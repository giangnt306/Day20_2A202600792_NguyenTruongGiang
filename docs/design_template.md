# Design Document — Multi-Agent Research System

## Problem

Build a research assistant that takes a long-form question (e.g. *"What is GraphRAG and
how does it improve retrieval-augmented generation?"*) and returns a well-structured,
**cited** ~500-word answer for a technical audience. The system must: find sources on the
open web, analyze them critically, write a grounded answer, and report latency / cost /
quality so we can compare a single-agent baseline against a multi-agent workflow.

## Why multi-agent?

A single agent doing search + analysis + writing in one prompt works, but it conflates
three concerns and gives us no place to insert checks. Splitting into roles buys us:

- **Separation of concerns** — each agent has one prompt, one job, one temperature
  (researcher 0.2 for grounded recall, analyst 0.1 for precision, writer 0.4 for fluency).
- **Inspectable handoffs** — `research_notes → analysis_notes → final_answer` are stored
  in shared state, so we can trace exactly where quality is lost.
- **Targeted guardrails** — a dedicated critic checks citation coverage *after* writing,
  which is awkward to bolt onto a one-shot baseline.

Honest caveat (see [benchmark report](../reports/benchmark_report.md)): on **simple**
queries the multi-agent path costs ~2× the tokens and ~1.6× the latency for *equal* quality.
Multi-agent earns its cost on **multi-faceted** questions where a single context window
can't hold research + analysis + drafting without degrading. The design below is built so
the supervisor can short-circuit cheap queries.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode → mitigation |
|---|---|---|---|---|
| **Supervisor** | Pure router: pick next worker or stop | `ResearchState` (which fields are filled) | next route label + `route_history` entry | infinite loop → `max_iterations` guardrail forces `done` |
| **Researcher** | Search web, write cited notes grounded only in sources | `query`, `max_sources` | `sources[]`, `research_notes` | search outage → graceful fallback to mock sources; hallucination → "cite only [n]" prompt |
| **Analyst** | Extract claims, compare viewpoints, flag weak evidence | `research_notes` | `analysis_notes` | over-trusting one source → prompt asks for contradictions + evidence gaps |
| **Writer** | Synthesize final cited answer for the audience | research + analysis notes, source map | `final_answer` | uncited claims → critic catches low coverage |
| **Critic** *(optional)* | Verify citation coverage, flag unsupported claims | `final_answer`, `sources` | findings + `errors[]` entry | low coverage (<50%) recorded as an error for the benchmark |

## Shared state

Defined in [`core/state.py`](../src/multi_agent_research_lab/core/state.py) as a Pydantic
`ResearchState` — single source of truth passed through the graph.

| Field | Why it exists |
|---|---|
| `request: ResearchQuery` | Immutable inputs (query, max_sources, audience); validated (`query` min length 5, `max_sources` 1–20). |
| `iteration`, `route_history` | Drive the supervisor's stop condition and make the run replayable. |
| `sources[]` | Evidence shared between researcher, writer, and critic so citations stay consistent. |
| `research_notes`, `analysis_notes`, `final_answer` | The three handoff artifacts; their *presence* is what the supervisor routes on. |
| `agent_results[]` | Per-agent content + `{input_tokens, output_tokens, cost_usd}` metadata → feeds cost/token aggregation. |
| `trace[]` | Lightweight event log mirrored from every node for debugging / screenshots. |
| `errors[]` | Non-fatal problems (e.g. low citation coverage) surfaced as a benchmark metric. |

Derived helpers: `total_cost_usd` and `total_tokens` sum the per-agent metadata so the
benchmark never re-counts tokens by hand.

## Routing policy

The supervisor is the **only** router; every worker returns to it
([`agents/supervisor.py`](../src/multi_agent_research_lab/agents/supervisor.py),
[`graph/workflow.py`](../src/multi_agent_research_lab/graph/workflow.py)).

```text
            ┌───────────────► END (done)
            │
   ┌────────┴────────┐
   │   Supervisor    │◄────────────┐
   └────────┬────────┘             │
   no notes │ have notes,          │
            │ no analysis ─┐       │
            ▼              ▼        │
        Researcher      Analyst     │
            │              │        │
            └──────────────┴────────┤  have analysis, no answer → Writer
                                    │            │
                                    └────────────┘
   After Writer fills final_answer → Supervisor → done → (optional) Critic
```

Decision rule (`SupervisorAgent.decide`), evaluated top-down:

1. `iteration >= max_iterations` → **done** (hard guardrail, checked first).
2. no `research_notes` → **researcher**
3. `research_notes` but no `analysis_notes` → **analyst**
4. `analysis_notes` but no `final_answer` → **writer**
5. otherwise → **done**

It's a deterministic state-completion policy — cheap, debuggable, and impossible to loop
forever. A more advanced version could let the supervisor *re-dispatch* the researcher when
the critic reports low coverage; the state (`errors[]`) already carries the signal needed.

## Guardrails

- **Max iterations:** `MAX_ITERATIONS=6` (config), enforced as the supervisor's first
  check **and** as LangGraph's `recursion_limit = max_iterations*2 + 4`.
- **Timeout:** `TIMEOUT_SECONDS=60` passed to the OpenAI client per request.
- **Retry:** `tenacity` — 3 attempts, exponential backoff (1→8s) on the LLM call, centralized
  in `LLMClient` so agents stay clean.
- **Fallback:** no `OPENAI_API_KEY` → deterministic mock LLM; no `TAVILY_API_KEY` or a
  Tavily error → mock search. The pipeline (and the whole test suite) runs offline.
- **Validation:** Pydantic schemas validate all inputs/outputs (`ResearchQuery`,
  `AgentResult`, `BenchmarkMetrics`); the critic validates citation coverage and pushes a
  benchmark-visible error when it drops below 50%.

## Benchmark plan

Queries (from [`configs/lab_default.yaml`](../configs/lab_default.yaml)):

1. "Research GraphRAG state-of-the-art and write a 500-word summary"
2. "Compare single-agent and multi-agent workflows for customer support"
3. "Summarize production guardrails for LLM agents"

| Metric | How measured | Expected outcome |
|---|---|---|
| Latency | wall-clock around the runner | multi-agent slower (more sequential LLM calls) |
| Cost (USD) | summed per-call token cost via price table | multi-agent ~2× baseline |
| Tokens | summed `input+output` across agents | multi-agent higher |
| Quality (0–10) | LLM-as-judge (`judge_quality`) | comparable on simple Qs; multi-agent pulls ahead on complex Qs |
| Citation coverage | `cited [n] / available sources` | both high; writer prompt enforces a Sources list |
| Failure rate | `len(state.errors)` / runs | ~0 in normal operation |

Run it with:

```bash
malab benchmark -q "Summarize production guardrails for LLM agents"
```

Output is written to [`reports/benchmark_report.md`](../reports/benchmark_report.md).
See [`failure_modes.md`](../reports/failure_modes.md) for the required failure-mode analysis.
