# AI Engineering: Independent Learning Paths & Stitched Ecosystem Stacks

This document provides a dual-dimensional mastery architecture for modern AI engineering:
1. **Vertical Isolation:** Five independent, leveled learning tracks (Foundations $\rightarrow$ Intermediate $\rightarrow$ Advanced $\rightarrow$ Frontier).
2. **Horizontal Synthesis:** Four "Stitched Ecosystem Stacks" that integrate each pillar into one functional system.

**Key principle:** the stacks are *not* four separate projects. They are **one evolving system**, built once as Stack 1, where each new level swaps in an upgraded component. Every upgrade is measured against the same golden evaluation set (see `eval/`).

---

## Part 0: Constraints & Target Role

### Hardware profile
| Resource | Reality | Consequence |
|---|---|---|
| CPU | i7 (2018), 4 threads | All local inference on CPU via Ollama / llama.cpp / `transformers` |
| RAM | 12 GB | `qwen2.5:3b` (Q4, ~2 GB) as the daily model; 7B only for one-off tests. Never run 7B + Docker stack together. |
| GPU | 2 GB VRAM | Treated as unusable for LLMs |
| Training GPU | Kaggle (~30 GPU-h/week free) or Colab T4 | All LoRA / QLoRA / DPO work |
| API budget | A few dollars (e.g. Claude Haiku 4.5) | Synthetic data generation, LLM-as-judge, "frontier tier" in model routing |
| Time | 15 h/week | ~10 h build · ~3 h measure + write ADR · ~2 h reading for next level |

### Target role: AI Architect / Solution Architect / AI Consultant
The goal is **decision-making fluency**, not research-grade reimplementation. Each level is tagged:
* **[Deep]**: build properly; this is what clients build and get stuck on.
* **[Working]**: build a minimal working version once to know the mechanism, cost, and failure modes; then move on.

### Definition of done for every level
1. **Code**: built from scratch first, then compared against the standard library/tool (e.g. own BM25 vs `rank_bm25`). The comparison is the correctness check.
2. **Measured table**: quality / latency / cost / RAM, before vs after, from the golden eval harness.
3. **ADR** (`notes/adr/`): problem, options, decision, consequences.
4. **Client note**: one paragraph: "when I'd recommend this to a client, and when I wouldn't."

---

## Part 1: The 5 Independent Learning Paths

```
    [PATH 1: RETRIEVAL & CONTEXT]      [PATH 2: RUNTIMES & INFERENCE]     [PATH 3: DATA & ALIGNMENT]
    Level 1: Semantic Chunker          Level 1: Tokenizer & Logit Stream  Level 1: Evol-Instruct Data Gen
    Level 2: Hybrid Search & RRF       Level 2: KV Cache & VRAM Sizing    Level 2: LoRA / QLoRA
    Level 3: Cross-Encoder Reranking   Level 3: Speculative Decoding      Level 3: DPO / Preference
    Level 4: GraphRAG & Triples        Level 4: Continuous Batching       Level 4: Model Merging (SLERP)

    [PATH 4: EVAL & OBSERVABILITY]     [PATH 5: PROTOCOLS & TOOLING]
    Level 1: Schema / Contract Tests   Level 1: JSON-RPC & Stdio Transport
    Level 2: RAG Triad Evaluator       Level 2: Dynamic Tool Discovery
    Level 3: Semantic Guardrails       Level 3: Streamable HTTP Transport
    Level 4: OpenTelemetry Tracing     Level 4: Enterprise MCP Gateway
```

---

### Path 1: Context, Indexing & Retrieval Engineering (RAG): **[Deep] all levels**

* **Level 1 (Foundations): AST & Semantic Chunking**
  * *Objective:* Eliminate naive token slicing. Parse text, markdown, and code into trees (ASTs) preserving logical boundaries (headers, classes, functions, markdown tables).
  * *Deliverable:* A recursive Python chunker outputting chunks with structural metadata (`breadcrumbs`, `parent_section`, `token_count`).
  * *Runs on:* CPU. Python `ast` for code; markdown via header parsing.
* **Level 2 (Intermediate): Hybrid Search & Reciprocal Rank Fusion (RRF)**
  * *Objective:* Combine dense semantic embeddings (cosine similarity) with sparse lexical matching (BM25).
  * *Deliverable:* A fusion engine calculating:
    $$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)} \quad (k = 60 \text{ by convention})$$
    Show on the golden set that queries with exact part numbers/acronyms succeed where pure vector search fails (recall@5, before vs after).
  * *Runs on:* CPU. Embeddings: `bge-small-en-v1.5` or `nomic-embed-text` (Ollama).
* **Level 3 (Advanced): Cross-Encoder Re-Ranking**
  * *Objective:* Solve the bi-encoder bottleneck. Retrieve a broad $N=50$ candidate set fast, then score query–document pairs jointly with a cross-encoder.
  * *Deliverable:* A two-stage pipeline (50 → 5) with a measured quality-vs-latency table.
  * *Runs on:* CPU. `cross-encoder/ms-marco-MiniLM-L-6-v2`.
* **Level 4 (Frontier): Multi-Hop GraphRAG**
  * *Objective:* Extract `(Entity)-[Relationship]->(Entity)` triples into a knowledge graph and traverse it for multi-hop questions.
  * *Deliverable:* A multi-hop query engine answering questions that span disparate document fragments.
  * *Runs on:* CPU, **NetworkX** (not Neo4j: RAM). Triple extraction via API or local 3B.
  * *Optional:* ColPali visual retrieval for table/image-heavy docs, Colab only.

---

### Path 2: Local Runtimes, Inference & Quantization Mechanics

* **Level 1 (Foundations): Tokenization, Logits & Temperature Mechanics**: **[Working]**
  * *Objective:* Demystify auto-regressive generation. Tokenize text to IDs, inspect logits from local weights, apply Top-$K$/Top-$P$ filtering and temperature, and sample token-by-token.
  * *Deliverable:* A script generating text from raw output distributions, compared against `model.generate()`.
  * *Runs on:* CPU, `transformers` + `Qwen2.5-0.5B-Instruct`.
* **Level 2 (Intermediate): KV Cache & Deployment Sizing Calculator**: **[Deep]** *(core architect skill: "what hardware for N users?")*
  * *Objective:* Understand why inference is memory-bandwidth bound and size deployments. Total memory = weights + KV cache + overhead, where:
    $$\text{KV Cache (bytes)} = 2 \times b \times s \times l \times h_{kv} \times d \times p$$
    *(2 = keys and values; $b$=concurrent sequences, $s$=sequence length in tokens, $l$=layers, $h_{kv}$=**KV heads** (smaller than attention heads under GQA, e.g. Llama-3-8B: 32 attention heads, 8 KV heads), $d$=head dimension, $p$=bytes per element (2 for FP16, 1 for FP8 KV cache))*.
  * *Deliverable:* A sizing calculator reading `config.json` from Hugging Face, covering FP16 / Q8 / Q4 (GGUF) / AWQ weights, context length, and concurrency, predicting OOM points and recommending a GPU class. Validate predictions against real measurements on small models.
  * *Runs on:* CPU (pure math + validation on 0.5B/3B).
* **Level 3 (Advanced): Speculative Decoding**: **[Working]**
  * *Objective:* Accelerate inference without changing output distribution. A small draft model proposes $k$ tokens; the target model verifies them in one forward pass. **Draft and target must share a tokenizer.**
  * *Deliverable:* A harness measuring acceptance rate and tokens/sec vs $k$. Speedup is *measured*, not assumed; on CPU it may be small or negative, and explaining why is the lesson.
  * *Runs on:* CPU, `Qwen2.5-0.5B` (draft) + `Qwen2.5-1.5B` (target) via `transformers` assisted generation or llama.cpp `--model-draft`.
* **Level 4 (Frontier): Continuous Batching & Paged Attention**: **[Working]**
  * *Objective:* Understand production serving (vLLM / SGLang / TensorRT-LLM): dynamic insertion/eviction of requests per step, paged KV blocks.
  * *Deliverable:* An async simulator comparing static vs continuous batching throughput and GPU-idle time under variable-length workloads.
  * *Runs on:* CPU (simulation).

---

### Path 3: Synthetic Data, Alignment & Specialization: **[Working]**, separate track (Kaggle/Colab)

* **Level 1 (Foundations): Evol-Instruct Data Generation**
  * *Objective:* Create fine-tuning datasets from seed prompts via iterative mutation (In-Depth: add constraints, deepen reasoning; In-Breadth: diverse domain variants).
  * *Deliverable:* A pipeline generating 500 validated instruction–response pairs in ChatML, deduplicated and schema-checked.
  * *Runs on:* API (cheap model) for generation, CPU for validation.
* **Level 2 (Intermediate): Parameter-Efficient Fine-Tuning (LoRA / QLoRA)**
  * *Objective:* Freeze base weights and train low-rank matrices ($W_0 + \Delta W = W_0 + BA$).
  * *Deliverable:* An Unsloth/PEFT notebook producing an adapter for one narrow task (e.g. strict JSON output), exported to GGUF and served back home in Ollama. Measure base vs adapter on a task-specific eval.
  * *Runs on:* Kaggle/Colab T4, `Qwen2.5-0.5B` or `1.5B`.
* **Level 3 (Advanced): Direct Preference Optimization (DPO)**
  * *Objective:* Align outputs with preferences without a separate reward model (PPO).
  * *Deliverable:* Sample multiple completions, score against a deterministic rubric, compile `{"prompt", "chosen", "rejected"}`, run DPO (TRL), measure before/after.
  * *Runs on:* Kaggle/Colab.
* **Level 4 (Frontier, optional): Model Merging (SLERP & TIES)**
  * *Objective:* Combine specialized checkpoints of the **same architecture** without retraining.
  * *Deliverable:* `mergekit` SLERP merge of `Qwen2.5-Coder-0.5B-Instruct` + `Qwen2.5-0.5B-Instruct`; evaluate both skills on the merge.
  * *Runs on:* CPU (mergekit does not need a GPU at this size).

*Architect takeaway for Path 3:* a decision framework for **prompting vs RAG vs fine-tuning vs preference tuning**, with cost and data requirements for each.

---

### Path 4: Evaluation, Guardrails & LLMOps Observability: **[Deep] all levels**

* **Level 1 (Foundations): Deterministic Schema & Unit Testing**
  * *Objective:* Treat LLM outputs like unit-tested code. Validate responses against strict Pydantic models with automated retries.
  * *Deliverable:* A test suite asserting field existence, type consistency, and range constraints; retry/repair rate measured.
* **Level 2 (Intermediate): The RAG Triad Evaluator**
  * *Objective:* Score retrieval and generation on three independent dimensions:
    1. *Context Relevance:* Does the retrieved context match the query?
    2. *Groundedness / Faithfulness:* Is every assertion in the response supported by the context?
    3. *Answer Relevance:* Does the response address the question?
  * *Deliverable:* An automated 0.0–1.0 scorer per query, plugged into the golden eval harness. Cross-check your judge against Ragas on a sample.
  * *Runs on:* API judge (local 3B judges are too noisy; measuring that noise is a worthwhile side exercise).
* **Level 3 (Advanced): Semantic Guardrails & Red-Teaming**
  * *Objective:* Guard against prompt injection, jailbreaks, and PII leakage.
  * *Deliverable:* An inline proxy checking embedding similarity against known attack clusters, plus regex/entity masking on outputs; a red-team set with measured block rate and false-positive rate.
* **Level 4 (Frontier): OpenTelemetry Distributed Tracing**
  * *Objective:* Production visibility across multi-step, multi-agent workflows.
  * *Deliverable:* Agents instrumented with OTel spans following the GenAI semantic conventions: token usage, latency p50/p95/p99, tool failure rates, routing decisions. View in Arize Phoenix or Jaeger.

---

### Path 5: Protocols & System Interoperability (MCP & Tooling): **[Deep] all levels**

* **Level 1 (Foundations): JSON-RPC & Stdio Tool Mechanics**
  * *Objective:* Understand the wire protocol behind agent extensions.
  * *Deliverable:* A raw Python server reading JSON-RPC 2.0 from `stdin` and writing responses to `stdout` (`server/discover`, `tools/list`, `tools/call`, plus the legacy `initialize` handshake for pre-2026-07-28 clients), then the same server rewritten with the official MCP Python SDK. *(MCP 2026-07-28 made the protocol stateless: no handshake; every request carries its protocol version in `_meta`.)*
* **Level 2 (Intermediate): Dynamic MCP Discovery Agent**
  * *Objective:* Decouple agents from hardcoded functions. Query `tools/list`, convert JSON schemas to the model's tool format, call dynamically.
  * *Deliverable:* An agent that drives an MCP server it has never seen (official filesystem server, or your own SQLite server) without code changes.
* **Level 3 (Advanced): Remote Transport (Streamable HTTP)**
  * *Objective:* Host MCP tools as network services. *(The older HTTP+SSE transport was deprecated in the March 2025 MCP spec; Streamable HTTP replaces it.)*
  * *Deliverable:* A Streamable HTTP MCP server + client with authentication, serving many concurrent agents statelessly (2026-07-28 removed protocol sessions and `Mcp-Session-Id`).
* **Level 4 (Frontier): Enterprise MCP Gateway & Circuit Breaker**
  * *Objective:* Production tool governance.
  * *Deliverable:* A central gateway with authN/authZ per tool, rate limiting, audit logging, and circuit breakers (trip on consecutive failures, half-open recovery).

---

## Part 2: The Stitched Ecosystem Stacks

One codebase, upgraded stage by stage. Each stack is the previous one with components swapped.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               ONE SYSTEM, FOUR STAGES OF MATURITY                                        │
├───────────────┬────────────────────────┬────────────────────────┬────────────────────────┬───────────────┤
│               │ STACK 1: Minimalist    │ STACK 2: Observable    │ STACK 3: Hardened      │ STACK 4: Swarm│
├───────────────┼────────────────────────┼────────────────────────┼────────────────────────┼───────────────┤
│ Retrieval     │ AST chunks + BM25      │ Hybrid search (RRF)    │ + Cross-encoder rerank │ + GraphRAG    │
│ Inference     │ Ollama qwen2.5:3b      │ + sizing calculator    │ + speculative decoding │ + batching sim│
│ Model         │ base                   │ base → QLoRA adapter*  │ → DPO-aligned model*   │ + routing tier│
│ Quality/Safety│ Pydantic schema guard  │ RAG triad evaluator    │ Semantic guardrails    │ OTel tracing  │
│ Tools         │ Stdio JSON-RPC         │ Dynamic MCP discovery  │ Streamable HTTP MCP    │ MCP gateway   │
└───────────────┴────────────────────────┴────────────────────────┴────────────────────────┴───────────────┘
  * swapped in after the Path 3 phase (weeks 13–15); stacks run on the base model until then.
```

---

### Stack 1: The Minimalist Functional System
* **Goal:** A small, dependable agent answering questions over local documents with guaranteed schema compliance.
* **Topology:**
  1. **Ingest:** AST chunker reads markdown/code, preserving header hierarchies; BM25 retrieval.
  2. **Serving:** Ollama OpenAI-compatible endpoint, `qwen2.5:3b`.
  3. **Execution:** ReAct loop calling a local stdio JSON-RPC calculator and document reader.
  4. **Validation:** Output validated against a Pydantic schema, with retry.
* **Measured:** golden-set answer accuracy and retrieval hit@5 vs the no-retrieval baseline.
* **Use Case:** Deterministic CLI utilities, structured extractors, local document analyzers.

### Stack 2: The Observable Hybrid Prototype
* **Goal:** Finds exact identifiers *and* conceptual matches, with quality measured on every query.
* **Topology:**
  1. **Retrieval:** BM25 + vector search fused with RRF.
  2. **Inference:** Deployment sized with the Path 2 calculator (and a written sizing recommendation for a hypothetical 50-user client).
  3. **Specialization:** Base model; later swapped for the QLoRA adapter.
  4. **Interoperability:** Dynamic discovery against an MCP server.
  5. **Validation:** RAG triad scores per query.
* **Use Case:** Internal knowledge assistants, database Q&A.

### Stack 3: The Production-Hardened Engine
* **Goal:** High-accuracy retrieval behind safety controls, with tools served over the network.
* **Topology:**
  1. **Retrieval:** $N=50$ candidates → cross-encoder → top 5.
  2. **Inference:** Speculative decoding where measurements justify it.
  3. **Alignment:** DPO-aligned model (after Path 3) penalizing hallucination and verbosity.
  4. **Governance:** Inbound guardrail proxy; tools over Streamable HTTP MCP.
* **Use Case:** Customer-facing search, technical support agents with live APIs.

### Stack 4: The Autonomous Frontier Swarm
* **Goal:** Multi-hop reasoning, multi-agent delegation, model tiering, and full telemetry.
* **Topology:**
  1. **Retrieval:** GraphRAG alongside hybrid + rerank.
  2. **Coordination:** Supervisor pattern with model tiering: local 3B for filtering/classification, API frontier model for orchestration.
  3. **Serving insight:** Continuous batching simulator informs the production serving recommendation (vLLM/SGLang) in the ADR.
  4. **Infrastructure:** All spans to an OTel collector; tools gated by the MCP gateway with circuit breakers.
* **Use Case:** Research/synthesis agents, software engineering agents, incident response.

---

## Part 3: Learning & Implementation Roadmap (15 h/week, ~19 weeks)

Build vertically, stitch horizontally, measure everything against the same golden set.

```
[ Week 1: Setup & Measurement Baseline ]
  ├── uv project, Ollama models, choose your corpus (docs you actually care about)
  ├── Write 40-question golden set (include exact-identifier/acronym questions)
  └── Run no-retrieval baseline through eval harness → first number on the board

[ Weeks 2–4: Foundational Literacy ]
  ├── Path 1 L1 (AST chunking)          ├── Path 2 L1 (logits & sampling)
  ├── Path 4 L1 (Pydantic guard)        └── Path 5 L1 (stdio JSON-RPC)
  └── STITCH: Stack 1 → beat the baseline

[ Weeks 5–8: Retrieval & Sizing ]
  ├── Path 1 L2 (hybrid + RRF)          ├── Path 2 L2 (sizing calculator)
  ├── Path 4 L2 (RAG triad)             └── Path 5 L2 (dynamic MCP discovery)
  └── STITCH: Stack 2 (base model)

[ Weeks 9–12: Latency, Safety & Network Tools ]
  ├── Path 1 L3 (cross-encoder)         ├── Path 2 L3 (speculative decoding)
  ├── Path 4 L3 (guardrails)            └── Path 5 L3 (Streamable HTTP)
  └── STITCH: Stack 3 (base model)

[ Weeks 13–15: Specialization Track (Kaggle/Colab) ]
  ├── Path 3 L1 (Evol-Instruct) → L2 (QLoRA) → L3 (DPO)
  └── SWAP: adapter into Stack 2, DPO model into Stack 3; measure the delta
       → ADR: "fine-tune vs RAG vs prompt" decision framework

[ Weeks 16–19: Swarms, Graphs & Telemetry ]
  ├── Path 1 L4 (GraphRAG)              ├── Path 2 L4 (batching simulator)
  ├── Path 4 L4 (OpenTelemetry)         └── Path 5 L4 (MCP gateway)
  └── STITCH: Stack 4

[ Optional ]
  └── Path 3 L4 (mergekit SLERP, CPU) · ColPali (Colab)
```

### Portfolio output by week 19
* ~20 ADRs + client notes: a ready-made consulting playbook.
* A deployment sizing calculator.
* One system with a measured improvement curve from baseline → Stack 4.
