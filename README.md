# Claims Coverage Copilot — advanced-RAG starter (insurance)

A small, runnable advanced-RAG pipeline for insurance claims, driven by prompts.
An adjuster pastes a claim (FNOL); the system plans retrieval, runs **hybrid
search**, **reranks**, pulls **real-time signals**, drafts a coverage
determination **with citations**, runs a **faithfulness gate** that abstains
when evidence is missing, and outputs a **confidence score (0–100)** with an
ASCII bar visualization. It is a *copilot*: a human reviews every draft.

## Repo layout
```
claims-copilot/
├── claims_copilot.py     # the whole pipeline (start here)
├── requirements.txt
└── README.md
# Grow into this as you scale:
#   data/        policies, endorsements, claims, guidelines (synthetic only)
#   retrievers/  hybrid + an EmbeddingRetriever (Voyage / OpenAI / local)
#   prompts/     the 4 prompt templates, version-controlled
#   tools/       real-time integrations (weather/CAT, sanctions)
#   eval/        RAGAS gold set + faithfulness/relevance scores
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

**Offline (no key) — proves the flow end to end:**
```bash
python claims_copilot.py --dry-run
```

**Live Anthropic API.** Set your key as an environment variable — never paste it
into the code or share it in chat:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."     # macOS / Linux
setx   ANTHROPIC_API_KEY "sk-ant-..."     # Windows (open a new shell after)

python claims_copilot.py                                  # default sample claim
python claims_copilot.py --claim "FNOL: ... your claim ..."
python claims_copilot.py --model claude-opus-4-8          # heavier reasoning
python claims_copilot.py --embedder hashing               # + dense semantic retrieval (no key needed)
```
The SDK reads `ANTHROPIC_API_KEY` automatically. Models are configurable via
`COPILOT_MODEL` and `COPILOT_RERANK_MODEL`.

### Dense (semantic) retrieval — `--embedder`

`HybridRetriever` always fuses BM25 + TF-IDF via Reciprocal Rank Fusion. Pass
`--embedder` to fuse in a third, *semantic* ranked list from dense embeddings:

| `--embedder` | Backend | Needs |
|---|---|---|
| `none` (default) | lexical only | nothing |
| `hashing` | deterministic hashing-trick vectors | nothing (offline-safe stand-in) |
| `voyage` | Voyage AI | `pip install voyageai`, `VOYAGE_API_KEY` |
| `openai` | OpenAI embeddings | `pip install openai`, `OPENAI_API_KEY` |
| `st` | local sentence-transformers model | `pip install sentence-transformers` |

Also configurable via `COPILOT_EMBEDDER`.

## Prompt evaluation harness

`eval_prompts.py` runs a small gold set of claims through the pipeline and
scores decision accuracy, retrieval recall, citation grounding, and abstain
correctness — and can A/B compare prompt variants.
```bash
python eval_prompts.py                    # offline, proves the harness + scorers work
python eval_prompts.py --live              # real API — where prompt edits actually move the numbers
python eval_prompts.py --live --compare    # run every variant in VARIANTS and compare
python eval_prompts.py --live --judge      # add an LLM-as-judge faithfulness score
```

## The four prompts (this is the product)
All live near the top of `claims_copilot.py` and are easy to tune:
- **PLANNER_SYSTEM** — decides which queries to run and which real-time tools to call.
- **RERANK_SYSTEM** — orders retrieved candidates by relevance (semantic layer).
- **DRAFTER_SYSTEM** — drafts the determination; every point must cite a `doc_id`.
- **VERIFIER_SYSTEM** — checks grounding; forces `ABSTAIN` when unsupported.

## Which "advanced RAG" piece is where
- **Agentic** — the planner chooses sources/tools and the verifier can abstain.
- **Hybrid + RRF** — `HybridRetriever` fuses BM25 and TF-IDF rankings.
- **Rerank** — `RERANK_SYSTEM` re-scores candidates (swap in a cross-encoder later).
- **Faithfulness/citations** — drafter cites, verifier gates.
- **Real-time** — `tool_weather_cat` / `tool_sanctions_screen` (stubbed).
- **Confidence score** — 0–100 score derived from the verifier decision and
  faithfulness check, displayed as an ASCII bar (0 = LOW → 100 = HIGH).

## Confidence score

Step 7 of the pipeline computes a 0–100 score and renders it as a bar — in the
terminal **and** in the GitHub Actions job summary (see the **Actions** tab after
any push).

**Terminal output — example for a COVERED claim (Austin water heater case)**
```
7) CONFIDENCE SCORE
------------------------------------------------------------------------------
  [██████████████████████████████████░░░░░░]  85/100  (HIGH  )
   0 ←──────────────────────────────────────→ 100
   LOW                                    HIGH
```

> **COVERED** means the evidence conclusively supports coverage with no open
> questions — here, a confirmed sudden/accidental appliance rupture with a
> matching prior-claim precedent.

> **PENDING** means the proximate cause is unconfirmed and no final determination
> can be issued until a field adjuster investigates. It scores lower than PARTIAL
> because the claim is not yet resolvable — not merely uncertain.

**All possible decisions at a glance**

| Decision | Meaning | Confidence base |
|----------|---------|-----------------|
| `COVERED` | Evidence conclusively supports coverage | 85 |
| `DENIED` | Evidence conclusively supports denial | 70 |
| `PARTIAL` | Partial coverage clearly applies; open questions remain | 55 |
| `PENDING` | Proximate cause unconfirmed; field investigation required | 35 |
| `ABSTAIN` | Verifier found unsupported claims; escalate to human | 15 |

**GitHub Actions job summary** — the workflow renders an HTML progress bar with
colour coding that is visible directly on GitHub without opening logs:

| Threshold | Label | Colour |
|-----------|-------|--------|
| ≥ 70 | **HIGH** | 🟢 green |
| 40 – 69 | **MEDIUM** | 🟡 yellow |
| < 40 | **LOW** | 🔴 red |

**Full scoring formula**

| Factor | Effect |
|--------|--------|
| Decision = COVERED | base 85 |
| Decision = DENIED | base 70 |
| Decision = PARTIAL | base 55 |
| Decision = PENDING | base 35 |
| Decision = ABSTAIN | base 15 |
| `all_supported` is false | − 30 |
| Each open question | − 8 (max − 24) |

The score is also returned in the `run()` dict as `"confidence_score"` for
downstream use.

## Where to extend (in order of payoff)
1. ~~**Real embeddings**~~ — done: `--embedder {hashing,voyage,openai,st}` fuses a
   third, semantic ranked list into `HybridRetriever` via RRF. See
   [Dense (semantic) retrieval](#dense-semantic-retrieval---embedder) above.
2. **Real-time feeds** — replace the tool stubs with live calls (e.g. National
   Weather Service for CAT, the public OFAC SDN list for screening). Your runtime
   must allow network egress to those domains.
3. **GraphRAG** — model policy → coverage → exclusion → endorsement and
   claimant → prior-claim relationships so multi-hop questions ("excluded given
   endorsement X *and* prior claim Y?") are answered by traversal, not proximity.
4. **Evaluation** — build a small gold set of correct determinations and score
   faithfulness + answer-relevance with RAGAS before changing prompts.

## Responsible-use notes (important in a regulated domain)
- Keep a **human in the loop**; the copilot drafts, it does not decide.
- **Always cite**, and **abstain** when evidence is missing.
- **Log** inputs/outputs for audit; use **synthetic data only** in development —
  no real customer PII/PHI.
