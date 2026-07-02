#!/usr/bin/env python3
"""
RAGAS-style evaluation for the Claims Coverage Copilot
=======================================================
Measures four metrics across a gold set of claims:

  1. Faithfulness      (0-1) — what fraction of the draft's atomic claims are
                               supported by the retrieved evidence?
  2. Answer Relevance  (0-1) — does the determination actually address the
                               original claim? (reverse-question cosine similarity)
  3. Context Recall    (0-1) — did retrieval surface the docs needed to answer?
                               (gold must-retrieve set)
  4. Context Precision (0-1) — of the retrieved docs, what fraction are relevant?
                               (cited ∪ must-retrieve intersected with retrieved)

A composite RAGAS score is the harmonic mean of all four.

Offline approximations (--dry-run): all four metrics have deterministic
stand-ins that need no API key, so the harness exercises the full scoring
pipeline offline. Run with --live for LLM-computed faithfulness and
answer-relevance, which are meaningfully more accurate.

How to run
----------
    python eval_ragas.py                  # offline, no key needed
    python eval_ragas.py --live           # full LLM-computed metrics (needs ANTHROPIC_API_KEY)
    python eval_ragas.py --live --json    # dump per-case detail as JSON
    python eval_ragas.py --live --compare # A/B two prompt variants
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import re
from dataclasses import dataclass, field
from statistics import mean
from typing import Optional

import claims_copilot as cc

# --------------------------------------------------------------------------- #
# Gold set — ground-truth answers let us compute reference-based metrics.
# Each case carries:
#   reference_answer : the correct rationale (used by faithfulness decomposer)
#   must_retrieve    : doc_ids that MUST appear in retrieved evidence
#   useful_docs      : superset of docs that are relevant to the claim
#                      (used for context precision — anything outside is noise)
#   expected_decision: the correct final decision
#   expect_abstain   : True if the correct behaviour is to ABSTAIN
# --------------------------------------------------------------------------- #
@dataclass
class GoldCase:
    name: str
    claim: str
    expected_decision: str
    reference_answer: str           # ground-truth rationale string
    must_retrieve: set = field(default_factory=set)
    useful_docs: set = field(default_factory=set)   # relevant docs (superset of must_retrieve)
    expect_abstain: bool = False


GOLD: list[GoldCase] = [
    GoldCase(
        name="water_heater_covered",
        claim=(
            "FNOL: Water heater tank ruptured suddenly and discharged into the "
            "basement, damaging flooring and stored property. Adjuster confirmed "
            "sudden and accidental cause. Policy HO-1002."
        ),
        expected_decision="COVERED",
        reference_answer=(
            "A water heater tank rupture is sudden and accidental discharge from a "
            "household appliance. The base policy (POL-BASE) explicitly covers "
            "sudden and accidental discharge from household appliances. A prior claim "
            "with an identical fact pattern was covered (CLM-2202). No exclusion "
            "applies because the adjuster confirmed the cause was sudden, not "
            "long-term seepage."
        ),
        must_retrieve={"POL-BASE", "CLM-2202"},
        useful_docs={"POL-BASE", "CLM-2202", "GUIDE-WTR"},
    ),
    GoldCase(
        name="surface_water_denied",
        claim=(
            "FNOL: Adjuster confirmed surface water entered through a foundation crack "
            "after heavy rain and flooded the basement. No sewer or drain backup "
            "involved. Policy HO-1002."
        ),
        expected_decision="DENIED",
        reference_answer=(
            "The confirmed proximate cause is surface water entering through a "
            "foundation crack. The policy definition (DEF-FLOOD) classifies rain "
            "runoff on the ground as surface water. Endorsement HO-217 (END-HO217) "
            "excludes loss caused by surface water regardless of any contributing "
            "cause, and no backup coverage endorsement applies because the cause "
            "was external surface water, not a sewer or drain backup."
        ),
        must_retrieve={"END-HO217", "DEF-FLOOD"},
        useful_docs={"END-HO217", "DEF-FLOOD", "CLM-2201", "GUIDE-WTR"},
    ),
    GoldCase(
        name="basement_water_partial",
        claim=(
            "FNOL: Water in finished basement after heavy rain on 2026-03-14 in "
            "Austin, TX. Cause unconfirmed — possibly surface water, possibly a "
            "drain backup. Policy HO-1002."
        ),
        expected_decision="PARTIAL",
        reference_answer=(
            "The base policy covers sudden accidental discharge (POL-BASE) but "
            "endorsement HO-217 excludes flood and surface water (END-HO217). "
            "If the water entered via a sewer or drain backup, endorsement HO-305 "
            "provides coverage up to $5,000 (END-HO305). Because the proximate cause "
            "is unconfirmed, partial coverage applies pending adjuster investigation "
            "per the water-loss guideline (GUIDE-WTR)."
        ),
        must_retrieve={"POL-BASE", "END-HO217", "END-HO305"},
        useful_docs={"POL-BASE", "END-HO217", "END-HO305", "GUIDE-WTR", "DEF-FLOOD"},
    ),
    GoldCase(
        name="thunderstorm_wind_damage",
        claim=(
            "FNOL: Severe thunderstorm on 2026-03-14 in Austin, TX caused a tree "
            "branch to fall on the roof. Wind and hail damage to shingles and gutters. "
            "NWS issued a Severe Thunderstorm Warning. Claimant: Jane Doe. Policy HO-1002."
        ),
        expected_decision="COVERED",
        reference_answer=(
            "Wind and hail damage from a severe thunderstorm are covered perils under "
            "the base policy's sudden and accidental direct physical loss clause "
            "(POL-BASE). The NWS Severe Thunderstorm Warning for the loss date "
            "corroborates the storm event. No applicable exclusion exists for wind "
            "or hail damage. Coverage is recommended pending standard inspection."
        ),
        must_retrieve={"POL-BASE"},
        useful_docs={"POL-BASE", "GUIDE-WTR"},
    ),
    GoldCase(
        name="ambiguous_no_evidence",
        claim="FNOL: Claimant reports 'damage to property'. No cause, peril, date, or policy number given.",
        expected_decision="ABSTAIN",
        reference_answer=(
            "The claim provides no cause of loss, no peril, no loss date, and no "
            "policy identifier. Without these facts no coverage determination is "
            "possible and the claim must be escalated to a human adjuster."
        ),
        must_retrieve=set(),
        useful_docs=set(),
        expect_abstain=True,
    ),
]


# --------------------------------------------------------------------------- #
# Offline canned replies (per case) — used when not --live.
# --------------------------------------------------------------------------- #
OFFLINE: dict[str, dict[str, str]] = {
    "water_heater_covered": {
        "plan": json.dumps({"queries": [
            "sudden accidental appliance water discharge coverage",
            "water heater rupture covered claim",
            "dwelling personal property appliance damage",
        ], "tools": [], "reasoning": "appliance burst, confirmed sudden cause"}),
        "rerank": json.dumps({"ranked_ids": ["POL-BASE", "CLM-2202", "GUIDE-WTR", "END-HO305"]}),
        "draft": json.dumps({
            "decision": "COVERED",
            "rationale": (
                "A water heater rupture is sudden and accidental discharge from an appliance; "
                "the base policy covers this peril (POL-BASE). A prior claim with identical "
                "facts was covered (CLM-2202). No exclusion applies."
            ),
            "citations": [
                {"doc_id": "POL-BASE", "point": "Covers sudden and accidental discharge from household appliance"},
                {"doc_id": "CLM-2202", "point": "Prior water-heater rupture with same facts was covered"},
            ],
            "open_questions": [],
        }),
        "verify": json.dumps({
            "all_supported": True, "unsupported_points": [],
            "final_decision": "COVERED",
            "notes": "Both citations trace to retrieved documents; sudden cause confirmed.",
        }),
    },
    "surface_water_denied": {
        "plan": json.dumps({"queries": [
            "flood surface water exclusion endorsement HO-217",
            "surface water foundation crack denied",
            "definition flood surface water rainfall runoff",
        ], "tools": [], "reasoning": "confirmed surface water, exclusion applies"}),
        "rerank": json.dumps({"ranked_ids": ["END-HO217", "DEF-FLOOD", "CLM-2201", "GUIDE-WTR"]}),
        "draft": json.dumps({
            "decision": "DENIED",
            "rationale": (
                "The confirmed cause is surface water through a foundation crack, which meets "
                "the policy definition of surface water (DEF-FLOOD). Endorsement HO-217 "
                "excludes surface water loss regardless of contributing cause (END-HO217)."
            ),
            "citations": [
                {"doc_id": "END-HO217", "point": "Excludes flood and surface water"},
                {"doc_id": "DEF-FLOOD", "point": "Defines surface water as rain runoff on ground"},
            ],
            "open_questions": [],
        }),
        "verify": json.dumps({
            "all_supported": True, "unsupported_points": [],
            "final_decision": "DENIED",
            "notes": "Exclusion clearly applies on the confirmed facts.",
        }),
    },
    "basement_water_partial": {
        "plan": json.dumps({"queries": [
            "policy covers sudden accidental discharge water",
            "HO-217 water exclusion flood surface water",
            "HO-305 water backup sump overflow sublimit",
            "proximate cause water loss guideline",
        ], "tools": ["weather_cat"], "reasoning": "ambiguous water loss — need cause"}),
        "rerank": json.dumps({"ranked_ids": ["END-HO217", "END-HO305", "POL-BASE", "GUIDE-WTR", "DEF-FLOOD", "CLM-2201"]}),
        "draft": json.dumps({
            "decision": "PARTIAL",
            "rationale": (
                "Base policy covers sudden discharge (POL-BASE); HO-217 excludes surface water (END-HO217); "
                "HO-305 provides up to $5,000 for backup/drain water (END-HO305). "
                "Cause unconfirmed; partial coverage pending investigation per guideline (GUIDE-WTR)."
            ),
            "citations": [
                {"doc_id": "POL-BASE", "point": "Covers sudden and accidental discharge"},
                {"doc_id": "END-HO217", "point": "Excludes flood and surface water"},
                {"doc_id": "END-HO305", "point": "Backup coverage up to $5,000"},
                {"doc_id": "GUIDE-WTR", "point": "Confirm proximate cause before deciding"},
            ],
            "open_questions": ["Was the intrusion surface water (excluded) or drain backup (up to $5,000)?"],
        }),
        "verify": json.dumps({
            "all_supported": True, "unsupported_points": [],
            "final_decision": "PARTIAL",
            "notes": "All cited points trace to evidence; cause remains an open question.",
        }),
    },
    "thunderstorm_wind_damage": {
        "plan": json.dumps({"queries": [
            "wind hail storm damage covered peril dwelling",
            "thunderstorm roof damage coverage",
            "sudden accidental direct physical loss policy",
        ], "tools": ["weather_cat"], "reasoning": "storm event — weather tool needed"}),
        "rerank": json.dumps({"ranked_ids": ["POL-BASE", "GUIDE-WTR", "END-HO217", "DEF-FLOOD"]}),
        "draft": json.dumps({
            "decision": "COVERED",
            "rationale": (
                "Wind and hail from a severe thunderstorm are sudden accidental direct physical "
                "loss events covered under the base policy (POL-BASE). The NWS Severe Thunderstorm "
                "Warning corroborates the storm event on the loss date. No exclusion applies to "
                "wind or hail damage."
            ),
            "citations": [
                {"doc_id": "POL-BASE", "point": "Covers sudden and accidental direct physical loss"},
            ],
            "open_questions": ["Confirm extent of roof/gutter damage via inspection."],
        }),
        "verify": json.dumps({
            "all_supported": True, "unsupported_points": [],
            "final_decision": "COVERED",
            "notes": "Citation traces to policy; NWS alert corroborates the storm event.",
        }),
    },
    "ambiguous_no_evidence": {
        "plan": json.dumps({"queries": [
            "property damage coverage policy",
            "covered perils dwelling",
        ], "tools": [], "reasoning": "vague claim — no peril or date stated"}),
        "rerank": json.dumps({"ranked_ids": ["POL-BASE", "GUIDE-WTR", "END-HO217", "DEF-FLOOD"]}),
        "draft": json.dumps({
            "decision": "PENDING",
            "rationale": "No cause of loss, peril, date, or policy number provided. Cannot determine coverage.",
            "citations": [],
            "open_questions": ["What is the cause of loss?", "What is the policy number?", "What is the loss date?"],
        }),
        "verify": json.dumps({
            "all_supported": True, "unsupported_points": [],
            "final_decision": "ABSTAIN",
            "notes": "Insufficient information; escalate to human adjuster.",
        }),
    },
}


def make_mock_fn(case_name: str):
    bank = OFFLINE[case_name]
    def _fn(*, step: str, system: str, user: str) -> str:
        return bank.get(step, "{}")
    return _fn


# --------------------------------------------------------------------------- #
# Metric 1: Faithfulness
# LLM mode: decompose rationale into atomic claims, verify each against evidence.
# Offline proxy: use verifier's all_supported + unsupported_points count.
# --------------------------------------------------------------------------- #
_FAITH_DECOMPOSE = """You are a faithfulness evaluator for insurance RAG systems.
Given a draft rationale and the retrieved evidence, decompose the rationale into
individual atomic claims (one per citation point or factual assertion), then for
each claim mark it as SUPPORTED or UNSUPPORTED by the evidence.
Respond ONLY with JSON:
{
  "claims": [
    {"claim": "short description", "supported": true},
    ...
  ]
}"""

def faithfulness_llm(result: dict, llm: cc.LLM) -> float:
    ev_text = "\n".join(
        f"[{i}] {cc.ID_TO_DOC[i]['text']}" for i in result["evidence"] if i in cc.ID_TO_DOC
    )
    rationale = result["draft"].get("rationale", "")
    citations = result["draft"].get("citations", [])
    inp = (f"<rationale>\n{rationale}\n</rationale>\n"
           f"<citations>\n{json.dumps(citations)}\n</citations>\n"
           f"<evidence>\n{ev_text}\n</evidence>")
    raw = llm.complete(step="judge", system=_FAITH_DECOMPOSE, model=cc.REASONING_MODEL, user=inp)
    try:
        claims = cc.parse_json(raw).get("claims", [])
        if not claims:
            return 0.0
        return sum(1 for c in claims if c.get("supported")) / len(claims)
    except Exception:
        return 0.0


def faithfulness_offline(result: dict) -> float:
    """Proxy: full credit if verifier says all_supported, partial deductions otherwise."""
    verify = result["verify"]
    if verify.get("all_supported", False):
        return 1.0
    unsupported = len(verify.get("unsupported_points", []))
    cited = len(result["draft"].get("citations", []))
    if cited == 0:
        return 0.0
    return max(0.0, 1.0 - unsupported / cited)


# --------------------------------------------------------------------------- #
# Metric 2: Answer Relevance
# LLM mode: generate N reverse questions from the answer, score cosine similarity
#           against the original claim using token overlap (no embedding needed).
# Offline proxy: token Jaccard between claim keywords and rationale keywords.
# --------------------------------------------------------------------------- #
_RELEVANCE_SYSTEM = """You are evaluating answer relevance for insurance RAG.
Given a coverage determination, generate 3-4 questions that the determination
is implicitly answering. Focus on the decision and rationale, not the process.
Respond ONLY with JSON:
{"questions": ["question 1", "question 2", ...]}"""

def _token_set(text: str) -> set[str]:
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "and", "or",
                 "of", "to", "in", "for", "on", "with", "at", "by", "from",
                 "that", "this", "it", "be", "not", "no", "has", "have", "had"}
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in stopwords}

def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def answer_relevance_llm(case: GoldCase, result: dict, llm: cc.LLM) -> float:
    determination = (
        f"decision: {result['draft'].get('decision')}\n"
        f"rationale: {result['draft'].get('rationale')}"
    )
    raw = llm.complete(
        step="judge", system=_RELEVANCE_SYSTEM, model=cc.REASONING_MODEL,
        user=f"<determination>\n{determination}\n</determination>",
    )
    try:
        questions = cc.parse_json(raw).get("questions", [])
    except Exception:
        questions = []
    if not questions:
        return 0.0
    claim_toks = _token_set(case.claim)
    scores = [_jaccard(claim_toks, _token_set(q)) for q in questions]
    return mean(scores) if scores else 0.0


def answer_relevance_offline(case: GoldCase, result: dict) -> float:
    """Token Jaccard between claim keywords and the full rationale."""
    rationale = result["draft"].get("rationale", "")
    decision = result["verify"].get("final_decision", "")
    answer_text = f"{decision} {rationale}"
    return _jaccard(_token_set(case.claim), _token_set(answer_text))


# --------------------------------------------------------------------------- #
# Metric 3: Context Recall (0-1)
# Fraction of must-retrieve docs that actually appear in the evidence set.
# Fully deterministic; no LLM needed.
# --------------------------------------------------------------------------- #
def context_recall(case: GoldCase, result: dict) -> Optional[float]:
    if not case.must_retrieve:
        return None   # vacuously undefined — skip from aggregate
    evidence = set(result["evidence"])
    return len(case.must_retrieve & evidence) / len(case.must_retrieve)


# --------------------------------------------------------------------------- #
# Metric 4: Context Precision (0-1)
# Of the retrieved docs, what fraction are relevant (in useful_docs)?
# Fully deterministic; no LLM needed.
# --------------------------------------------------------------------------- #
def context_precision(case: GoldCase, result: dict) -> Optional[float]:
    if not case.useful_docs:
        return None
    evidence = set(result["evidence"])
    if not evidence:
        return 0.0
    relevant_retrieved = case.useful_docs & evidence
    return len(relevant_retrieved) / len(evidence)


# --------------------------------------------------------------------------- #
# Composite RAGAS score: harmonic mean of the four metrics.
# --------------------------------------------------------------------------- #
def ragas_score(faith: float, relevance: float, recall: Optional[float], precision: Optional[float]) -> float:
    values = [faith, relevance]
    if recall is not None:
        values.append(recall)
    if precision is not None:
        values.append(precision)
    if not values or any(v == 0 for v in values):
        return 0.0
    return len(values) / sum(1 / v for v in values)


# --------------------------------------------------------------------------- #
# Run one case through the pipeline and collect all four metrics.
# --------------------------------------------------------------------------- #
def evaluate_case(case: GoldCase, retriever: cc.HybridRetriever, llm: cc.LLM,
                  live: bool, judge_llm: Optional[cc.LLM]) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        result = cc.run(case.claim, retriever, llm)

    faith = (faithfulness_llm(result, judge_llm) if live and judge_llm
             else faithfulness_offline(result))
    relevance = (answer_relevance_llm(case, result, judge_llm) if live and judge_llm
                 else answer_relevance_offline(case, result))
    recall    = context_recall(case, result)
    precision = context_precision(case, result)
    composite = ragas_score(faith, relevance, recall, precision)

    final = result["verify"].get("final_decision", "")
    decision_ok = final == case.expected_decision
    abstain_ok  = (final == "ABSTAIN") if case.expect_abstain else (final != "ABSTAIN")

    return {
        "case": case,
        "result": result,
        "metrics": {
            "faithfulness":      faith,
            "answer_relevance":  relevance,
            "context_recall":    recall,
            "context_precision": precision,
            "ragas":             composite,
            "final_decision":    final,
            "decision_ok":       decision_ok,
            "abstain_ok":        abstain_ok,
        },
    }


def evaluate(live: bool) -> list[dict]:
    retriever = cc.HybridRetriever(cc.CORPUS)
    judge_llm = cc.LLM(dry_run=False) if live else None
    rows = []
    for case in GOLD:
        llm = (cc.LLM(dry_run=False) if live
               else cc.LLM(dry_run=True, mock_fn=make_mock_fn(case.name)))
        rows.append(evaluate_case(case, retriever, llm, live, judge_llm))
    return rows


# --------------------------------------------------------------------------- #
# Reporting.
# --------------------------------------------------------------------------- #
def _pct(v: Optional[float]) -> str:
    return " n/a " if v is None else f"{v:5.1%}"

def _ok(b: bool) -> str:
    return "✓" if b else "✗"

def report(rows: list[dict], emit_json: bool = False) -> dict:
    COL = 22
    header = (f"{'case':<{COL}}{'faith':>7}{'relevnc':>8}{'recall':>7}{'precis':>7}"
              f"{'RAGAS':>7}  {'dec':>4}{'abs':>4}")
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)

    agg: dict[str, list] = {k: [] for k in
        ("faithfulness", "answer_relevance", "context_recall", "context_precision", "ragas")}
    dec_results, abs_results = [], []

    for r in rows:
        m = r["metrics"]
        name = r["case"].name[:COL]
        print(
            f"{name:<{COL}}"
            f"{_pct(m['faithfulness']):>7}"
            f"{_pct(m['answer_relevance']):>8}"
            f"{_pct(m['context_recall']):>7}"
            f"{_pct(m['context_precision']):>7}"
            f"{_pct(m['ragas']):>7}"
            f"  {_ok(m['decision_ok']):>4}{_ok(m['abstain_ok']):>4}"
        )
        for k in ("faithfulness", "answer_relevance", "ragas"):
            agg[k].append(m[k])
        if m["context_recall"] is not None:
            agg["context_recall"].append(m["context_recall"])
        if m["context_precision"] is not None:
            agg["context_precision"].append(m["context_precision"])
        dec_results.append(m["decision_ok"])
        abs_results.append(m["abstain_ok"])

    def _avg(lst): return mean(lst) if lst else None

    summary = {
        "faithfulness":     _avg(agg["faithfulness"]),
        "answer_relevance": _avg(agg["answer_relevance"]),
        "context_recall":   _avg(agg["context_recall"]),
        "context_precision":_avg(agg["context_precision"]),
        "ragas":            _avg(agg["ragas"]),
        "decision_acc":     _avg(dec_results),
        "abstain_acc":      _avg(abs_results),
    }

    print(sep)
    print(
        f"{'MEAN':<{COL}}"
        f"{_pct(summary['faithfulness']):>7}"
        f"{_pct(summary['answer_relevance']):>8}"
        f"{_pct(summary['context_recall']):>7}"
        f"{_pct(summary['context_precision']):>7}"
        f"{_pct(summary['ragas']):>7}"
        f"  {_pct(summary['decision_acc']):>4}{_pct(summary['abstain_acc']):>4}"
    )
    print(sep)
    print(f"\nDecision accuracy: {_pct(summary['decision_acc'])}  |  "
          f"Abstain accuracy: {_pct(summary['abstain_acc'])}  |  "
          f"Composite RAGAS: {_pct(summary['ragas'])}")

    if emit_json:
        out = []
        for r in rows:
            m = r["metrics"].copy()
            m.pop("result", None)
            out.append({"case": r["case"].name, "metrics": m})
        print("\n" + json.dumps({"summary": summary, "cases": out}, indent=2))

    return summary


# --------------------------------------------------------------------------- #
# Prompt variant A/B comparison.
# --------------------------------------------------------------------------- #
VARIANTS: dict[str, dict[str, str]] = {
    "baseline": {},
    "strict_verifier": {
        "VERIFIER_SYSTEM": cc.VERIFIER_SYSTEM + (
            "\n\nHARD RULE: if the claim provides no cause of loss, no peril, no date, "
            'and no policy number, you MUST output final_decision "ABSTAIN" regardless '
            "of what the draft concluded."
        ),
    },
}

from contextlib import contextmanager

@contextmanager
def apply_overrides(overrides: dict[str, str]):
    saved = {k: getattr(cc, k) for k in overrides}
    for k, v in overrides.items():
        setattr(cc, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(cc, k, v)


def compare(live: bool) -> None:
    summaries = {}
    for name, overrides in VARIANTS.items():
        print(f"\n{'═'*60}")
        print(f"  VARIANT: {name}")
        print(f"{'═'*60}\n")
        with apply_overrides(overrides):
            summaries[name] = report(evaluate(live))

    metrics = ["faithfulness", "answer_relevance", "context_recall",
               "context_precision", "ragas", "decision_acc"]
    print(f"\n{'═'*60}")
    print("  VARIANT COMPARISON  (higher = better)")
    print(f"{'═'*60}")
    hdr = f"{'variant':<26}" + "".join(f"{m[:9]:>10}" for m in metrics)
    print(hdr)
    print("─" * len(hdr))
    for name, s in summaries.items():
        print(f"{name:<26}" + "".join(f"{_pct(s[m]):>10}" for m in metrics))
    if not live:
        print("\n(Offline replies are fixed — variants tie here. "
              "Run with --live to see real prompt differences.)")


# --------------------------------------------------------------------------- #
# Entry point.
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="RAGAS-style eval for the claims copilot")
    ap.add_argument("--live",    action="store_true",
                    help="call the real Anthropic API (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--compare", action="store_true",
                    help="run every prompt variant in VARIANTS and compare")
    ap.add_argument("--json",    action="store_true",
                    help="dump per-case metrics as JSON at the end")
    args = ap.parse_args()

    if args.live and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to use --live, or drop --live for offline mode.")

    mode = "LIVE (LLM-computed faithfulness + relevance)" if args.live else "OFFLINE (deterministic proxies)"
    print(f"\nRAGAS Eval  —  {mode}  —  {len(GOLD)} gold cases\n")

    if args.compare:
        compare(args.live)
    else:
        report(evaluate(args.live), emit_json=args.json)


if __name__ == "__main__":
    main()
