#!/usr/bin/env python3
"""
Prompt eval + A/B harness for the Claims Coverage Copilot
=========================================================
Test and improve the four prompts (planner / rerank / drafter / verifier) by
running them over a small GOLD SET of claims and scoring the outputs.

What it measures (all computed automatically):
  - decision   : did the final decision match the expected one?
  - recall     : did the must-have documents reach the evidence set?
              (driven by the PLANNER queries + the RERANK ordering)
  - grounded   : is every cited doc_id actually in the evidence?  (no hallucinated cites)
  - cite_cov   : did it cite the docs we expect it to rely on?
  - abstain    : did it ABSTAIN exactly when it should, and not otherwise?

How to run
----------
    # Offline, no key — proves the harness + scorers work (uses canned replies):
    python eval_prompts.py

    # Real API — this is where prompt changes actually move the numbers:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python eval_prompts.py --live

    # Compare prompt variants head-to-head (best with --live):
    python eval_prompts.py --live --compare

    # Add an LLM-as-judge faithfulness score (live only):
    python eval_prompts.py --live --judge

The improvement loop: edit a prompt in claims_copilot.py (or add a VARIANT
below), run `--live --compare`, keep the version with the better numbers.

NOTE on offline mode: in --dry-run the LLM replies are CANNED, so editing a
prompt does NOT change them — offline is for exercising the harness and scorers.
The canned set below represents plausible *current* behaviour, and the
'ambiguous_no_evidence' case deliberately GUESSES instead of abstaining so you
can see a real failure being caught. Run --live to measure real prompt changes.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from contextlib import contextmanager
from dataclasses import dataclass, field

import claims_copilot as cc


# --------------------------------------------------------------------------- #
# Gold test set. Add your own cases here — this is what you tune prompts against.
# --------------------------------------------------------------------------- #
@dataclass
class Case:
    name: str
    claim: str
    expected_decision: str          # COVERED | PARTIAL | DENIED | ABSTAIN
    must_retrieve: set = field(default_factory=set)   # doc_ids that should reach evidence
    must_cite: set = field(default_factory=set)       # doc_ids the draft should cite
    expect_abstain: bool = False


CASES: list[Case] = [
    Case(
        name="basement_water_partial",
        claim=("FNOL: Water in finished basement after heavy rain on 2026-03-14 in "
               "Austin, TX. Cause unconfirmed — possibly surface water, possibly a "
               "drain backup. Policy HO-1002."),
        expected_decision="PARTIAL",
        must_retrieve={"POL-BASE", "END-HO217", "END-HO305"},
        must_cite={"END-HO217", "END-HO305"},
    ),
    Case(
        name="water_heater_covered",
        claim=("FNOL: Water heater tank ruptured suddenly and discharged into the "
               "basement, damaging flooring and stored property. Policy HO-1002."),
        expected_decision="COVERED",
        must_retrieve={"POL-BASE", "CLM-2202"},
        must_cite={"POL-BASE"},
    ),
    Case(
        name="surface_water_denied",
        claim=("FNOL: Adjuster confirmed surface water entered through a foundation "
               "crack after heavy rain and flooded the basement. No sewer/drain "
               "backup involved. Policy HO-1002."),
        expected_decision="DENIED",
        must_retrieve={"END-HO217", "DEF-FLOOD"},
        must_cite={"END-HO217"},
    ),
    Case(
        name="ambiguous_no_evidence",
        claim="FNOL: Claimant reports 'damage to property'. No cause, peril, date, or policy number given.",
        expected_decision="ABSTAIN",
        must_retrieve=set(),          # nothing clearly relevant -> recall is N/A
        must_cite=set(),
        expect_abstain=True,
    ),
]


# --------------------------------------------------------------------------- #
# Offline canned replies, per case (used ONLY with --dry-run).
# Keys must be the pipeline steps: plan / rerank / draft / verify.
# --------------------------------------------------------------------------- #
OFFLINE: dict[str, dict[str, str]] = {
    "basement_water_partial": {
        "plan": json.dumps({"queries": [
            "policy covers sudden accidental discharge water",
            "HO-217 water exclusion flood surface water excluded",
            "HO-305 water backup sump overflow sublimit 5000",
            "proximate cause water loss guideline",
        ], "tools": ["weather_cat", "sanctions_screen"], "reasoning": "ambiguous water loss"}),
        "rerank": json.dumps({"ranked_ids": ["END-HO217", "END-HO305", "POL-BASE", "GUIDE-WTR", "DEF-FLOOD", "CLM-2201"]}),
        "draft": json.dumps({
            "decision": "PARTIAL",
            "rationale": "Base policy covers sudden discharge (POL-BASE); HO-217 excludes flood/surface water (END-HO217); HO-305 adds backup coverage to $5,000 (END-HO305); confirm cause per guideline (GUIDE-WTR).",
            "citations": [
                {"doc_id": "POL-BASE", "point": "Covers sudden and accidental water discharge"},
                {"doc_id": "END-HO217", "point": "Excludes flood and surface water"},
                {"doc_id": "END-HO305", "point": "Adds water-backup coverage up to $5,000"},
                {"doc_id": "GUIDE-WTR", "point": "Confirm proximate cause; pull weather data"},
            ],
            "open_questions": ["Surface water (excluded) or drain backup (covered to $5,000)?"],
        }),
        "verify": json.dumps({"all_supported": True, "unsupported_points": [], "final_decision": "PARTIAL", "notes": "Cited points trace to evidence; cause is an open question."}),
    },
    "water_heater_covered": {
        "plan": json.dumps({"queries": [
            "sudden accidental appliance water discharge coverage",
            "water heater tank ruptured covered claim",
            "dwelling personal property water damage",
        ], "tools": [], "reasoning": "appliance burst, likely covered"}),
        "rerank": json.dumps({"ranked_ids": ["POL-BASE", "CLM-2202", "GUIDE-WTR", "END-HO305"]}),
        "draft": json.dumps({
            "decision": "COVERED",
            "rationale": "A water heater rupture is sudden and accidental discharge from an appliance, which the base policy covers (POL-BASE); a prior claim with the same facts was covered (CLM-2202).",
            "citations": [
                {"doc_id": "POL-BASE", "point": "Covers sudden and accidental discharge from an appliance"},
                {"doc_id": "CLM-2202", "point": "Prior water-heater rupture covered"},
            ],
            "open_questions": ["Confirm the discharge was sudden, not long-term seepage."],
        }),
        "verify": json.dumps({"all_supported": True, "unsupported_points": [], "final_decision": "COVERED", "notes": "Both citations supported by evidence."}),
    },
    "surface_water_denied": {
        "plan": json.dumps({"queries": [
            "flood surface water exclusion endorsement",
            "surface water foundation crack denied claim",
            "definition flood surface water rainfall",
        ], "tools": [], "reasoning": "confirmed surface water -> exclusion"}),
        "rerank": json.dumps({"ranked_ids": ["END-HO217", "DEF-FLOOD", "CLM-2201", "GUIDE-WTR"]}),
        "draft": json.dumps({
            "decision": "DENIED",
            "rationale": "The confirmed cause is surface water entering through the foundation, which meets the policy definition (DEF-FLOOD) and is excluded by HO-217 regardless of contributing cause (END-HO217).",
            "citations": [
                {"doc_id": "END-HO217", "point": "Excludes flood and surface water"},
                {"doc_id": "DEF-FLOOD", "point": "Surface water includes rain runoff on the ground"},
            ],
            "open_questions": [],
        }),
        "verify": json.dumps({"all_supported": True, "unsupported_points": [], "final_decision": "DENIED", "notes": "Exclusion applies on the confirmed facts."}),
    },
    "ambiguous_no_evidence": {
        # Deliberately WEAK: guesses DENIED and the verifier fails to abstain.
        "plan": json.dumps({"queries": [
            "property damage coverage policy",
            "covered perils dwelling",
            "loss cause peril determination",
        ], "tools": [], "reasoning": "vague claim"}),
        "rerank": json.dumps({"ranked_ids": ["POL-BASE", "GUIDE-WTR", "END-HO217", "DEF-FLOOD"]}),
        "draft": json.dumps({
            "decision": "DENIED",
            "rationale": "No covered peril is evident from the description, so the loss is not covered (POL-BASE).",
            "citations": [{"doc_id": "POL-BASE", "point": "Lists covered perils"}],
            "open_questions": [],
        }),
        "verify": json.dumps({"all_supported": True, "unsupported_points": [], "final_decision": "DENIED", "notes": "Accepted the draft."}),
    },
}


def make_mock_fn(case_name: str):
    bank = OFFLINE[case_name]
    def mock_fn(*, step: str, system: str, user: str) -> str:
        return bank[step]
    return mock_fn


# --------------------------------------------------------------------------- #
# Scorers (deterministic; no API needed).
# --------------------------------------------------------------------------- #
def score_case(case: Case, result: dict) -> dict:
    evidence = set(result["evidence"])
    cited = {c.get("doc_id") for c in result["draft"].get("citations", [])}
    final = result["verify"].get("final_decision", "")

    recall = (len(case.must_retrieve & evidence) / len(case.must_retrieve)
              if case.must_retrieve else None)
    cite_cov = (len(case.must_cite & cited) / len(case.must_cite)
                if case.must_cite else None)
    grounded = all(c in evidence for c in cited) if cited else True
    abstain_ok = (final == "ABSTAIN") if case.expect_abstain else (final != "ABSTAIN")

    return {
        "final": final,
        "decision_ok": final == case.expected_decision,
        "recall": recall,
        "grounded": grounded,
        "cite_cov": cite_cov,
        "abstain_ok": abstain_ok,
    }


# --------------------------------------------------------------------------- #
# Optional LLM-as-judge faithfulness (live only).
# --------------------------------------------------------------------------- #
JUDGE_SYSTEM = """You grade an insurance coverage determination for faithfulness.
Given the draft and the evidence it must rely on, rate how fully every point in
the rationale is supported by that evidence. Respond with ONLY JSON:
{"faithfulness": 0.0-1.0, "notes": "one sentence"}"""


def judge_faithfulness(result: dict, judge_llm: "cc.LLM") -> float:
    ev_text = "\n".join(f"[{i}] {cc.ID_TO_DOC[i]['text']}" for i in result["evidence"])
    out = judge_llm.complete(
        step="judge", system=JUDGE_SYSTEM, model=cc.REASONING_MODEL,
        user=f"<draft>\n{json.dumps(result['draft'])}\n</draft>\n<evidence>\n{ev_text}\n</evidence>",
    )
    try:
        return float(cc.parse_json(out).get("faithfulness", 0.0))
    except Exception:
        return 0.0


# --------------------------------------------------------------------------- #
# Run the gold set once and score it.
# --------------------------------------------------------------------------- #
def evaluate(live: bool, judge: bool) -> list[dict]:
    retriever = cc.HybridRetriever(cc.CORPUS)
    live_llm = cc.LLM(dry_run=False) if live else None
    rows = []
    for case in CASES:
        llm = live_llm if live else cc.LLM(dry_run=True, mock_fn=make_mock_fn(case.name))
        with contextlib.redirect_stdout(io.StringIO()):   # silence the pipeline's own printing
            result = cc.run(case.claim, retriever, llm)
        scores = score_case(case, result)
        if judge and live:
            scores["faith"] = judge_faithfulness(result, live_llm)
        rows.append({"case": case, "scores": scores})
    return rows


# --------------------------------------------------------------------------- #
# Reporting.
# --------------------------------------------------------------------------- #
def _b(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _p(x) -> str:
    return " n/a" if x is None else f"{x:>3.0%}"


def report(rows: list[dict], judge: bool) -> dict:
    head = f"{'case':<24}{'final':<9}{'decision':<10}{'recall':<8}{'cited?':<8}{'ground':<8}{'abstain':<9}"
    if judge:
        head += f"{'faith':<7}"
    print(head)
    print("-" * len(head))
    agg = {"decision": [], "recall": [], "grounded": [], "abstain": [], "faith": []}
    for r in rows:
        s = r["scores"]
        line = (f"{r['case'].name:<24}{s['final']:<9}{_b(s['decision_ok']):<10}"
                f"{_p(s['recall']):<8}{_p(s['cite_cov']):<8}{_b(s['grounded']):<8}{_b(s['abstain_ok']):<9}")
        if judge:
            line += f"{s.get('faith', 0.0):<7.2f}"
        print(line)
        agg["decision"].append(s["decision_ok"])
        agg["abstain"].append(s["abstain_ok"])
        agg["grounded"].append(s["grounded"])
        if s["recall"] is not None:
            agg["recall"].append(s["recall"])
        if judge and "faith" in s:
            agg["faith"].append(s["faith"])

    mean = lambda xs: (sum(xs) / len(xs)) if xs else None
    summary = {
        "decision_acc": mean(agg["decision"]),
        "recall": mean(agg["recall"]),
        "grounded_rate": mean(agg["grounded"]),
        "abstain_acc": mean(agg["abstain"]),
        "faith": mean(agg["faith"]) if judge else None,
    }
    print("-" * len(head))
    tail = (f"{'AGGREGATE':<24}{'':<9}{_p(summary['decision_acc']):<10}"
            f"{_p(summary['recall']):<8}{'':<8}{_p(summary['grounded_rate']):<8}{_p(summary['abstain_acc']):<9}")
    if judge:
        tail += f"{(summary['faith'] or 0.0):<7.2f}"
    print(tail)
    return summary


# --------------------------------------------------------------------------- #
# Prompt variants. Each maps prompt-constant names -> replacement text.
# Add variants here, then run `--compare` to see which scores better.
# --------------------------------------------------------------------------- #
VARIANTS: dict[str, dict[str, str]] = {
    "baseline": {},
    "verifier_v2_strict_abstain": {
        "VERIFIER_SYSTEM": cc.VERIFIER_SYSTEM + (
            "\n\nHARD RULE: if the claim does not state a cause of loss, a peril, a "
            "date, or a policy identifier, you MUST set final_decision to "
            '"ABSTAIN" regardless of what the draft concluded.'
        ),
    },
}


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


def compare(live: bool, judge: bool) -> None:
    summaries = {}
    for name, overrides in VARIANTS.items():
        print("\n" + "=" * 78)
        print(f"VARIANT: {name}   ({'overrides: ' + ', '.join(overrides) if overrides else 'no changes'})")
        print("=" * 78)
        with apply_overrides(overrides):
            summaries[name] = report(evaluate(live, judge), judge)

    print("\n" + "=" * 78)
    print("VARIANT COMPARISON (higher is better)")
    print("=" * 78)
    metrics = ["decision_acc", "recall", "grounded_rate", "abstain_acc"] + (["faith"] if judge else [])
    print(f"{'variant':<30}" + "".join(f"{m:<15}" for m in metrics))
    print("-" * (30 + 15 * len(metrics)))
    for name, s in summaries.items():
        print(f"{name:<30}" + "".join(f"{_p(s[m]) if m != 'faith' else f'{(s[m] or 0):.2f}':<15}" for m in metrics))
    if not live:
        print("\n(Offline replies are fixed, so variants tie here. Run with --live to "
              "measure real prompt changes — verifier_v2 should fix 'ambiguous_no_evidence'.)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prompt eval + A/B harness for the claims copilot")
    ap.add_argument("--live", action="store_true", help="call the real Anthropic API (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--compare", action="store_true", help="run every prompt variant and compare")
    ap.add_argument("--judge", action="store_true", help="add an LLM-as-judge faithfulness score (live only)")
    args = ap.parse_args()

    import os
    if args.live and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit('Set ANTHROPIC_API_KEY to use --live, or drop --live to run offline.')
    if args.judge and not args.live:
        print("(--judge needs --live; ignoring the judge offline.)\n")
        args.judge = False

    mode = "LIVE API" if args.live else "DRY-RUN (offline, canned replies)"
    print(f"Prompt eval — mode: {mode} — {len(CASES)} gold cases\n")

    if args.compare:
        compare(args.live, args.judge)
    else:
        report(evaluate(args.live, args.judge), args.judge)


if __name__ == "__main__":
    main()
