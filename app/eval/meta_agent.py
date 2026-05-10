"""
Meta-agent: reads eval failures, proposes prompt rewrites.
Rewrites are stored but NOT automatically applied.
Human approval required via /approve-rewrite endpoint.
"""
from __future__ import annotations
import difflib
import json

from app.agents.llm import call_llm_json
from app.agents.prompts import get_prompt, set_prompt


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        return json.loads(text)
    except Exception:
        return {}


def compute_diff(original: str, proposed: str) -> str:
    """Compute a readable diff between two prompts."""
    orig_lines = original.splitlines(keepends=True)
    prop_lines = proposed.splitlines(keepends=True)
    diff = difflib.unified_diff(orig_lines, prop_lines, fromfile="original", tofile="proposed")
    return "".join(diff)


async def run_meta_agent(eval_results: list[dict], worst_dimension: str) -> dict:
    """
    Analyze eval failures, identify worst prompt, propose rewrite.
    Returns dict with proposed rewrite details.
    """
    # Find which agent dimension failed most
    dimension_scores: dict[str, list[float]] = {}
    for r in eval_results:
        for s in r.get("scores", []):
            dim = s["name"]
            dimension_scores.setdefault(dim, []).append(s["score"])

    if not dimension_scores:
        return {"error": "No dimension scores found"}

    # Find worst-performing dimension
    avg_scores = {dim: sum(v) / len(v) for dim, v in dimension_scores.items()}
    worst_dim = min(avg_scores, key=avg_scores.get)

    # Map dimension to agent
    dim_to_agent = {
        "correctness": "retrieval",
        "citation_accuracy": "retrieval",
        "contradiction_resolution": "synthesis",
        "tool_efficiency": "orchestrator",
        "budget_compliance": "orchestrator",
        "critique_agreement": "critique",
        "adversarial_resistance": "decomposition",
    }
    agent_id = dim_to_agent.get(worst_dim, "orchestrator")
    original_prompt = get_prompt(agent_id)

    # Collect failed cases for context
    failed_cases = [r for r in eval_results if r.get("overall_score", 1.0) < 0.6]
    failed_summary = json.dumps([
        {"id": r["test_case_id"], "query": r["query"][:100], "answer": r["answer"][:100]}
        for r in failed_cases[:5]
    ])

    prompt = f"""{get_prompt("meta")}

Worst performing dimension: {worst_dim} (avg score: {avg_scores[worst_dim]:.2f})
Agent to improve: {agent_id}
Current prompt for {agent_id}:
{original_prompt}

Failed test cases:
{failed_summary}

Propose a rewrite that addresses these failures.
"""

    raw = await call_llm_json(prompt)
    data = _parse_json(raw)

    proposed = data.get("proposed_prompt", original_prompt)
    diff = compute_diff(original_prompt, proposed)
    justification = data.get("justification", "No justification provided")

    return {
        "agent_id": agent_id,
        "dimension": worst_dim,
        "original_prompt": original_prompt,
        "proposed_prompt": proposed,
        "diff": diff,
        "justification": justification,
        "avg_score_before": avg_scores[worst_dim],
    }


def apply_approved_rewrite(agent_id: str, proposed_prompt: str):
    """Apply an approved prompt rewrite to the live prompts."""
    set_prompt(agent_id, proposed_prompt)
