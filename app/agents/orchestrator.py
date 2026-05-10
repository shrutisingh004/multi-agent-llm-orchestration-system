"""
Master orchestrator: routes between agents dynamically.
Agents are NEVER called directly by each other - only through this.
"""
from __future__ import annotations
import asyncio
import json
import time
from typing import AsyncIterator, Callable

from app.agents.context import SharedContext, ContextBudgetManager, estimate_tokens
from app.agents.agents import (
    DecompositionAgent, RetrievalAgent, CritiqueAgent, SynthesisAgent, CompressionAgent
)
from app.agents.llm import call_llm_json
from app.agents.prompts import get_prompt


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        return json.loads(text)
    except Exception:
        return {}


class Orchestrator:
    def __init__(self):
        self.budget_manager = ContextBudgetManager()

    async def run(self, ctx: SharedContext, emit: Callable = None) -> SharedContext:
        """
        Run the full multi-agent pipeline.
        emit(event_type, agent_id, data) is called for SSE streaming.
        """
        async def _emit(event_type: str, agent_id: str, data: dict):
            if emit:
                await emit(event_type, agent_id, data)

        await _emit("start", "orchestrator", {"query": ctx.original_query, "job_id": ctx.job_id})

        # Step 1: Determine routing plan
        routing_prompt = (
            f"{get_prompt('orchestrator')}\n\nQuery: {ctx.original_query}\n\n"
            "Return JSON: {\"agents\": [\"decomposition\", \"retrieval\", \"critique\", \"synthesis\"], \"justification\": \"...\"}"
        )
        routing_raw = await call_llm_json(routing_prompt)
        routing_data = _parse_json(routing_raw)
        agents_plan = routing_data.get("agents", ["decomposition", "retrieval", "critique", "synthesis"])
        justification = routing_data.get("justification", "Default pipeline")

        ctx.log_routing("plan", justification, str(agents_plan))
        await _emit("routing", "orchestrator", {"plan": agents_plan, "justification": justification})

        # Step 2: Execute agents in order
        agent_map = {
            "decomposition": DecompositionAgent("decomposition", self.budget_manager),
            "retrieval": RetrievalAgent("retrieval", self.budget_manager),
            "critique": CritiqueAgent("critique", self.budget_manager),
            "synthesis": SynthesisAgent("synthesis", self.budget_manager),
        }

        for agent_id in agents_plan:
            if agent_id not in agent_map:
                continue

            # Check if compression is needed before agent runs
            remaining = self.budget_manager.remaining(agent_id)
            if remaining < 500:
                await _emit("compression", "orchestrator", {"reason": f"Low budget for {agent_id}: {remaining} tokens"})
                compressor = CompressionAgent("compression", self.budget_manager)
                await compressor.run(ctx)

            await _emit("agent_start", agent_id, {
                "budget_remaining": self.budget_manager.remaining(agent_id),
                "context_keys": list(ctx.agent_outputs.keys()),
            })
            start = time.time()

            try:
                output = await agent_map[agent_id].run(ctx)
                latency = (time.time() - start) * 1000

                await _emit("agent_done", agent_id, {
                    "latency_ms": latency,
                    "token_count": output.token_count,
                    "output_preview": output.content[:200],
                    "budget_remaining": self.budget_manager.remaining(agent_id),
                })
                ctx.log_routing(f"{agent_id}_complete", f"Completed in {latency:.0f}ms", "next")

            except Exception as e:
                await _emit("agent_error", agent_id, {"error": str(e)})
                ctx.log_routing(f"{agent_id}_error", str(e), "continue")

        # Emit policy violations if any
        if ctx.policy_violations:
            await _emit("policy_violations", "orchestrator", {"violations": ctx.policy_violations})

        await _emit("done", "orchestrator", {
            "final_answer_preview": (ctx.final_answer or "")[:300],
            "budget_usage": self.budget_manager.get_usage(),
            "tool_calls": len(ctx.tool_results),
        })

        return ctx
