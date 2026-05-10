from __future__ import annotations
import json
import time
import uuid
from typing import AsyncIterator

from app.agents.context import (
    SharedContext, SubTask, Chunk, AgentOutput, Claim, ProvenanceEntry,
    ContextBudgetManager, estimate_tokens, hash_text
)
from app.agents.llm import call_llm_json, call_llm
from app.agents.prompts import get_prompt
from app.tools.tools import call_tool


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        return json.loads(text)
    except Exception:
        return {}


class BaseAgent:
    def __init__(self, agent_id: str, budget_manager: ContextBudgetManager):
        self.agent_id = agent_id
        self.budget_manager = budget_manager

    def _check_budget(self, text: str, ctx: SharedContext) -> bool:
        tokens = estimate_tokens(text)
        ok, violation = self.budget_manager.consume(self.agent_id, tokens)
        if not ok and violation:
            ctx.policy_violations.append({"agent": self.agent_id, "violation": violation, "ts": time.time()})
        return ok

    async def _call_tool_with_retry(self, tool_name: str, ctx: SharedContext, max_retries: int = 2, **kwargs) -> dict:
        result = None
        for attempt in range(1, max_retries + 2):
            start = time.time()
            result = await call_tool(tool_name, **kwargs)
            latency = (time.time() - start) * 1000
            accepted = result.get("error") is None

            ctx.tool_results.append({
                "job_id": ctx.job_id,
                "agent_id": self.agent_id,
                "tool_name": tool_name,
                "attempt": attempt,
                "input": kwargs,
                "output": result,
                "latency_ms": latency,
                "accepted": accepted,
            })

            if accepted or attempt > max_retries:
                break

            # Modify input for retry (append " retry" to string inputs)
            for k, v in kwargs.items():
                if isinstance(v, str):
                    kwargs[k] = v + " (retry)"
                    break

        return result


class DecompositionAgent(BaseAgent):
    async def run(self, ctx: SharedContext) -> AgentOutput:
        prompt_template = get_prompt("decomposition")
        prompt = f"{prompt_template}\n\nQuery: {ctx.original_query}"

        self._check_budget(prompt, ctx)
        raw = await call_llm_json(prompt)
        data = _parse_json(raw)

        subtasks = []
        for t in data.get("subtasks", []):
            subtasks.append(SubTask(
                id=t.get("id", str(uuid.uuid4())[:8]),
                description=t.get("description", ""),
                task_type=t.get("task_type", "analysis"),
                dependencies=t.get("dependencies", []),
            ))

        if not subtasks:
            # Fallback: create one generic task
            subtasks = [SubTask(id="t1", description=ctx.original_query, task_type="analysis")]

        ctx.subtasks = subtasks
        output = AgentOutput(
            agent_id=self.agent_id,
            content=json.dumps({"subtasks": [vars(t) for t in subtasks], "reasoning": data.get("reasoning", "")}),
            token_count=estimate_tokens(raw),
        )
        ctx.agent_outputs[self.agent_id] = output
        ctx.session_history.append({"role": "agent", "agent": self.agent_id, "content": output.content})
        return output


class RetrievalAgent(BaseAgent):
    async def run(self, ctx: SharedContext) -> AgentOutput:
        # Use web_search tool to get chunks
        query = ctx.original_query
        search_result = await self._call_tool_with_retry("web_search", ctx, query=query)

        chunks = []
        if search_result.get("data"):
            for i, r in enumerate(search_result["data"]):
                chunks.append(Chunk(
                    id=f"chunk_{i}",
                    content=r.get("snippet", ""),
                    source=r.get("url", "unknown"),
                    relevance=r.get("relevance", 0.5),
                ))

        # Ensure at least 2 chunks (multi-hop requirement)
        if len(chunks) < 2:
            chunks.append(Chunk(id="chunk_fallback", content=f"General knowledge about: {query}", source="internal", relevance=0.4))

        ctx.retrieved_chunks = chunks

        chunks_text = "\n".join([f"[{c.id}] (source: {c.source}): {c.content}" for c in chunks])
        prompt = f"{get_prompt('retrieval')}\n\nQuery: {query}\n\nChunks:\n{chunks_text}"

        self._check_budget(prompt, ctx)
        raw = await call_llm_json(prompt)
        data = _parse_json(raw)

        output = AgentOutput(
            agent_id=self.agent_id,
            content=data.get("answer", raw),
            chunks_used=[c["chunk_id"] for c in data.get("citations", [])],
            token_count=estimate_tokens(raw),
        )
        ctx.agent_outputs[self.agent_id] = output
        ctx.session_history.append({"role": "agent", "agent": self.agent_id, "content": output.content})
        return output


class CritiqueAgent(BaseAgent):
    async def run(self, ctx: SharedContext) -> AgentOutput:
        # Critique all other agent outputs
        outputs_text = "\n\n".join([
            f"Agent {k}:\n{v.content}"
            for k, v in ctx.agent_outputs.items()
            if k != self.agent_id
        ])

        if not outputs_text:
            outputs_text = "No outputs to critique yet."

        prompt = f"{get_prompt('critique')}\n\nOutputs to review:\n{outputs_text}"
        self._check_budget(prompt, ctx)
        raw = await call_llm_json(prompt)
        data = _parse_json(raw)

        claims = [
            Claim(
                text=c.get("text", ""),
                confidence=float(c.get("confidence", 0.5)),
                flagged=bool(c.get("flagged", False)),
                flag_reason=c.get("flag_reason"),
            )
            for c in data.get("claims", [])
        ]

        output = AgentOutput(
            agent_id=self.agent_id,
            content=json.dumps({"claims": [vars(c) for c in claims]}),
            claims=claims,
            token_count=estimate_tokens(raw),
        )
        ctx.agent_outputs[self.agent_id] = output
        ctx.session_history.append({"role": "agent", "agent": self.agent_id, "content": output.content})
        return output


class SynthesisAgent(BaseAgent):
    async def run(self, ctx: SharedContext) -> AgentOutput:
        # Self-reflection to check contradictions first
        reflection = await self._call_tool_with_retry("self_reflection", ctx, session_history=ctx.session_history)

        all_outputs = "\n\n".join([f"[{k}]: {v.content}" for k, v in ctx.agent_outputs.items()])
        flagged_claims = []
        if "critique" in ctx.agent_outputs:
            critique_data = _parse_json(ctx.agent_outputs["critique"].content)
            flagged_claims = [c for c in critique_data.get("claims", []) if c.get("flagged")]

        prompt = (
            f"{get_prompt('synthesis')}\n\n"
            f"All agent outputs:\n{all_outputs}\n\n"
            f"Flagged claims to resolve: {json.dumps(flagged_claims)}\n"
            f"Contradictions found: {json.dumps(reflection.get('data', {}).get('contradictions', []))}"
        )

        self._check_budget(prompt, ctx)
        raw = await call_llm_json(prompt)
        data = _parse_json(raw)

        answer = data.get("answer", raw)
        ctx.final_answer = answer

        provenance = [
            ProvenanceEntry(
                sentence=p.get("sentence", ""),
                source_agent=p.get("source_agent", "synthesis"),
                source_chunk_ids=p.get("chunk_ids", []),
            )
            for p in data.get("provenance", [])
        ]
        ctx.provenance_map = provenance

        output = AgentOutput(
            agent_id=self.agent_id,
            content=answer,
            token_count=estimate_tokens(raw),
        )
        ctx.agent_outputs[self.agent_id] = output
        ctx.session_history.append({"role": "agent", "agent": self.agent_id, "content": answer})
        return output


class CompressionAgent(BaseAgent):
    async def run(self, ctx: SharedContext) -> str:
        history_text = json.dumps(ctx.session_history)
        prompt = f"{get_prompt('compression')}\n\nHistory:\n{history_text}"
        raw = await call_llm_json(prompt)
        data = _parse_json(raw)
        compressed = data.get("compressed", history_text[:len(history_text) // 2])
        # Replace session history with compressed version
        ctx.session_history = [{"role": "system", "content": compressed}]
        return compressed
