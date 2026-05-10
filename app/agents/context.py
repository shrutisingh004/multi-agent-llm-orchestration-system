"""
Shared context object passed between all agents via the orchestrator.
Agents never call each other directly - they read/write to this context.
"""
from __future__ import annotations
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from app.config import AGENT_BUDGETS


@dataclass
class SubTask:
    id: str
    description: str
    task_type: str  # "retrieval", "analysis", "computation"
    dependencies: list[str] = field(default_factory=list)
    status: str = "pending"  # pending | running | done
    result: Optional[str] = None


@dataclass
class Chunk:
    id: str
    content: str
    source: str
    relevance: float = 1.0


@dataclass
class Claim:
    text: str
    confidence: float
    flagged: bool = False
    flag_reason: Optional[str] = None


@dataclass
class ProvenanceEntry:
    sentence: str
    source_agent: str
    source_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class AgentOutput:
    agent_id: str
    content: str
    claims: list[Claim] = field(default_factory=list)
    chunks_used: list[str] = field(default_factory=list)
    token_count: int = 0


@dataclass
class SharedContext:
    job_id: str
    original_query: str
    subtasks: list[SubTask] = field(default_factory=list)
    retrieved_chunks: list[Chunk] = field(default_factory=list)
    agent_outputs: dict[str, AgentOutput] = field(default_factory=dict)
    tool_results: list[dict] = field(default_factory=list)
    final_answer: Optional[str] = None
    provenance_map: list[ProvenanceEntry] = field(default_factory=list)
    routing_log: list[dict] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=dict)
    policy_violations: list[dict] = field(default_factory=list)
    session_history: list[dict] = field(default_factory=list)  # for self-reflection tool

    def log_routing(self, decision: str, justification: str, next_agent: str):
        self.routing_log.append({
            "timestamp": time.time(),
            "decision": decision,
            "justification": justification,
            "next_agent": next_agent,
        })

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "original_query": self.original_query,
            "subtasks": [vars(t) for t in self.subtasks],
            "retrieved_chunks": [vars(c) for c in self.retrieved_chunks],
            "agent_outputs": {k: vars(v) for k, v in self.agent_outputs.items()},
            "tool_results": self.tool_results,
            "final_answer": self.final_answer,
            "provenance_map": [vars(p) for p in self.provenance_map],
            "routing_log": self.routing_log,
            "token_usage": self.token_usage,
            "policy_violations": self.policy_violations,
        }


class ContextBudgetManager:
    """Tracks token consumption per agent. Enforces budgets."""

    def __init__(self, budgets: dict[str, int] = None):
        self.budgets = budgets or AGENT_BUDGETS.copy()
        self.used: dict[str, int] = {}

    def remaining(self, agent_id: str) -> int:
        budget = self.budgets.get(agent_id, 2000)
        used = self.used.get(agent_id, 0)
        return budget - used

    def consume(self, agent_id: str, tokens: int) -> tuple[bool, Optional[str]]:
        """Returns (ok, violation_msg). If over budget, logs violation."""
        budget = self.budgets.get(agent_id, 2000)
        current = self.used.get(agent_id, 0)
        if current + tokens > budget:
            violation = f"Agent {agent_id} exceeded budget: used {current + tokens}, limit {budget}"
            return False, violation
        self.used[agent_id] = current + tokens
        return True, None

    def get_usage(self) -> dict[str, dict]:
        return {
            agent: {
                "budget": self.budgets.get(agent, 2000),
                "used": self.used.get(agent, 0),
                "remaining": self.remaining(agent),
            }
            for agent in set(list(self.budgets.keys()) + list(self.used.keys()))
        }


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]
