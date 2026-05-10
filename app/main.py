import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import REDIS_URL
from app.db.models import init_db, get_db, Job, AgentEvent, ToolCall, EvalRun, EvalResult, PromptRewrite
from app.agents.context import SharedContext, hash_text
from app.agents.orchestrator import Orchestrator
from app.agents.prompts import get_prompt
from app.eval.harness import run_eval
from app.eval.meta_agent import run_meta_agent, apply_approved_rewrite


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="Multi-Agent LLM Orchestration System",
    description="Production-grade multi-agent pipeline with self-improving evaluation loop",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# Request/Response models

class QueryRequest(BaseModel):
    query: str

class ApprovalRequest(BaseModel):
    rewrite_id: int
    decision: str  # "approve" or "reject"

class RerunRequest(BaseModel):
    failed_ids: list[str] = []


# Helpers

async def _save_job_events(db: AsyncSession, ctx: SharedContext, job_id: str):
    """Persist agent events and tool calls to DB."""
    for event in ctx.routing_log:
        db.add(AgentEvent(
            job_id=job_id,
            agent_id="orchestrator",
            event_type="routing",
            data=event,
        ))
    for output in ctx.agent_outputs.values():
        db.add(AgentEvent(
            job_id=job_id,
            agent_id=output.agent_id,
            event_type="agent_output",
            input_hash=hash_text(ctx.original_query),
            output_hash=hash_text(output.content),
            token_count=output.token_count,
            data={"content_preview": output.content[:500]},
        ))
    for tc in ctx.tool_results:
        db.add(ToolCall(
            job_id=job_id,
            agent_id=tc.get("agent_id", "unknown"),
            tool_name=tc.get("tool_name", "unknown"),
            attempt=tc.get("attempt", 1),
            input_data=tc.get("input"),
            output_data=tc.get("output"),
            latency_ms=tc.get("latency_ms", 0),
            accepted=tc.get("accepted", True),
        ))
    for v in ctx.policy_violations:
        db.add(AgentEvent(
            job_id=job_id,
            agent_id=v.get("agent", "unknown"),
            event_type="policy_violation",
            policy_violation=v.get("violation"),
            data=v,
        ))
    await db.commit()


# Endpoint 1: Submit query with SSE streaming

@app.post("/query", summary="Submit a query and receive a real-time SSE stream of agent activity")
async def submit_query(req: QueryRequest, db: AsyncSession = Depends(get_db)):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail={"code": "EMPTY_QUERY", "message": "Query cannot be empty"})

    job_id = str(uuid.uuid4())

    # Persist job record
    job = Job(id=job_id, query=req.query, status="running")
    db.add(job)
    await db.commit()

    events_buffer: list[dict] = []

    async def emit(event_type: str, agent_id: str, data: dict):
        events_buffer.append({
            "event_type": event_type,
            "agent_id": agent_id,
            "data": data,
            "timestamp": time.time(),
        })

    async def stream() -> AsyncIterator[str]:
        ctx = SharedContext(job_id=job_id, original_query=req.query)
        orchestrator = Orchestrator()

        # Kick off pipeline; we'll emit events as they happen
        task = asyncio.create_task(orchestrator.run(ctx, emit=emit))

        # Stream events as they arrive
        sent = 0
        while not task.done() or sent < len(events_buffer):
            while sent < len(events_buffer):
                ev = events_buffer[sent]
                sent += 1
                yield f"data: {json.dumps(ev)}\n\n"
            if not task.done():
                await asyncio.sleep(0.1)

        # Await completion and handle errors
        try:
            await task
        except Exception as e:
            yield f"data: {json.dumps({'event_type': 'error', 'agent_id': 'orchestrator', 'data': {'error': str(e)}})}\n\n"

        # Save to DB
        try:
            await _save_job_events(db, ctx, job_id)
            job.status = "completed"
            job.final_answer = ctx.final_answer
            job.completed_at = datetime.now(timezone.utc)
            await db.commit()
        except Exception:
            pass

        # Final done event
        yield f"data: {json.dumps({'event_type': 'complete', 'job_id': job_id, 'answer': ctx.final_answer})}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"X-Job-Id": job_id, "Cache-Control": "no-cache"},
    )


# Endpoint 2: Full execution trace

@app.get("/trace/{job_id}", summary="Retrieve the full execution trace for a completed job")
async def get_trace(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail={"code": "JOB_NOT_FOUND", "message": f"No job with id {job_id}", "job_id": job_id}
        )

    events_q = await db.execute(
        select(AgentEvent).where(AgentEvent.job_id == job_id).order_by(AgentEvent.timestamp)
    )
    events = events_q.scalars().all()

    tools_q = await db.execute(
        select(ToolCall).where(ToolCall.job_id == job_id).order_by(ToolCall.timestamp)
    )
    tools = tools_q.scalars().all()

    return {
        "job_id": job_id,
        "query": job.query,
        "status": job.status,
        "final_answer": job.final_answer,
        "created_at": job.created_at.isoformat(),
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "agent_events": [
            {
                "id": e.id,
                "agent_id": e.agent_id,
                "event_type": e.event_type,
                "timestamp": e.timestamp.isoformat(),
                "input_hash": e.input_hash,
                "output_hash": e.output_hash,
                "latency_ms": e.latency_ms,
                "token_count": e.token_count,
                "data": e.data,
                "policy_violation": e.policy_violation,
            }
            for e in events
        ],
        "tool_calls": [
            {
                "id": t.id,
                "agent_id": t.agent_id,
                "tool_name": t.tool_name,
                "attempt": t.attempt,
                "input": t.input_data,
                "output": t.output_data,
                "latency_ms": t.latency_ms,
                "accepted": t.accepted,
                "timestamp": t.timestamp.isoformat(),
            }
            for t in tools
        ],
    }


# Endpoint 3: Latest eval summary

@app.get("/eval/latest", summary="Retrieve the latest eval run summary by category and dimension")
async def get_latest_eval(db: AsyncSession = Depends(get_db)):
    q = await db.execute(select(EvalRun).order_by(desc(EvalRun.run_at)).limit(1))
    run = q.scalar_one_or_none()

    if not run:
        raise HTTPException(
            status_code=404,
            detail={"code": "NO_EVAL_RUN", "message": "No eval runs found. POST /eval/rerun to run evaluation."}
        )

    results_q = await db.execute(select(EvalResult).where(EvalResult.run_id == run.id))
    results = results_q.scalars().all()

    pending_rewrites_q = await db.execute(
        select(PromptRewrite).where(PromptRewrite.status == "pending").order_by(desc(PromptRewrite.created_at))
    )
    pending_rewrites = pending_rewrites_q.scalars().all()

    return {
        "run_id": run.id,
        "run_at": run.run_at.isoformat(),
        "summary": run.summary,
        "results": [
            {
                "test_case_id": r.test_case_id,
                "category": r.category,
                "query": r.query[:200],
                "answer_preview": (r.answer or "")[:300],
                "scores": r.scores,
                "job_id": r.job_id,
            }
            for r in results
        ],
        "pending_rewrites": [
            {
                "id": pr.id,
                "agent_id": pr.agent_id,
                "dimension": pr.dimension,
                "justification": pr.justification,
                "diff_preview": pr.diff[:300],
                "created_at": pr.created_at.isoformat(),
            }
            for pr in pending_rewrites
        ],
    }


# Endpoint 4: Approve/reject prompt rewrite

@app.post("/rewrite/approve", summary="Human approval or rejection of a pending prompt rewrite")
async def approve_rewrite(req: ApprovalRequest, db: AsyncSession = Depends(get_db)):
    if req.decision not in ("approve", "reject"):
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_DECISION", "message": "decision must be 'approve' or 'reject'"}
        )

    rewrite = await db.get(PromptRewrite, req.rewrite_id)
    if not rewrite:
        raise HTTPException(
            status_code=404,
            detail={"code": "REWRITE_NOT_FOUND", "message": f"No rewrite with id {req.rewrite_id}"}
        )

    if rewrite.status != "pending":
        raise HTTPException(
            status_code=409,
            detail={"code": "ALREADY_REVIEWED", "message": f"Rewrite {req.rewrite_id} already {rewrite.status}"}
        )

    rewrite.status = req.decision + "d"  # "approved" or "rejected"
    rewrite.reviewed_at = datetime.now(timezone.utc)

    if req.decision == "approve":
        apply_approved_rewrite(rewrite.agent_id, rewrite.proposed_prompt)

    await db.commit()
    return {
        "rewrite_id": req.rewrite_id,
        "decision": req.decision,
        "agent_id": rewrite.agent_id,
        "message": f"Prompt for '{rewrite.agent_id}' {'updated' if req.decision == 'approve' else 'unchanged'}",
    }


# Endpoint 5: Targeted re-eval

@app.post("/eval/rerun", summary="Trigger evaluation (full or on failed cases) using latest approved prompts")
async def trigger_eval(req: RerunRequest, db: AsyncSession = Depends(get_db)):
    failed_ids = req.failed_ids or None  # None = run all 15

    results, summary = await run_eval(failed_ids=failed_ids)

    # Save eval run
    run = EvalRun(summary=summary)
    db.add(run)
    await db.flush()

    # Save individual results
    for r in results:
        prompts_snapshot = {agent: get_prompt(agent) for agent in ["orchestrator", "decomposition", "retrieval", "critique", "synthesis"]}
        db.add(EvalResult(
            run_id=run.id,
            job_id=r.job_id,
            test_case_id=r.test_case_id,
            category=r.category,
            query=r.query,
            answer=r.answer,
            scores=[{"name": s.name, "score": s.score, "justification": s.justification} for s in r.scores],
            prompts_snapshot=prompts_snapshot,
        ))

    await db.flush()

    # Run meta-agent for self-improving loop
    results_dicts = [r.to_dict() for r in results]
    worst_dim = summary.get("by_dimension", {})
    worst_dim_name = min(worst_dim, key=lambda d: worst_dim[d]["avg"]) if worst_dim else "correctness"

    try:
        meta_result = await run_meta_agent(results_dicts, worst_dim_name)
        if "error" not in meta_result:
            rewrite = PromptRewrite(
                eval_run_id=run.id,
                agent_id=meta_result["agent_id"],
                dimension=meta_result["dimension"],
                original_prompt=meta_result["original_prompt"],
                proposed_prompt=meta_result["proposed_prompt"],
                diff=meta_result["diff"],
                justification=meta_result["justification"],
            )
            db.add(rewrite)
    except Exception as e:
        meta_result = {"error": str(e)}

    await db.commit()

    return {
        "run_id": run.id,
        "cases_evaluated": len(results),
        "summary": summary,
        "prompt_rewrite_proposed": "error" not in meta_result,
        "rewrite_agent": meta_result.get("agent_id"),
        "rewrite_dimension": meta_result.get("dimension"),
        "results": [r.to_dict() for r in results],
    }