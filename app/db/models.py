from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Text, Float, Integer, DateTime, JSON, ForeignKey, Boolean
from datetime import datetime, timezone
from typing import Optional
from app.config import ASYNC_DATABASE_URL

engine = create_async_engine(ASYNC_DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    query: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    final_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    events: Mapped[list["AgentEvent"]] = relationship("AgentEvent", back_populates="job")

class AgentEvent(Base):
    __tablename__ = "agent_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), ForeignKey("jobs.id"))
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    agent_id: Mapped[str] = mapped_column(String(64))
    event_type: Mapped[str] = mapped_column(String(64))
    input_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    output_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    token_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    policy_violation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    job: Mapped["Job"] = relationship("Job", back_populates="events")

class ToolCall(Base):
    __tablename__ = "tool_calls"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), ForeignKey("jobs.id"))
    agent_id: Mapped[str] = mapped_column(String(64))
    tool_name: Mapped[str] = mapped_column(String(64))
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    input_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    output_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, default=0)
    accepted: Mapped[bool] = mapped_column(Boolean, default=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

class EvalRun(Base):
    __tablename__ = "eval_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    summary: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    results: Mapped[list["EvalResult"]] = relationship("EvalResult", back_populates="run")

class EvalResult(Base):
    __tablename__ = "eval_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(Integer, ForeignKey("eval_runs.id"))
    job_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    test_case_id: Mapped[str] = mapped_column(String(64))
    category: Mapped[str] = mapped_column(String(64))
    query: Mapped[str] = mapped_column(Text)
    answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    scores: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    prompts_snapshot: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    run: Mapped["EvalRun"] = relationship("EvalRun", back_populates="results")

class PromptRewrite(Base):
    __tablename__ = "prompt_rewrites"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    eval_run_id: Mapped[int] = mapped_column(Integer, ForeignKey("eval_runs.id"))
    agent_id: Mapped[str] = mapped_column(String(64))
    dimension: Mapped[str] = mapped_column(String(64))
    original_prompt: Mapped[str] = mapped_column(Text)
    proposed_prompt: Mapped[str] = mapped_column(Text)
    diff: Mapped[str] = mapped_column(Text)
    justification: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    delta: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
