import asyncio
import json
import uuid
import logging

import redis.asyncio as aioredis
from app.config import REDIS_URL
from app.db.models import init_db, AsyncSessionLocal, Job
from app.agents.context import SharedContext
from app.agents.orchestrator import Orchestrator
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [worker] %(message)s")
logger = logging.getLogger(__name__)


async def process_job(job_data: dict):
    job_id = job_data.get("job_id", str(uuid.uuid4()))
    query = job_data.get("query", "")

    ctx = SharedContext(job_id=job_id, original_query=query)
    orchestrator = Orchestrator()

    logger.info(f"Processing job {job_id}: {query[:60]}")

    async def emit(event_type, agent_id, data):
        logger.info(f"[{job_id}] {event_type} | {agent_id} | {str(data)[:80]}")

    await orchestrator.run(ctx, emit=emit)

    async with AsyncSessionLocal() as db:
        job = await db.get(Job, job_id)
        if job:
            job.status = "completed"
            job.final_answer = ctx.final_answer
            job.completed_at = datetime.now(timezone.utc)
            await db.commit()

    logger.info(f"Job {job_id} completed.")


async def main():
    await init_db()
    r = aioredis.from_url(REDIS_URL, decode_responses=True)
    logger.info("Worker started. Waiting for jobs...")

    while True:
        try:
            item = await r.blpop("job_queue", timeout=5)
            if item:
                _, data = item
                job_data = json.loads(data)
                await process_job(job_data)
        except Exception as e:
            logger.error(f"Worker error: {e}")
            await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())