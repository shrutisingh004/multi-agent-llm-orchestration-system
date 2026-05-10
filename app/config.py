import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://agent:agentpass@localhost:5432/agentsys")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Sync DB URL for Alembic
SYNC_DATABASE_URL = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://").replace("asyncpg://", "postgresql://")
if "postgresql://" not in SYNC_DATABASE_URL and "postgresql+asyncpg://" not in DATABASE_URL:
    SYNC_DATABASE_URL = DATABASE_URL

# Make sure asyncpg is used for async
if DATABASE_URL.startswith("postgresql://"):
    ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
else:
    ASYNC_DATABASE_URL = DATABASE_URL

# Token budget defaults per agent
AGENT_BUDGETS = {
    "orchestrator": 4000,
    "decomposition": 2000,
    "retrieval": 3000,
    "critique": 2000,
    "synthesis": 3000,
    "compression": 1500,
    "meta": 2000,
}

GEMINI_MODEL = "gemini-2.5-flash"
