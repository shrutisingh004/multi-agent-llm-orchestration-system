import asyncio
import io
import json
import time
import contextlib
from typing import Any

# ---- Tool failure contracts ----
# timeout:       {"error": "timeout", "data": None, "latency_ms": X}
# empty_results: {"error": "empty", "data": [], "latency_ms": X}
# malformed:     {"error": "malformed_input", "data": None, "latency_ms": X}


def _result(data: Any, error: str = None, latency_ms: float = 0) -> dict:
    return {"data": data, "error": error, "latency_ms": latency_ms}


async def web_search(query: str, timeout: float = 5.0) -> dict:
    start = time.time()
    if not isinstance(query, str) or not query.strip():
        return _result(None, "malformed_input", 0)

    # Stub: return relevant fake results based on query keywords
    await asyncio.sleep(0.1)  # simulate network
    latency = (time.time() - start) * 1000

    # Build stub results
    results = [
        {
            "title": f"Result 1 for: {query[:40]}",
            "url": f"https://example.com/result1?q={query[:20].replace(' ', '+')}",
            "snippet": f"This article discusses {query}. Key points include recent developments and analysis.",
            "relevance": 0.92,
        },
        {
            "title": f"Result 2 for: {query[:40]}",
            "url": f"https://news.example.com/article2",
            "snippet": f"An overview of {query} with supporting evidence from multiple sources.",
            "relevance": 0.78,
        },
        {
            "title": f"Background reference: {query[:30]}",
            "url": f"https://wiki.example.com/{query[:15].replace(' ', '_')}",
            "snippet": f"Historical context and definitions related to {query}.",
            "relevance": 0.65,
        },
    ]
    if not results:
        return _result([], "empty", latency)
    return _result(results, None, latency)


async def code_execution(code: str, timeout: float = 10.0) -> dict:
    start = time.time()
    if not isinstance(code, str) or not code.strip():
        return _result(None, "malformed_input", 0)

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    exit_code = 0

    # Restricted globals - block dangerous imports
    restricted_globals = {
        "__builtins__": {
            "print": lambda *a, **kw: print(*a, **kw, file=stdout_buf),
            "range": range, "len": len, "int": int, "float": float,
            "str": str, "list": list, "dict": dict, "sum": sum,
            "min": min, "max": max, "abs": abs, "round": round,
            "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
            "True": True, "False": False, "None": None,
        }
    }

    try:
        async def _run():
            with contextlib.redirect_stdout(stdout_buf):
                exec(compile(code, "<sandbox>", "exec"), restricted_globals)

        await asyncio.wait_for(_run(), timeout=timeout)
    except asyncio.TimeoutError:
        return _result(None, "timeout", (time.time() - start) * 1000)
    except Exception as e:
        stderr_buf.write(str(e))
        exit_code = 1

    latency = (time.time() - start) * 1000
    return _result(
        {"stdout": stdout_buf.getvalue(), "stderr": stderr_buf.getvalue(), "exit_code": exit_code},
        "runtime_error" if exit_code != 0 else None,
        latency,
    )


# In-memory stub database
_STUB_DB = {
    "employees": [
        {"id": 1, "name": "Alice", "dept": "Engineering", "salary": 95000},
        {"id": 2, "name": "Bob", "dept": "Marketing", "salary": 75000},
        {"id": 3, "name": "Carol", "dept": "Engineering", "salary": 102000},
    ],
    "products": [
        {"id": 1, "name": "Widget A", "price": 29.99, "stock": 150},
        {"id": 2, "name": "Widget B", "price": 49.99, "stock": 30},
    ],
    "sales": [
        {"product_id": 1, "qty": 50, "date": "2024-01"},
        {"product_id": 2, "qty": 12, "date": "2024-01"},
    ],
}


async def structured_data_lookup(nl_query: str) -> dict:
    start = time.time()
    if not isinstance(nl_query, str) or not nl_query.strip():
        return _result(None, "malformed_input", 0)

    query_lower = nl_query.lower()
    results = []

    # Simple NL → lookup heuristics
    if "employee" in query_lower or "salary" in query_lower or "staff" in query_lower:
        data = _STUB_DB["employees"]
        if "engineering" in query_lower:
            data = [r for r in data if r["dept"] == "Engineering"]
        if "average" in query_lower or "avg" in query_lower:
            avg = sum(r["salary"] for r in data) / len(data) if data else 0
            results = [{"average_salary": avg, "count": len(data)}]
        else:
            results = data
    elif "product" in query_lower or "price" in query_lower or "stock" in query_lower:
        results = _STUB_DB["products"]
        if "low stock" in query_lower or "out of stock" in query_lower:
            results = [r for r in results if r["stock"] < 50]
    elif "sale" in query_lower or "revenue" in query_lower:
        results = _STUB_DB["sales"]
    else:
        results = list(_STUB_DB.keys())  # return table list as fallback

    latency = (time.time() - start) * 1000
    if not results:
        return _result([], "empty", latency)
    return _result(results, None, latency)


async def self_reflection(session_history: list[dict], focus: str = "") -> dict:
    start = time.time()
    if not isinstance(session_history, list):
        return _result(None, "malformed_input", 0)
    if not session_history:
        return _result(None, "empty", 0)

    # Find contradictions in history entries
    contradictions = []
    outputs = [h for h in session_history if h.get("role") == "agent"]

    for i, a in enumerate(outputs):
        for j, b in enumerate(outputs):
            if i >= j:
                continue
            a_text = str(a.get("content", "")).lower()
            b_text = str(b.get("content", "")).lower()

            # Simple heuristic: look for direct negations
            negation_pairs = [("is true", "is false"), ("yes", "no"), ("correct", "incorrect"),
                               ("does", "does not"), ("can", "cannot"), ("will", "will not")]
            for pos, neg in negation_pairs:
                if pos in a_text and neg in b_text:
                    contradictions.append({
                        "entry_a": i,
                        "entry_b": j,
                        "pattern": f"'{pos}' vs '{neg}'",
                        "context_a": a.get("content", "")[:100],
                        "context_b": b.get("content", "")[:100],
                    })

    latency = (time.time() - start) * 1000
    return _result({"contradictions": contradictions, "history_length": len(session_history)}, None, latency)


TOOLS = {
    "web_search": web_search,
    "code_execution": code_execution,
    "structured_data_lookup": structured_data_lookup,
    "self_reflection": self_reflection,
}


async def call_tool(tool_name: str, **kwargs) -> dict:
    """Unified tool dispatcher."""
    if tool_name not in TOOLS:
        return _result(None, "unknown_tool", 0)
    return await TOOLS[tool_name](**kwargs)