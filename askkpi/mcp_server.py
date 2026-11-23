"""MCP server exposing the governed KPI layer to any MCP client (stdio).

Tools never accept SQL. `ask_kpi` routes a question; `query_metric` takes typed parameters.
Run:  python -m askkpi.mcp_server
"""
from mcp.server.mcpserver import MCPServer

from .ask import Assistant

mcp = MCPServer("ask-the-kpi")
_bot = Assistant()


@mcp.tool()
def list_metrics() -> list[dict]:
    """List the certified metrics (id, name, unit, entities, years)."""
    out = []
    for mid, m in _bot.engine.dict.items():
        lo, hi = _bot.engine.years(mid)
        out.append({"id": mid, "name": m["name"], "unit": m["unit"], "entities": m["entities"], "years": [lo, hi]})
    return out


@mcp.tool()
def describe_metric(metric_id: str) -> dict:
    """Definition, unit, source, owner and caveat for one certified metric."""
    m = _bot.engine.dict.get(metric_id)
    if not m:
        return {"error": f"unknown metric '{metric_id}'"}
    return {k: m[k] for k in ("name", "unit", "definition", "source", "owner", "caveat", "entities")}


@mcp.tool()
def ask_kpi(question: str) -> dict:
    """Answer a business question from certified metrics, or decline / ask for clarification."""
    return _bot.ask(question)


@mcp.tool()
def query_metric(metric_id: str, op: str = "value", geos: list[str] | None = None, year: int | None = None,
                 year2: int | None = None, n: int | None = None) -> dict:
    """Run a certified metric with typed parameters. op: value, top_n, bottom_n, compare, series, change, cagr."""
    a = _bot.engine.run(metric_id, op, tuple(geos or ()), year, year2, n)
    return a.__dict__


if __name__ == "__main__":
    mcp.run()
