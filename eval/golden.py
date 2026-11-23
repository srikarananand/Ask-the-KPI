"""Independent reference calculations for the eval harness.

Deliberately does NOT use askkpi.semantic or the metric SQL: it recomputes each expected value
straight from the raw CSV snapshots with plain pandas, so a bug in a metric definition shows up
as an answer mismatch instead of passing silently.
"""
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
R = pd.read_csv(DATA / "commercial_electricity_by_state.csv", names=["geo", "year", "price", "sales"])
K = pd.read_csv(DATA / "cost_risk_study_state_2027.csv").set_index("state")
STATES = set(K.index)
LATEST = 2025
RISK = {"budget_overrun_p90": "p90_overrun", "recommended_contingency": "contingency_%",
        "base_case_price_2027": "base_2027", "high_case_price_2027": "high_case_2027",
        "outlook_growth": "steo_growth_%"}


def level(metric, geo, year):
    """Value of a metric for one geography-year, or None."""
    if metric in RISK:
        return float(K.loc[geo, RISK[metric]]) if geo in K.index and year == 2027 else None
    a = R[(R.geo == geo) & (R.year == year)]
    if a.empty:
        return None
    p, s = float(a.price.iloc[0]), float(a.sales.iloc[0])
    if metric == "commercial_price":
        return p
    if metric == "commercial_sales":
        return s
    if metric == "commercial_spend":
        return p * s / 100
    if metric == "price_yoy":
        b = R[(R.geo == geo) & (R.year == year - 1)]
        return None if b.empty else (p / float(b.price.iloc[0]) - 1) * 100
    if metric == "price_vs_national":
        u = R[(R.geo == "US") & (R.year == year)]
        return None if u.empty or geo == "US" else (p / float(u.price.iloc[0]) - 1) * 100
    raise KeyError(metric)


def expected(spec):
    """Expected numeric values (list) for a route spec, mirroring the documented semantics."""
    m, op, geos = spec["metric"], spec["op"], spec.get("geos", [])
    year, year2, n = spec.get("year"), spec.get("year2"), spec.get("n")
    hi = 2027 if m in RISK else LATEST
    if op in ("value", "top_n", "bottom_n", "compare") and year is None:
        year = hi
    if op == "value":
        return [level(m, geos[0], year)]
    if op == "compare":
        return [level(m, g, year) for g in geos]
    if op in ("top_n", "bottom_n"):
        vals = [(g, level(m, g, year)) for g in sorted(STATES)]
        vals = [(g, v) for g, v in vals if v is not None]
        vals.sort(key=lambda t: t[1], reverse=(op == "top_n"))
        return [v for _, v in vals[:n]]
    if op == "series":
        ys = [y for y in range(2005, hi + 1) if (year is None or y >= year) and (year2 is None or y <= year2)]
        return [level(m, geos[0], y) for y in ys if level(m, geos[0], y) is not None]
    a, b = level(m, geos[0], year), level(m, geos[0], year2)
    if m in ("price_yoy", "price_vs_national"):
        return [a, b]
    if op == "change":
        return [a, b, (b / a - 1) * 100]
    if op == "cagr":
        return [((b / a) ** (1 / (year2 - year)) - 1) * 100]
    raise KeyError(op)
