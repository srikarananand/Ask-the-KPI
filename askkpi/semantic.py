"""Warehouse loader and certified-metric engine.

No user text is ever placed in SQL. A metric's SQL is fixed in metrics.yaml; the engine runs it,
then filters the resulting frame with typed parameters (pandas), so injection has no path in.
"""
import sqlite3
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

from . import geo as G

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OPS = {"value", "top_n", "bottom_n", "compare", "series", "change", "cagr"}


def build_warehouse(path=":memory:"):
    con = sqlite3.connect(path)
    retail = pd.read_csv(DATA / "commercial_electricity_by_state.csv",
                         names=["geo", "year", "price", "sales_mkwh"])
    retail.to_sql("retail", con, index=False, if_exists="replace")
    risk = pd.read_csv(DATA / "cost_risk_study_state_2027.csv").rename(columns={
        "steo_growth_%": "steo_growth_pct", "contingency_%": "contingency_pct",
        "spend_2025_$m": "spend_2025_musd"})
    risk.to_sql("risk", con, index=False, if_exists="replace")
    kind = [(c, n, "state") for c, n in G.STATES.items()] + \
           [(c, n, "region") for c, n in G.REGIONS.items()] + [("US", "United States", "national")]
    pd.DataFrame(kind, columns=["geo", "name", "kind"]).to_sql("geo", con, index=False, if_exists="replace")
    return con


@lru_cache(maxsize=1)
def load_dictionary():
    d = yaml.safe_load((ROOT / "askkpi" / "metrics.yaml").read_text())
    return {m["id"]: m for m in d["metrics"]}


@dataclass
class Answer:
    ok: bool
    metric: str = ""
    op: str = ""
    text: str = ""
    rows: list = field(default_factory=list)   # [{geo, year, value}]
    unit: str = ""
    certified: bool = True
    notes: list = field(default_factory=list)  # assumptions made, e.g. default year
    caveat: str = ""
    source: str = ""
    error: str = ""


class Engine:
    def __init__(self, con=None):
        self.con = con or build_warehouse()
        self.dict = load_dictionary()
        self._frames = {}

    def frame(self, mid):
        if mid not in self._frames:
            sql = self.dict[mid]["sql"]
            df = pd.read_sql_query(sql, self.con)
            kinds = dict(self.con.execute("SELECT geo, kind FROM geo").fetchall())
            df["kind"] = df["geo"].map(kinds)
            df["year"] = df["year"].astype(int)
            self._frames[mid] = df
        return self._frames[mid]

    def years(self, mid):
        f = self.frame(mid)
        return int(f.year.min()), int(f.year.max())

    # ---- validation: every route, whatever produced it, passes through here ----
    def validate(self, mid, op, geos=(), year=None, year2=None, n=None):
        if mid not in self.dict:
            return f"unknown metric '{mid}'"
        if op not in OPS:
            return f"unsupported operation '{op}'"
        m = self.dict[mid]
        lo, hi = self.years(mid)
        for y in (year, year2):
            if y is not None and not (lo <= int(y) <= hi):
                return f"{m['name']} is certified for {lo}–{hi}" if lo != hi else f"{m['name']} is certified for {lo} only"
        for g in geos:
            if g not in G.ALL:
                return f"unknown geography '{g}'"
            k = "state" if g in G.STATES else "national" if g == "US" else "region"
            if k not in m["entities"]:
                return f"{m['name']} is not available for {G.name(g)}"
        if op in ("value", "series", "change", "cagr") and len(geos) != 1:
            return "needs exactly one geography"
        if op == "compare" and len(geos) != 2:
            return "needs exactly two geographies"
        if op in ("change", "cagr") and (year is None or year2 is None or year >= year2):
            return "needs a start year and a later end year"
        if op in ("top_n", "bottom_n") and not (n and 1 <= int(n) <= 51):
            return "n must be between 1 and 51"
        return ""

    def run(self, mid, op, geos=(), year=None, year2=None, n=None):
        err = self.validate(mid, op, geos, year, year2, n)
        if err:
            return Answer(False, mid, op, certified=False, error=err)
        m = self.dict[mid]
        f = self.frame(mid)
        lo, hi = self.years(mid)
        d = m["decimals"]
        notes = []
        if year is None and op in ("value", "top_n", "bottom_n", "compare"):
            year = hi
            if lo != hi:
                notes.append(f"No year given; used the latest available ({hi}).")
        unit = m["unit"]
        sp = "" if unit == "%" else " "

        def val(g, y):
            r = f[(f.geo == g) & (f.year == y)]
            return None if r.empty else float(r.value.iloc[0])

        def fmt(v):
            return f"{v:,.{d}f}"

        rows, text = [], ""
        if op == "value":
            v = val(geos[0], year)
            if v is None:
                return Answer(False, mid, op, certified=False, error=f"no data for {G.name(geos[0])} in {year}")
            rows = [dict(geo=geos[0], year=year, value=round(v, 6))]
            text = f"{m['name']} in {G.name(geos[0])}, {year}: {fmt(v)}{sp}{unit}."
        elif op in ("top_n", "bottom_n"):
            s = f[(f.year == year) & (f.kind == "state")].sort_values("value", ascending=(op == "bottom_n")).head(int(n))
            rows = [dict(geo=r.geo, year=year, value=round(r.value, 6)) for r in s.itertuples()]
            word = "Highest" if op == "top_n" else "Lowest"
            items = "; ".join(f"{G.name(r['geo'])} {fmt(r['value'])}" for r in rows)
            text = f"{word} {len(rows)} by {m['name'].lower()}, {year} ({unit}): {items}."
        elif op == "compare":
            a, b = val(geos[0], year), val(geos[1], year)
            if a is None or b is None:
                return Answer(False, mid, op, certified=False, error="no data for one of the geographies")
            rows = [dict(geo=geos[0], year=year, value=round(a, 6)), dict(geo=geos[1], year=year, value=round(b, 6))]
            text = (f"{m['name']}, {year}: {G.name(geos[0])} {fmt(a)}{sp}{unit} vs {G.name(geos[1])} {fmt(b)}{sp}{unit} "
                    f"(difference {fmt(abs(a - b))}{sp}{unit}).")
        elif op == "series":
            s = f[f.geo == geos[0]].sort_values("year")
            if year is not None:
                s = s[s.year >= year]
            if year2 is not None:
                s = s[s.year <= year2]
            rows = [dict(geo=geos[0], year=int(r.year), value=round(r.value, 6)) for r in s.itertuples()]
            text = (f"{m['name']} in {G.name(geos[0])}, {rows[0]['year']}–{rows[-1]['year']} ({unit}): "
                    f"{fmt(rows[0]['value'])} to {fmt(rows[-1]['value'])}.")
        elif op in ("change", "cagr"):
            a, b = val(geos[0], year), val(geos[0], year2)
            if a is None or b is None or a == 0:
                return Answer(False, mid, op, certified=False, error="no data for one of the years")
            if unit == "%":
                diff = b - a
                rows = [dict(geo=geos[0], year=year, value=round(a, 6)), dict(geo=geos[0], year=year2, value=round(b, 6))]
                text = (f"{m['name']} in {G.name(geos[0])}: {fmt(a)}% in {year} and {fmt(b)}% in {year2} "
                        f"(change of {diff:+.{d}f} percentage points).")
            elif op == "change":
                pct = (b / a - 1) * 100
                rows = [dict(geo=geos[0], year=year, value=round(a, 6)), dict(geo=geos[0], year=year2, value=round(b, 6)),
                        dict(geo=geos[0], year=f"{year}-{year2}", value=round(pct, 6))]
                text = (f"{m['name']} in {G.name(geos[0])} went from {fmt(a)} to {fmt(b)}{sp}{unit} between {year} and {year2}, "
                        f"a change of {pct:+.1f}%.")
            else:
                c = ((b / a) ** (1 / (year2 - year)) - 1) * 100
                rows = [dict(geo=geos[0], year=f"{year}-{year2}", value=round(c, 6))]
                text = (f"{m['name']} in {G.name(geos[0])} grew at a compound annual rate of {c:.1f}% "
                        f"between {year} and {year2} ({fmt(a)} to {fmt(b)}{sp}{unit}).")
        return Answer(True, mid, op, text, rows, unit, True, notes, m["caveat"], m["source"])
