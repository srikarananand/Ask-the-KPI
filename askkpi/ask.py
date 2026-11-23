"""Orchestration: question -> router -> validator -> certified metric -> governed answer."""
from . import geo as G
from .router import RuleRouter, Route
from .semantic import Engine

SUGGEST = ("Try asking about commercial electricity price, sales or spend by state or region (2005–2025), "
           "or the 2027 base case, overrun risk and contingency by state.")
REFUSAL_TEXT = {
    "unsafe_or_instruction_override": "I only answer from certified metrics and can't follow instructions to change that.",
    "advice_not_certified": "I can report certified metrics but I don't give purchasing, hedging or investment advice.",
    "sector_not_in_warehouse": "The warehouse covers the commercial sector only.",
    "causal_explanation_not_certified": "I report what the data shows, not why; causal explanations aren't certified.",
    "beyond_forecast_horizon": "Certified figures end at the 2027 base case; there is nothing certified beyond that.",
    "topic_not_in_warehouse": "That topic isn't in the warehouse.",
    "no_matching_certified_metric": "No certified metric matches that question.",
    "year_not_certified": "Only 2005–2025 actuals and the 2027 base case are certified; 2026 is not.",
    "unparseable_model_output": "I could not route that question to a certified metric.",
    "invalid_model_output": "I could not route that question to a certified metric.",
}
CLARIFY_TEXT = {
    "which_geography": "Which state, region or the U.S. total do you mean?",
    "more_than_one_metric": "That mixes more than one metric. Which one first: price, sales or spend?",
    "compare_two_at_a_time": "I can compare two geographies at a time. Which two?",
    "one_geography_at_a_time": "Please pick one geography for that question.",
    "use_top_n_for_many_states": "For many states, ask for the top or bottom N (for example, the five highest).",
    "which_metric_to_define": "Which metric do you want defined?",
}


class Assistant:
    def __init__(self, router=None, engine=None):
        self.engine = engine or Engine()
        self.router = router or RuleRouter()

    def ask(self, q):
        try:
            r = self.router.route(q)
        except Exception as e:  # a router failure must never become an answer
            r = Route("refuse", reason="router_error")
        out = {"question": q, "router": self.router.name, "route": r.__dict__.copy()}
        if r.status == "list":
            lines = [f"{m['name']} [{m['unit']}]" for m in self.engine.dict.values()]
            out.update(status="list", text="Certified metrics: " + "; ".join(lines) + ".")
        elif r.status == "define":
            m = self.engine.dict.get(r.metric)
            if not m:
                out.update(status="refuse", text=REFUSAL_TEXT["no_matching_certified_metric"], reason="unknown_metric")
            else:
                out.update(status="define", text=f"{m['name']}: {m['definition']} Unit: {m['unit']}. Source: {m['source']}. Caveat: {m['caveat']}")
        elif r.status == "refuse":
            out.update(status="refuse", reason=r.reason, text=f"{REFUSAL_TEXT.get(r.reason, 'Out of scope.')} {SUGGEST}")
        elif r.status == "clarify":
            out.update(status="clarify", reason=r.reason, text=CLARIFY_TEXT.get(r.reason, "Could you be more specific?"))
        else:
            err = self.engine.validate(r.metric, r.op, r.geos, r.year, r.year2, r.n)
            if err:
                out.update(status="refuse", reason="failed_validation", text=f"Can't answer that from certified metrics: {err}. {SUGGEST}")
            else:
                a = self.engine.run(r.metric, r.op, r.geos, r.year, r.year2, r.n)
                if not a.ok:
                    out.update(status="refuse", reason="no_data", text=f"Can't answer that: {a.error}.")
                else:
                    out.update(status="answer", text=a.text, rows=a.rows, unit=a.unit, notes=a.notes,
                               caveat=a.caveat, source=a.source, metric=a.metric, op=a.op)
        return out
