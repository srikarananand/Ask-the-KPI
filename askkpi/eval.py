"""Evaluation harness: score a router on a labeled question set.

    python -m askkpi.eval --set dev --router rule
    python -m askkpi.eval --set holdout --router gemini      # needs GEMINI_API_KEY

Each question carries the expected status and, for answerable questions, the expected route.
Expected VALUES come from eval/golden.py (independent pandas on the raw CSVs), not from the engine.
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import golden  # noqa: E402
from .ask import Assistant  # noqa: E402
from .router import GeminiRouter, RuleRouter  # noqa: E402

TOL = 1e-4


def close(a, b):
    return a is not None and b is not None and abs(a - b) <= TOL * max(1.0, abs(b))


def score_one(item, out):
    exp = item["expect"]
    es, st = exp["status"], out["status"]
    res = {"id": item["id"], "category": item["category"], "q": item["q"], "expected": es, "actual": st,
           "text": out["text"], "reason": out.get("reason", "")}
    if es == "answer":
        want = golden.expected(exp)
        got = [r["value"] for r in out.get("rows", [])] if st == "answer" else []
        ok = st == "answer" and len(got) == len(want) and all(close(g, w) for g, w in zip(got, want))
        r = out["route"]
        res["route_exact"] = (st == "answer" and r["metric"] == exp["metric"] and r["op"] == exp["op"]
                              and sorted(r["geos"]) == sorted(exp.get("geos", []))
                              and r["year"] == exp.get("year") and r["year2"] == exp.get("year2")
                              and (exp["op"] not in ("top_n", "bottom_n") or r["n"] == exp.get("n")))
        res["correct"] = ok
        res["outcome"] = "correct" if ok else ("wrong_answer" if st == "answer" else "false_refusal")
    elif es in ("refuse", "clarify"):
        res["correct"] = st in ("refuse", "clarify")
        res["exact_status"] = st == es
        res["outcome"] = "correct" if res["correct"] else "missed_refusal"
    else:  # list / define
        ok = st == es and (es == "list" or out["route"]["metric"] == exp.get("metric"))
        res["correct"] = ok
        res["outcome"] = "correct" if ok else "wrong_meta"
    return res


def summarize(results):
    n = len(results)
    ans = [r for r in results if r["expected"] == "answer"]
    non = [r for r in results if r["expected"] in ("refuse", "clarify")]
    meta = [r for r in results if r["expected"] in ("list", "define")]
    answered = [r for r in results if r["actual"] == "answer"]
    wrong = [r for r in results if r["outcome"] in ("wrong_answer", "missed_refusal")]
    pct = lambda a, b: round(100.0 * a / b, 1) if b else None
    cats = {}
    for r in results:
        c = cats.setdefault(r["category"], [0, 0])
        c[0] += 1
        c[1] += r["correct"]
    return {
        "n_questions": n,
        "n_answerable": len(ans), "n_should_decline": len(non), "n_meta": len(meta),
        "answer_accuracy_pct": pct(sum(r["correct"] for r in ans), len(ans)),
        "route_exact_pct": pct(sum(r.get("route_exact", False) for r in ans), len(ans)),
        "coverage_pct": pct(sum(r["actual"] == "answer" for r in ans), len(ans)),
        "precision_of_answers_pct": pct(sum(r["correct"] for r in answered if r["expected"] == "answer"), len(answered)),
        "correct_decline_pct": pct(sum(r["correct"] for r in non), len(non)),
        "false_refusal_pct": pct(sum(r["outcome"] == "false_refusal" for r in ans), len(ans)),
        "wrong_answer_rate_pct": pct(len(wrong), n),
        "meta_correct_pct": pct(sum(r["correct"] for r in meta), len(meta)),
        "overall_correct_pct": pct(sum(r["correct"] for r in results), n),
        "by_category": {k: {"n": v[0], "correct": v[1]} for k, v in sorted(cats.items())},
    }


def run(set_name, router_name="rule"):
    items = yaml.safe_load((ROOT / "eval" / f"{set_name}.yaml").read_text())["questions"]
    router = RuleRouter() if router_name == "rule" else GeminiRouter()
    bot = Assistant(router=router)
    results = [score_one(it, bot.ask(it["q"])) for it in items]
    return {"set": set_name, "router": router.name, "summary": summarize(results), "results": results}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev")
    ap.add_argument("--router", default="rule", choices=["rule", "gemini"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    r = run(a.set, a.router)
    print(json.dumps(r["summary"], indent=1))
    for x in r["results"]:
        if not x["correct"]:
            print(f"FAIL [{x['category']}] {x['q']!r}: expected {x['expected']} got {x['actual']} ({x['reason']}) :: {x['text'][:110]}")
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1))
