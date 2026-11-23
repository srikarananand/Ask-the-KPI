"""Render docs/index.html from outputs/eval_*.json (answer-first consulting-style report)."""
import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "outputs"
load = lambda n: json.load(open(O / f"{n}.json"))
dev, h1, h2 = load("eval_dev"), load("eval_holdout_v1"), load("eval_holdout2_v2")

# --- cause of every failure on the two unseen first-run sets (assigned by reading each failure) ---
CAUSE = {
    "h01": "Vocabulary gap", "h07": "Input normalization", "h11": "Synonym overlap", "h27": "Unsupported grain",
    "h41": "Synonym overlap", "h44": "Vocabulary gap", "h47": "Meta-question phrasing",
    "h48": "Meta-question phrasing", "h49": "Meta-question phrasing",
    "k03": "Vocabulary gap", "k05": "Vocabulary gap", "k25": "Vocabulary gap", "k33": "Meta-question phrasing",
    "k34": "Vocabulary gap",
}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def counts(r):
    res = r["results"]
    ans = [x for x in res if x["expected"] == "answer"]
    non = [x for x in res if x["expected"] in ("refuse", "clarify")]
    return dict(n=len(res), na=len(ans), nc=sum(x["correct"] for x in ans), nn=len(non),
                nd=sum(x["correct"] for x in non), wrong=sum(x["outcome"] in ("wrong_answer", "missed_refusal") for x in res),
                fr=sum(x["outcome"] == "false_refusal" for x in ans),
                meta=sum(x["expected"] in ("list", "define") for x in res),
                metac=sum(x["correct"] for x in res if x["expected"] in ("list", "define")),
                tot=sum(x["correct"] for x in res))


cd, c1, c2 = counts(dev), counts(h1), counts(h2)
pool = {k: c1[k] + c2[k] for k in c1}
pct = lambda a, b: f"{100 * a / b:.1f}%"
lo, hi = wilson(pool["nc"], pool["na"])
fails = [x for r in (h1, h2) for x in r["results"] if not x["correct"]]
assert len(fails) == len(CAUSE) and all(x["id"] in CAUSE for x in fails), "label every failure"
wrongs = [x for x in fails if x["outcome"] in ("wrong_answer", "missed_refusal")]
cause_tab = {}
for x in fails:
    c = cause_tab.setdefault(CAUSE[x["id"]], {"safe": 0, "wrong": 0})
    c["wrong" if x in wrongs else "safe"] += 1
order = sorted(cause_tab, key=lambda k: -(cause_tab[k]["safe"] + cause_tab[k]["wrong"]))

esc = html.escape
# --- exhibit 1: architecture ---
def box(x, y, w, t1, t2, fill="#fff", stroke="#1f5fbf", tc="#14213d"):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="64" rx="6" fill="{fill}" stroke="{stroke}"/>'
            f'<text x="{x + w / 2}" y="{y + 27}" text-anchor="middle" style="fill:{tc};font-weight:700;font-size:13px">{t1}</text>'
            f'<text x="{x + w / 2}" y="{y + 46}" text-anchor="middle" style="fill:{tc};font-size:11px">{t2}</text>')
arch = ('<svg viewBox="0 0 900 230" width="100%" role="img" aria-label="Architecture: question, router, validator, certified metric on warehouse, governed answer; declines branch off the validator">'
        + box(0, 20, 130, "Question", "plain English")
        + box(170, 20, 150, "Router", "rule-based or LLM")
        + box(360, 20, 150, "Validator", "metric, geo, years, op")
        + box(550, 20, 160, "Certified metric", "fixed SQL, golden tests")
        + box(750, 20, 150, "Governed answer", "value, definition, caveat", fill="#1f5fbf", stroke="#1f5fbf", tc="#fff")
        + box(360, 140, 150, "Decline / clarify", "out of scope or unclear", fill="#fbeeea", stroke="#b4391f")
        + '<g stroke="#5b6577" stroke-width="1.6" fill="none"><path d="M130 52h40M320 52h40M510 52h40M710 52h40"/><path d="M435 84v56"/></g>'
        + '<g fill="#5b6577"><path d="M170 52l-7-4v8zM360 52l-7-4v8zM550 52l-7-4v8zM750 52l-7-4v8zM435 140l-4-7h8z"/></g>'
        + '<text x="170" y="120" style="font-size:11px">Router outputs a request, never a number or SQL</text>'
        + '<text x="550" y="120" style="font-size:11px">Only fixed, reviewed SQL touches the warehouse</text>'
        + '<text x="520" y="178" style="font-size:11px">Every answer states how the question was interpreted</text></svg>')

# --- exhibit 3: failure causes (stacked bars) ---
mx = max(cause_tab[k]["safe"] + cause_tab[k]["wrong"] for k in order)
rows = []
for i, k in enumerate(order):
    y = i * 36 + 6
    s, w = cause_tab[k]["safe"], cause_tab[k]["wrong"]
    sc = 520 / (mx + 0.001)
    rows.append(f'<text x="0" y="{y + 16}" style="fill:#14213d">{esc(k)}</text>'
                f'<rect x="230" y="{y}" width="{s * sc:.0f}" height="22" rx="3" fill="#8aa6d4"/>'
                f'<rect x="{230 + s * sc:.0f}" y="{y}" width="{w * sc:.0f}" height="22" rx="3" fill="#b4391f"/>'
                f'<text x="{238 + (s + w) * sc:.0f}" y="{y + 16}" style="fill:#14213d">{s + w}</text>')
fchart = (f'<svg viewBox="0 0 900 {len(order) * 36 + 36}" width="100%" role="img" aria-label="Failure causes">' + "".join(rows)
          + f'<rect x="230" y="{len(order) * 36 + 8}" width="12" height="12" fill="#8aa6d4"/><text x="248" y="{len(order) * 36 + 18}">Safe miss (declined or asked)</text>'
          + f'<rect x="440" y="{len(order) * 36 + 8}" width="12" height="12" fill="#b4391f"/><text x="458" y="{len(order) * 36 + 18}">Confidently wrong answer</text></svg>')

# --- exhibit 2: scorecard ---
def srow(label, c, note, tuned=False):
    return (f"<tr><td class='l'><b>{label}</b><br><span class='note'>{note}</span></td><td>{c['n']}</td>"
            f"<td>{pct(c['nc'], c['na'])}<br><span class='note'>{c['nc']}/{c['na']}</span></td>"
            f"<td>{pct(c['nd'], c['nn'])}<br><span class='note'>{c['nd']}/{c['nn']}</span></td>"
            f"<td>{pct(c['wrong'], c['n'])}<br><span class='note'>{c['wrong']}/{c['n']}</span></td>"
            f"<td>{pct(c['fr'], c['na'])}<br><span class='note'>{c['fr']}/{c['na']}</span></td></tr>")
score = ("<table><tr><th class='l'>Question set</th><th>Questions</th><th>Answer accuracy</th><th>Correct declines</th>"
         "<th>Wrong-answer rate</th><th>False refusals</th></tr>"
         + srow("Dev set", cd, "Router tuned on this set; shown for reference only")
         + srow("Unseen set 1, first run", c1, "Router v1, written after the router was frozen")
         + srow("Unseen set 2, first run", c2, "Router v2 (generic fixes for set 1 failures), new questions")
         + srow("Unseen sets pooled", pool, "Headline figures; mixes v1 and v2 first runs") + "</table>")

# --- exhibit 4: metric catalog ---
import yaml
D = yaml.safe_load((ROOT / "askkpi/metrics.yaml").read_text())["metrics"]
cat = "<table><tr><th class='l'>Certified metric</th><th class='l'>Definition</th><th>Unit</th><th class='l'>Caveat</th></tr>" + "".join(
    f"<tr><td class='l'><b>{esc(m['name'])}</b><br><span class='note'>{', '.join(m['entities'])}</span></td>"
    f"<td class='l'>{esc(m['definition'])}</td><td>{esc(m['unit'])}</td><td class='l'>{esc(m['caveat'])}</td></tr>" for m in D) + "</table>"

# --- exhibit 5: transcripts ---
def find(r, i):
    return next(x for x in r["results"] if x["id"] == i)
T = [("Governed answer", find(dev, "d15")), ("Change over time", find(dev, "d28")),
     ("Budget input with caveat", find(dev, "d38")), ("Declines advice", find(h1, "h59")),
     ("Asks instead of guessing", find(dev, "d46")), ("Refuses instruction override", find(dev, "d58")),
     ("Wrong answer, shown honestly", find(h2, "k05"))]
tr = "<table><tr><th class='l'>Pattern</th><th class='l'>Question</th><th class='l'>Response</th></tr>" + "".join(
    f"<tr><td class='l'>{esc(a)}</td><td class='l'>{esc(x['q'])}</td><td class='l'>{esc(x['text'][:230])}</td></tr>" for a, x in T) + "</table>"
tr_note = ("In the last row the router missed the hyphenated phrase 'year-over-year' and returned the price level instead of the "
           "year-over-year change. This is the failure type the interpretation line and an LLM router are meant to catch.")

CSS = """:root{--ink:#14213d;--mut:#5b6577;--line:#dfe3ea;--bg:#f6f7f9;--card:#fff;--acc:#1f5fbf;--warn:#b4391f}
*{box-sizing:border-box}body{margin:0;font:16px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg)}
main{max-width:980px;margin:0 auto;padding:32px 20px 80px}h1{font-size:30px;line-height:1.2;margin:0 0 6px}
h2{font-size:21px;margin:44px 0 6px;padding-top:14px;border-top:2px solid var(--ink)}h3{font-size:16px;margin:22px 0 4px}
.sub{color:var(--mut);margin:0 0 18px}.tag{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--acc);font-weight:700}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px 18px;margin:14px 0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:16px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px}.kpi b{display:block;font-size:30px;line-height:1.1}.kpi span{font-size:13px;color:var(--mut)}
table{border-collapse:collapse;width:100%;font-size:14px;background:#fff}th,td{padding:8px 9px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top}th.l,td.l{text-align:left}
th{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut);background:#fafbfc}
.scroll{overflow:auto;border:1px solid var(--line);border-radius:6px}.note{font-size:12.5px;color:var(--mut)}
.exh{font-weight:700;font-size:14px;margin:18px 0 6px}.src{font-size:12px;color:var(--mut);margin:6px 0 14px}
.callout{border-left:4px solid var(--acc);background:#eef3fb;padding:10px 14px;margin:14px 0;border-radius:0 6px 6px 0}
.k{display:inline-block;min-width:104px;font-weight:700;color:var(--acc);text-transform:uppercase;font-size:12px;letter-spacing:.06em}
.scr p{margin:6px 0}ol.rec{padding-left:22px}ol.rec li{margin:8px 0}svg text{font:12px -apple-system,Segoe UI,sans-serif;fill:#5b6577}code{background:#eef0f4;padding:1px 5px;border-radius:4px;font-size:13px}ul{padding-left:20px}"""

page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Can a business question get a governed answer? A KPI layer for EIA electricity data</title><style>{CSS}</style></head><body><main>
<div class="tag">AI-enabled analytics · Governed self-serve BI</div>
<h1>Can a business question get a governed answer?</h1>
<p class="sub">A certified-metric layer for U.S. commercial electricity data, with a measured evaluation of how often it is right, how often it declines, and how it fails.</p>

<h2>Executive summary</h2>
<div class="card scr">
<p><span class="k">Situation</span> Business teams want to ask data questions in plain English, and language models can now produce a fluent answer to almost any question.</p>
<p><span class="k">Complication</span> A model that writes its own SQL can return a confident number with no agreed definition behind it, and a finance team cannot audit it. Wrong answers are worse than no answer.</p>
<p><span class="k">Resolution</span> Let the router choose only among certified metrics, compute every number from fixed reviewed SQL, validate each request, and decline everything else. On {pool['n']} questions written after the router was frozen, it answered {pct(pool['nc'], pool['na'])} of answerable questions correctly and declined {pct(pool['nd'], pool['nn'])} of out-of-scope ones, with {len(wrongs)} confidently wrong answers ({pct(len(wrongs), pool['n'])}).</p>
</div>
<div class="kpis">
<div class="kpi"><b>{pct(pool['nc'], pool['na'])}</b><span>answer accuracy on unseen questions ({pool['nc']} of {pool['na']}; 95% interval {lo:.0f}–{hi:.0f}%)</span></div>
<div class="kpi"><b>{pct(pool['nd'], pool['nn'])}</b><span>of out-of-scope or ambiguous questions correctly declined ({pool['nd']} of {pool['nn']})</span></div>
<div class="kpi"><b>{pct(len(wrongs), pool['n'])}</b><span>confidently wrong answers ({len(wrongs)} of {pool['n']})</span></div>
<div class="kpi"><b>0</b><span>paths from user text into SQL; {len(D)} certified metrics, each with golden tests</span></div>
</div>
<h3>Recommendations</h3>
<ol class="rec">
<li><b>Make the metric dictionary the contract for any self-serve data assistant.</b> The model may route to a metric but never calculates, and every answer carries its definition, source and caveat.</li>
<li><b>Set acceptance targets before launch and measure on questions written after the router is frozen.</b> Proposed targets: at least 90% answer accuracy, at least 95% correct declines, and a wrong-answer rate under 2%. On point estimates the baseline meets the first two and misses the third.</li>
<li><b>Close the vocabulary gap with an LLM router behind the same validator, and score it on the same sets before switching.</b> Every confidently wrong answer was a vocabulary or grain error that a rule-based router cannot see.</li>
<li><b>Show how each question was interpreted and log every miss.</b> Misroutes then become new synonyms or new certified metrics rather than silent errors.</li>
</ol>
<h3>Next steps</h3>
<div class="scroll"><table><tr><th class="l">When</th><th class="l">Action</th><th class="l">Owner</th></tr>
<tr><td class="l">Next 2 weeks</td><td class="l">Run the Gemini router on all four question sets (<code>python -m askkpi.eval --router gemini</code>) and compare with this baseline</td><td class="l">Analytics lead</td></tr>
<tr><td class="l">Next 30 days</td><td class="l">Add a fresh unseen set drawn from real user questions; expand synonyms and metrics from the miss log</td><td class="l">Metric owners</td></tr>
<tr><td class="l">Before rollout</td><td class="l">Agree acceptance targets and the escalation path for declined questions</td><td class="l">Head of Finance Analytics</td></tr>
<tr><td class="l">Ongoing</td><td class="l">Review the miss log monthly and re-certify metrics when sources change</td><td class="l">Metric owners</td></tr></table></div>

<h2>Constraining the model to certified metrics keeps numbers auditable and makes failures visible</h2>
<div class="exh">Exhibit 1 · How a question becomes an answer</div>
<div class="card">{arch}</div>
<p class="src">Source: this repository (<code>askkpi/</code>). The router can be the deterministic baseline or an LLM; the validator and metrics are identical.</p>
<div class="callout"><b>So what:</b> an LLM can be swapped in at the router without changing what a number means. The risk it adds is misrouting, which is measurable, rather than invented figures, which are not.</div>

<h2>On unseen questions the baseline is right about {round(10 * pool['nc'] / pool['na'])} times in 10 and declines nearly everything it should</h2>
<div class="exh">Exhibit 2 · Evaluation scorecard (rule-based router)</div>
<div class="scroll">{score}</div>
<p class="src">Source: <code>eval/*.yaml</code> question sets and <code>outputs/eval_*.json</code>. Expected values come from an independent pandas reference (<code>eval/golden.py</code>), not from the metric SQL. Answer accuracy = correct values among answerable questions. Wrong-answer rate = confidently wrong answers or answers where a decline was expected, among all questions.</p>
<div class="callout"><b>So what:</b> the dev score of 100% comes from tuning on that set and says little. The unseen first-run scores are the evidence, and the sample is small: the 95% interval on answer accuracy runs {lo:.0f}–{hi:.0f}%.</div>

<h2>Most misses are safe; the dangerous ones are vocabulary and grain errors</h2>
<div class="exh">Exhibit 3 · Why the {len(fails)} unseen-set failures happened</div>
<div class="card">{fchart}</div>
<p class="src">Source: failures on unseen sets 1 and 2 ({c1['n']} + {c2['n']} questions), each labeled by cause in <code>scripts/build_report.py</code>.</p>
<div class="callout"><b>So what:</b> {len(fails) - len(wrongs)} of {len(fails)} misses were safe (declined or asked a follow-up). The {len(wrongs)} wrong answers all came from words the router did not know ("pay per kWh", "year-over-year", "drop") or from a ranking over regions it quietly answered with states. That is where an LLM router should help, and where it needs measuring.</div>

<h2>Ten certified metrics cover price, volume, spend and 2027 budget risk</h2>
<div class="exh">Exhibit 4 · Metric dictionary</div>
<div class="scroll">{cat}</div>
<p class="src">Source: <code>askkpi/metrics.yaml</code>; EIA retail sales and Short-Term Energy Outlook; budget metrics come from the companion Electricity Cost-Risk Study.</p>

<h2>What users see: answers, clarifications and declines</h2>
<div class="exh">Exhibit 5 · Real transcripts from the evaluation runs</div>
<div class="scroll">{tr}</div>
<p class="src">Source: <code>outputs/eval_dev.json</code>, <code>outputs/eval_holdout_v1.json</code>, <code>outputs/eval_holdout2_v2.json</code>. {tr_note}</p>

<h2>How the evaluation was run, and what it does not show</h2>
<ul>
<li><b>Question sets were written by the project author.</b> They are not independent users. The dev set (65) was used to tune; unseen set 1 (70) was written after the router and metrics were hashed (<code>eval/FREEZE.txt</code>); the router was then fixed generically for its failures (v2, <code>eval/FREEZE_v2.txt</code>) and scored once on unseen set 2 (45). Unseen set 1 is no longer clean after those fixes and is not rescored.</li>
<li><b>The pooled figure mixes two router versions</b> (v1 on set 1, v2 on set 2), each on its first run.</li>
<li><b>No LLM was run.</b> The sandbox had no model access, so only the deterministic router is measured. A Gemini router with the same output schema and validator is included and unit-tested against mocked responses, but its accuracy is unknown.</li>
<li><b>Warehouse:</b> SQLite behind a thin adapter (DuckDB was not available here). The data are public EIA series, so the numbers are real but the metric owner and policy targets are illustrative.</li>
<li><b>Small samples:</b> {pool['na']} answerable and {pool['nn']} decline questions in the pooled figure; differences of a few points are not meaningful.</li>
<li><b>An MCP server</b> (<code>askkpi/mcp_server.py</code>) exposes <code>ask_kpi</code>, <code>query_metric</code>, <code>list_metrics</code> and <code>describe_metric</code> to any MCP client; none accept SQL.</li>
</ul>
<p class="note" style="margin-top:36px">Companion project: Electricity Cost-Risk Study. Data: U.S. Energy Information Administration.</p>
</main></body></html>"""
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs/index.html").write_text(page)
print("docs/index.html", len(page), "fails", len(fails), "wrong", len(wrongs), pool)
