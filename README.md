# Ask the KPI: a governed natural-language layer over EIA electricity data

**Report:** `docs/index.html` (static, answer-first, with the evaluation scorecard and real transcripts).

**Where this started:** the EIA research assistant. The idea came from my Research Analyst internship at the Hamm Institute for American Energy at Oklahoma State University (2025). I built the assistant as a course project for Agentic AI Systems at Oklahoma State (April 2026), then took the same EIA data and asked what an AI assistant needs so that finance teams can trust its numbers: certified metrics and a measured evaluation.

## The answer

A language model should choose a certified metric, never write the number. On 115 questions written after the router was frozen, this design answered 90.1% of answerable questions correctly (73 of 81, 95% interval 82–95%), declined 96.3% of out-of-scope or ambiguous questions (26 of 27) and gave 4 confidently wrong answers (3.5%). All four were vocabulary or grain errors, and the report breaks every miss down by cause.

## The problem

Self-serve analytics assistants that write their own SQL can return a confident number with no agreed definition behind it. A finance team cannot audit that, and a wrong answer costs more than no answer.

## How it works

```
question -> router (rule-based or LLM) -> validator -> certified metric (fixed SQL) -> answer + definition + caveat
                                              \-> decline or ask a clarifying question
```

- **Metric dictionary** (`askkpi/metrics.yaml`): 10 certified metrics, each with a definition, unit, source, owner, caveat, fixed SQL, synonyms and golden test values.
- **Router** (`askkpi/router.py`): outputs a request (metric, operation, geography, years), never a number or SQL. A deterministic baseline is measured here; a Gemini router with the same schema is included but not run (no model access in the build environment).
- **Validator + engine** (`askkpi/semantic.py`): every request is validated against the dictionary and typed parameters; user text never reaches SQL. Rankings are by state only.
- **MCP server** (`askkpi/mcp_server.py`): `ask_kpi`, `query_metric`, `list_metrics`, `describe_metric` for any MCP client.
- **Evaluation** (`askkpi/eval.py`, `eval/`): labeled question sets; expected values come from an independent pandas reference (`eval/golden.py`), not from the metric SQL.

## KPIs

| KPI | Definition |
|---|---|
| Answer accuracy | Correct values among answerable questions |
| Correct declines | Out-of-scope or ambiguous questions that were declined or clarified |
| Wrong-answer rate | Confidently wrong answers, or answers where a decline was expected, among all questions |
| False refusals | Answerable questions that were declined |

## Results

| Set | Questions | Answer accuracy | Correct declines | Wrong-answer rate |
|---|---|---|---|---|
| Dev (tuned on) | 65 | 100% | 100% | 0% |
| Unseen 1, first run (router v1) | 70 | 89.4% | 94.7% | 2.9% |
| Unseen 2, first run (router v2) | 45 | 91.2% | 100% | 4.4% |
| Unseen pooled | 115 | 90.1% | 96.3% | 3.5% |

Question sets were written by the project author, not by independent users. The router was hashed before unseen set 1 was written (`eval/FREEZE.txt`), fixed generically for its failures (`eval/FREEZE_v2.txt`) and scored once on unseen set 2. Unseen set 1 is no longer clean after those fixes.

## Limits

No LLM was run, so only the rule-based router is measured. The warehouse is SQLite behind a thin adapter (DuckDB was unavailable in the build environment). Samples are small. Metric ownership and acceptance targets are illustrative; the underlying EIA data are real.

## Reproduce

```bash
pip install -r requirements.txt
pytest -q                                   # 24 tests: golden values, validator, injection, router, MCP, regression
python -m askkpi.eval --set dev             # also: holdout2 (holdout is the burned v1 set)
python -m askkpi.eval --set holdout2 --router gemini   # needs GEMINI_API_KEY
python scripts/build_report.py              # rebuild docs/index.html from outputs/
python -m askkpi.mcp_server                 # stdio MCP server
```

Data: U.S. Energy Information Administration (retail sales and Short-Term Energy Outlook), plus the budget-risk table from the companion Electricity Cost-Risk Study.
