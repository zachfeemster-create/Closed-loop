"""Ask: a private chat about your data with Gemini, where Gemini runs the real Closed Loop tests.

Gemini never makes up a verdict. Every pass or fail comes from closedloop.lab (matched fakes, fixed rules);
Gemini picks the test, runs it through the tools below, and talks through what came back.

The page (pages/6_Ask.py) keeps it private with a password and reads the Gemini key from Streamlit secrets:

    [ask]
    password = "something only you know"
    gemini_key = "your key from aistudio.google.com"
    persona = "optional: extra standing instructions, e.g. your thinking model"
"""
import json
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

from closedloop import lab as LB
from closedloop import sources as S

RULES = """You are the data partner inside Closed Loop, a method for telling real patterns from chance:
look for a pattern, then try hard to kill it with matched fakes, and keep only what survives.

How you work:
- You have tools that load data and run the real tests. Every pass or fail must come from a tool result.
  Never state a verdict, p-value or percentile you did not get from a tool.
- Before running a test, say in one short line which test, which series and settings, and what counts as a pass.
  That line is the locked plan. Then run it.
- If a result disappoints, don't quietly rerun with new settings until something passes. A new setting is a new
  look: say so, and say that each extra look makes a pass weaker.
- Report the verdict plainly first, then what it means and what it doesn't. A pass is a pattern, not a finding.
  The ladder: pattern -> survivor (beats the fakes) -> sturdy survivor (holds under another measure) ->
  lead (holds on fresh data) -> finding (someone else reproduces it).
- Keep a luck pile: anything that looks interesting but didn't pass is noted, not thrown away and not oversold.
- When the user's question is fuzzy, pick the test that fits best and say why in a sentence, or ask one question.
- Plain words, short answers. The user is a surveyor and builder, sharp and fast; skip textbook explanations.
- Don't inflate. Never call a result robust, structural, a fingerprint or a finding off one test. Say the plain
  reason a pass usually happens (for real_vs_fakes: momentum or smoothness that walks and shuffles lack) and name
  the next test that would rule that out. When series were cut into windows, say so and that it weakens the pass.
- Budget: the user is on a small free daily allowance and every tool call spends part of it. Run at most two tests
  per question unless asked for more, never rerun a test you already ran with the same settings, and answer from
  results you already have when you can.
- Only describe the data you have. Its dates, length and frequency come from describe_data (the span field); never
  claim a longer stretch than that. If the user wants a longer history, say so and suggest a source.
- The tests only see the numbers loaded. If a question is about news, headlines, narratives or anything else not in
  the data, say plainly that the loaded numbers can't answer that part and what data would. Never present
  speculation about news or causes as if a test showed it.
- Shared baseline: leave it off for the leftovers test unless the series are many and of one kind (one indicator
  across many countries), because on mixed series taking out a shared path manufactures links.
"""

TEST_HELP = {
    "real_vs_fakes": "Can a six-measure toolkit tell the series from noise, walks, spiky walks and shuffled steps? "
                     "Passes only if it beats walks AND shuffled steps. Fewer than 10 series get cut into windows. "
                     "A pass usually just means the series have momentum or smoothness (most real economic series "
                     "do); it is a first screen, not structure. Follow up with turnback or orbit, whose fakes copy momentum.",
    "leftovers": "Which pairs move together step by step, or one a step ahead of the other, beyond independent "
                 "look-alike fakes? False-discovery corrected.",
    "turnback": "Do the series reverse mid-curve more often than their own momentum explains?",
    "orbit": "Do the remainders cycle more regularly than momentum-matched fakes? Spiral out or in?",
    "interlock": "Do the series' cycles lock together across series? A quarter turn apart (zipper)?",
    "step_flip": "The corner Y: when a series sits in its usual corner, do others land on the flip corners one step "
                 "later, already parked there?",
    "vortex": "Are visits to the rare corners shorter than random walks would give?",
}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def _compact(res: dict, max_rows: int = 12) -> dict:
    """What Gemini sees from a test: the verdict, the scored rows, the notes. Nothing it could misread as data."""
    keep = ("stat", "real", "fakes_mid", "bar", "percentile", "z", "p", "q", "passed", "note")
    rows = [{k: r[k] for k in keep if k in r} for r in res.get("rows", [])[:max_rows]]
    return _jsonable(dict(test=res.get("test"), question=res.get("question"), passed=res.get("passed"),
                          verdict=res.get("verdict"), rows=rows, notes=res.get("notes", []),
                          details={k: v for k, v in res.get("details", {}).items() if not isinstance(v, (list, dict))}))


class Workspace:
    """The data Gemini is talking about, plus a log of every test it ran (shown to the user, never hidden)."""

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.source = ""
        self.runs: list[dict] = []

    # -------------------------------------------------------- data helpers
    def set_data(self, df: pd.DataFrame, source: str):
        self.df = df.reset_index(drop=True)
        self.source = source

    def numeric(self, columns=None) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("No data loaded yet. Load some first (upload, paste, FRED or World Bank).")
        df = self.df
        num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        first = df.columns[0]
        if first in num and str(first).lower() in ("year", "row", "t", "time", "date", "period", "week", "month"):
            num.remove(first)
        if columns:
            want = [str(c) for c in columns]
            missing = [c for c in want if c not in map(str, num)]
            if missing:
                raise ValueError(f"Not in the data (or not numeric): {missing[:5]}. Columns: {list(map(str, num))[:40]}")
            num = [c for c in num if str(c) in want]
        sub = df[num].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) > 5000:                           # same rule as the Run a Test page: keep every k-th row
            sub = sub.iloc[::int(np.ceil(len(sub) / 5000))]
        return sub

    def _log(self, kind: str, args: dict, res: dict):
        self.runs.append(dict(when=datetime.now(timezone.utc).isoformat(timespec="seconds"), kind=kind,
                              args=_jsonable(args), result=res))

    # -------------------------------------------------------- tools (Gemini calls these)
    def tools(self):
        ws = self

        def describe_data() -> dict:
            """Describe the loaded data: where it came from, how many rows, and each numeric column's first value,
            last value, smallest, largest and average. Call this before choosing a test."""
            if ws.df is None:
                return {"loaded": False, "message": "No data loaded yet."}
            num = ws.numeric()
            cols = {}
            for c in list(num.columns)[:60]:
                s = num[c]
                cols[str(c)] = dict(first=float(s.iloc[0]), last=float(s.iloc[-1]), min=float(s.min()),
                                    max=float(s.max()), mean=float(s.mean()))
            t0 = ws.df.iloc[:, 0]
            span = dict(first_column=str(ws.df.columns[0]), starts=str(t0.iloc[0])[:10], ends=str(t0.iloc[-1])[:10])
            return _jsonable(dict(loaded=True, source=ws.source, rows=int(len(ws.df)), usable_rows=int(len(num)),
                                  numeric_columns=int(num.shape[1]), span=span,
                                  columns=cols, more_columns=max(0, num.shape[1] - 60)))

        def peek_rows(how_many: int = 8, from_end: bool = False) -> dict:
            """Show a few raw rows of the loaded data, from the start or the end."""
            if ws.df is None:
                return {"loaded": False}
            n = int(max(1, min(how_many, 30)))
            part = ws.df.tail(n) if from_end else ws.df.head(n)
            return _jsonable(json.loads(part.iloc[:, :20].to_json(orient="records", date_format="iso")))

        def list_tests() -> dict:
            """List the tests that can be run on the loaded series, with what each one asks."""
            return dict(series_tests=TEST_HELP,
                        other_tests=dict(three_point="three positive values: amplitude and phase vs 500,000 random triplets",
                                         leaderboard="a top list of scores: do the gaps bunch or keep a rhythm",
                                         dose_response="dose and response pairs: Hill curve and the leftover under the ceiling"))

        def run_test(test: str, columns: Optional[List[str]] = None, shared_baseline: str = "off",
                     fakes: int = 200, bar_percent: float = 95.0) -> dict:
            """Run one Closed Loop test on the loaded series and return its verdict.
            test: one of real_vs_fakes, leftovers, turnback, orbit, interlock, step_flip, vortex.
            columns: which series to include (leave empty for all numeric columns).
            shared_baseline: 'off', 'all' (take out the average path of all series) or 'groups' (take out each
            group's own path, where a group is the part of a column name before '|' or ':').
            fakes: number of fake sets (100 to 1000). bar_percent: 95, 99 or 99.9."""
            test = str(test).strip()
            if test not in LB.PANEL_TESTS:
                return {"error": f"Unknown test '{test}'. Choose from {list(LB.PANEL_TESTS)}."}
            try:
                sub = ws.numeric(columns)
                X = sub.values.astype(float)
                names = list(map(str, sub.columns))
                n = int(min(max(int(fakes), 100), 1000))
                bar = float(bar_percent) if float(bar_percent) in (95.0, 99.0, 99.9) else 95.0
                kw = dict(seed=20261028, bar_pct=bar)
                mode = str(shared_baseline).lower()
                grp = LB.group_labels(names) if mode == "groups" else None
                remove = mode in ("all", "groups")
                if test == "real_vs_fakes":
                    res = LB.real_vs_fakes(X, **kw)
                elif test == "leftovers":
                    res = LB.leftovers(X, names=names, n_slides=max(n, 300), groups=grp, remove_shared=remove, **kw)
                elif test == "step_flip":
                    res = LB.step_flip(X, n_slides=n, groups=grp, remove_shared=remove, **kw)
                elif test == "orbit":
                    res = LB.orbit(X, n_fakes=n, groups=grp, remove_shared=remove, **kw)
                elif test == "interlock":
                    res = LB.interlock(X, n_fakes=n, groups=grp, **kw)
                else:
                    res = LB.PANEL_TESTS[test](X, n_fakes=n, **kw)
            except Exception as e:
                return {"error": str(e)}
            out = _compact(res)
            out["data"] = f"{sub.shape[1]} series x {sub.shape[0]} rows"
            ws._log(test, dict(columns=columns, shared_baseline=mode, fakes=n, bar_percent=bar), out)
            return out

        def load_fred(series_ids: str) -> dict:
            """Load FRED series by ID (comma-separated, e.g. 'UNRATE, CPIAUCSL'). Replaces the loaded data."""
            try:
                df = S.fetch_fred(series_ids, rename=False)
            except Exception as e:
                return {"error": str(e)}
            ws.set_data(df, f"FRED: {series_ids}")
            return describe_data()

        def load_worldbank(indicator: str, start_year: int = 2000, end_year: int = 2022) -> dict:
            """Load one World Bank indicator for every country with complete years (columns = countries).
            Example indicators: SP.DYN.LE00.IN life expectancy, SP.DYN.IMRT.IN infant mortality, NY.GDP.PCAP.CD GDP
            per person, SP.URB.TOTL.IN.ZS urban share, FP.CPI.TOTL.ZG inflation. Replaces the loaded data."""
            try:
                df = S.fetch_worldbank(indicator, int(start_year), int(end_year))
            except Exception as e:
                return {"error": str(e)}
            ws.set_data(df, f"World Bank {indicator} {start_year}-{end_year}")
            return describe_data()

        def three_point(first: float, second: float, third: float) -> dict:
            """Three positive values (like three masses): amplitude and phase of the three-point cycle, and how
            many of 500,000 random triplets land as close to root two and a simple phase."""
            try:
                out = _compact(LB.three_point([first, second, third]))
            except Exception as e:
                return {"error": str(e)}
            ws._log("three_point", dict(values=[first, second, third]), out)
            return out

        def leaderboard(scores: List[float]) -> dict:
            """A top list of scores (any order, at least 10): do the gaps between ranks bunch or keep a rhythm?"""
            try:
                out = _compact(LB.leaderboard([float(s) for s in scores]))
            except Exception as e:
                return {"error": str(e)}
            ws._log("leaderboard", dict(scores=len(scores)), out)
            return out

        def dose_response(doses: List[float], responses: List[float], ceiling: float = 100.0) -> dict:
            """Dose and response pairs: fit a Hill curve and read the leftover under the ceiling with a range."""
            try:
                out = _compact(LB.dose_response(doses, responses, ceiling=float(ceiling)))
            except Exception as e:
                return {"error": str(e)}
            ws._log("dose_response", dict(points=len(doses), ceiling=ceiling), out)
            return out

        return [describe_data, peek_rows, list_tests, run_test, load_fred, load_worldbank,
                three_point, leaderboard, dose_response]


def make_chat(api_key: str, model: str, ws: Workspace, persona: str = "", history=None):
    """A Gemini chat that can call the workspace tools on its own (up to 4 tool calls per question, to save the
    free daily allowance).
    Returns (client, chat); hold on to both."""
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    system = RULES + (("\nStanding instructions from the user:\n" + persona.strip()) if persona and persona.strip() else "")
    cfg = types.GenerateContentConfig(
        system_instruction=system, tools=ws.tools(), temperature=0.4,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=4))
    chat = client.chats.create(model=model, config=cfg, history=history or [])
    return client, chat          # keep the client alive with the chat; if it's dropped, the connection closes


def list_models(api_key: str) -> List[str]:
    """Gemini models this key can chat with."""
    from google import genai
    client = genai.Client(api_key=api_key)
    out = []
    for m in client.models.list():
        acts = getattr(m, "supported_actions", None) or []
        if not acts or "generateContent" in acts:
            out.append(m.name.replace("models/", ""))
    return sorted(out)


def summary_of_runs(runs: list) -> str:
    """Plain list of what was run, for when Gemini stops before writing an answer (no extra request spent)."""
    if not runs:
        return "(Gemini returned nothing. Try asking a shorter, more specific question.)"
    lines = ["Gemini ran these tests but stopped before writing an answer. Here are the verdicts straight from the bench:"]
    for r in runs:
        res = r.get("result", {})
        lines.append(f"- **{r.get('kind')}** on {res.get('data', 'the data')}: {res.get('verdict', res.get('error', '?'))}")
    lines.append("Ask \"what do those mean?\" for the read on them.")
    return "\n".join(lines)


def friendly_error(e: Exception) -> str:
    s = str(e)
    if "429" in s or "RESOURCE_EXHAUSTED" in s or "quota" in s.lower():
        return ("Gemini's free limit for this model is used up for now. It resets at midnight Pacific. "
                "Switch to the Flash-Lite model (it gets far more free questions) or try again later.")
    if "API key not valid" in s or "API_KEY_INVALID" in s or "401" in s or "403" in s:
        return "Gemini didn't accept the key. Check the gemini_key in the app's secrets (no spaces or quotes inside)."
    if "not found" in s.lower() and "model" in s.lower():
        return "That model name isn't available to this key. Tap 'Check my models' and pick one from the list."
    return f"Gemini error: {s[:300]}"
