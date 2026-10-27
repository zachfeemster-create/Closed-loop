"""Two kinds of auto log for the Closed Loop.

1. On this device: every locked plan and its results save in the visitor's own browser (localStorage).
   Nothing leaves their device. Done in app.py with a tiny browser bridge.
2. Shared log: every run is written to a private GitHub repo the app owner controls, if the owner has put
   a token in the app's Streamlit secrets:

       [github]
       token = "github_pat_..."
       repo = "owner/closed-loop-log"

   Each run gets a short ID (e.g. 7F3A1C2B) and is stored at runs/<ID>.json — the question, the locked plan,
   the result and the data's fingerprint. The raw data is only stored if the person ticks the box.
"""
from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone

API = "https://api.github.com"


def config(secrets) -> tuple[str, str] | None:
    """(token, repo) from st.secrets, or None when the shared log isn't set up."""
    try:
        g = secrets["github"]
        tok, repo = str(g["token"]).strip(), str(g["repo"]).strip()
        return (tok, repo) if tok and "/" in repo else None
    except Exception:
        return None


def _req(method: str, url: str, token: str, body: dict | None = None, raw: bool = False, timeout: int = 15):
    headers = {"Authorization": f"Bearer {token}", "User-Agent": "ClosedLoop/1.0",
               "Accept": "application/vnd.github.raw+json" if raw else "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28"}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def run_id(lock_id: str) -> str:
    return lock_id[:8].upper()


def put_file(cfg, path: str, content: bytes, message: str) -> bool:
    token, repo = cfg
    body = {"message": message, "content": base64.b64encode(content).decode()}
    try:
        status, _ = _req("PUT", f"{API}/repos/{repo}/contents/{path}", token, body)
        return status in (200, 201)
    except urllib.error.HTTPError as e:
        return e.code == 422          # 422 = that file already exists (same run logged twice): fine
    except Exception:
        return False


def log_run(cfg, rid: str, plan: dict, node: dict, data_csv: bytes | None = None) -> bool:
    """Write one run to runs/<ID>.json (and runs/<ID>_data.csv if the person shared their data)."""
    record = dict(run_id=rid, logged_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  keystone=plan.get("intake", {}).get("keystone", ""), plan=plan, run=node,
                  data_included=bool(data_csv))
    ok = put_file(cfg, f"runs/{rid}.json", json.dumps(record, indent=2, default=str).encode(),
                  f"Run {rid}: {record['keystone'][:60]}")
    if ok and data_csv is not None and len(data_csv) < 900_000:
        put_file(cfg, f"runs/{rid}_data.csv", data_csv, f"Data for run {rid}")
    return ok


def get_run(cfg, rid: str) -> dict | None:
    token, repo = cfg
    rid = rid.strip().upper()
    try:
        _, body = _req("GET", f"{API}/repos/{repo}/contents/runs/{rid}.json", token, raw=True)
        return json.loads(body)
    except Exception:
        return None


def get_data(cfg, rid: str) -> bytes | None:
    token, repo = cfg
    try:
        _, body = _req("GET", f"{API}/repos/{repo}/contents/runs/{rid.strip().upper()}_data.csv", token, raw=True)
        return body
    except Exception:
        return None


def list_files(cfg, folder: str) -> list[str]:
    """Names of the files in one folder of the shared log (newest names last when names carry dates)."""
    token, repo = cfg
    try:
        _, body = _req("GET", f"{API}/repos/{repo}/contents/{folder}", token)
        return sorted(i["name"] for i in json.loads(body) if i.get("type") == "file")
    except Exception:
        return []


def get_json(cfg, path: str) -> dict | None:
    token, repo = cfg
    try:
        _, body = _req("GET", f"{API}/repos/{repo}/contents/{path}", token, raw=True)
        return json.loads(body)
    except Exception:
        return None
