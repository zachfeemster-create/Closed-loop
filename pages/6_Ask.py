"""Ask: private chat about your data with Gemini. Gemini runs the real Closed Loop tests; it never invents a verdict.

Locked with a password and powered by the owner's own free Gemini key, both kept in the app's Streamlit secrets:

    [ask]
    password = "something only you know"
    gemini_key = "key from aistudio.google.com"
    persona = "optional standing instructions"
"""
import hmac
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from closedloop import ask as AK          # noqa: E402
from closedloop import sources as S       # noqa: E402

st.set_page_config(page_title="Ask · Closed Loop", page_icon="💬", layout="centered")
ss = st.session_state


def _secret(name, default=""):
    try:
        return str(st.secrets["ask"][name]).strip()
    except Exception:
        return default


PASSWORD, KEY, PERSONA = _secret("password"), _secret("gemini_key"), _secret("persona")

st.title("Ask")

# ---------------------------------------------------------------- not set up yet
if not PASSWORD or not KEY:
    st.markdown("This page is private: a chat with Gemini about your data, where Gemini runs the real tests. "
                "It isn't set up on this copy of the app yet.")
    with st.expander("Set it up (owner only, about 5 minutes, free)", expanded=True):
        st.markdown(
            "1. Go to **aistudio.google.com**, sign in with Google, tap **Get API key → Create API key**. "
            "No card needed. Copy the key.\n"
            "2. Go to **share.streamlit.io**, open this app's **⋮ → Settings → Secrets**, and paste:\n\n"
            "```toml\n[ask]\npassword = \"pick something only you know\"\ngemini_key = \"paste the key here\"\n```\n"
            "3. Save. The app restarts and this page asks for your password.\n\n"
            "Free tier notes: Google may use free-tier chats to improve its products, so don't paste anything private. "
            "Daily limits are small and reset at midnight Pacific; Flash-Lite gets the most free questions.")
    st.stop()

# ---------------------------------------------------------------- password gate
if not ss.get("ask_ok"):
    pw = st.text_input("Password", type="password")
    if pw:
        if hmac.compare_digest(pw.encode(), PASSWORD.encode()):
            ss["ask_ok"] = True
            st.rerun()
        else:
            st.error("Wrong password.")
    st.stop()

# ---------------------------------------------------------------- session state
if "ask_ws" not in ss:
    ss["ask_ws"] = AK.Workspace()
ws: AK.Workspace = ss["ask_ws"]
ss.setdefault("ask_msgs", [])           # [{"role": "user"|"assistant", "text": ..., "runs": [...]}]

MODELS = ["gemini-flash-lite-latest", "gemini-flash-latest"]

with st.sidebar:
    st.header("Gemini")
    model = st.selectbox("Model", MODELS + ss.get("ask_more_models", []), index=0,
                         help="Flash-Lite gets the most free questions a day. Flash is sharper but runs out fast.")
    if st.button("Check my models"):
        try:
            found = [m for m in AK.list_models(KEY) if "gemini" in m and m not in MODELS]
            ss["ask_more_models"] = found
            st.success(f"{len(found)} more available; they're in the list now.")
        except Exception as e:
            st.error(AK.friendly_error(e))
    if st.button("New chat"):
        ss["ask_msgs"] = []
        ss.pop("ask_chat", None)
        ws.runs = []
        st.rerun()
    if st.button("Lock this page"):
        for k in ("ask_ok", "ask_chat"):
            ss.pop(k, None)
        st.rerun()
    st.caption("Free tier: Google may use these chats to improve its products. Don't paste private info.")

# ---------------------------------------------------------------- data
with st.expander("Data" + (f": {ws.source}" if ws.df is not None else " (none loaded; Gemini can also load FRED or World Bank itself)"),
                 expanded=ws.df is None):
    src = st.radio("Where from", ["Upload", "Paste", "Google Sheet or link", "Demo"], horizontal=True, key="ask_src")
    try:
        if src == "Upload":
            f = st.file_uploader("CSV or Excel: one column per series, time down the rows", type=None, key="ask_up")
            if f is not None and st.button("Use this file"):
                raw = f.getvalue()
                df = S.read_excel(raw) if f.name.lower().endswith((".xlsx", ".xls")) else S.read_csv_bytes(raw)
                ws.set_data(df, f"upload: {f.name}")
                st.rerun()
        elif src == "Paste":
            txt = st.text_area("Paste cells (header row first)", height=140, key="ask_paste")
            if txt.strip() and st.button("Use pasted cells"):
                ws.set_data(S.read_pasted(txt), "pasted cells")
                st.rerun()
        elif src == "Google Sheet or link":
            url = st.text_input("Link", key="ask_link")
            if url.strip() and st.button("Load link"):
                ws.set_data(S.fetch_link(url.strip()), f"link: {url.strip()[:100]}")
                st.rerun()
        else:
            pick = st.selectbox("Demo", list(S.SAMPLES), key="ask_demo")
            if st.button("Load demo"):
                ws.set_data(S.load_sample(pick), f"demo: {pick}")
                st.rerun()
    except Exception as e:
        st.error(str(e))
    if ws.df is not None:
        st.caption(f"{ws.df.shape[0]:,} rows × {ws.df.shape[1]} columns")
        st.dataframe(ws.df.head(6), hide_index=True, width="stretch")

# ---------------------------------------------------------------- chat


def _chat():
    """One Gemini chat per session and model; rebuilt (keeping the history) if the model changes."""
    cur = ss.get("ask_chat")
    if cur and cur[0] == model:
        return cur[2]
    hist = cur[2].get_history() if cur else None
    client, chat = AK.make_chat(KEY, model, ws, persona=PERSONA, history=hist)
    ss["ask_chat"] = (model, client, chat)
    return chat


def _show_runs(runs):
    for r in runs:
        res = r["result"]
        label = f"Ran {r['kind']}: " + (str(res.get("verdict")) if isinstance(res, dict) else "")
        with st.expander(label[:120]):
            if isinstance(res, dict) and res.get("rows"):
                tbl = pd.DataFrame(res["rows"])
                if "bar" in tbl:
                    tbl["bar"] = tbl["bar"].map(lambda b: f"{b[0]:.4g} to {b[1]:.4g}" if isinstance(b, (list, tuple))
                                                else ("" if b is None else f"{b:.4g}"))
                st.dataframe(tbl, hide_index=True, width="stretch")
            if isinstance(res, dict) and res.get("floor_table"):
                st.caption("Smallest wave each lens heard 80% of the time (typical steps; columns are periods in rows)")
                st.dataframe(pd.DataFrame(res["floor_table"]), hide_index=True, width="stretch")
            st.json(r, expanded=False)


if not ss["ask_msgs"]:
    st.caption("Ask the way you'd ask me: \"is there anything real in this?\", \"does gas answer crude a week later?\", "
               "\"pull World Bank infant mortality and check the orbit\". Every verdict comes from the test bench, "
               "and each test it runs shows up under its answer.")

for m in ss["ask_msgs"]:
    with st.chat_message(m["role"]):
        st.markdown(m["text"])
        if m.get("runs"):
            _show_runs(m["runs"])

q = st.chat_input("Ask about your data")
if q:
    ss["ask_msgs"].append(dict(role="user", text=q))
    with st.chat_message("user"):
        st.markdown(q)
    before = len(ws.runs)
    with st.chat_message("assistant"):
        with st.spinner("Thinking and running tests…"):
            try:
                resp = _chat().send_message(q)
                text = (resp.text or "").strip() or AK.summary_of_runs(ws.runs[before:])
            except Exception as e:
                text = AK.friendly_error(e)
                if ws.runs[before:]:
                    text += "\n\n" + AK.summary_of_runs(ws.runs[before:])
        st.markdown(text)
        new_runs = ws.runs[before:]
        _show_runs(new_runs)
    ss["ask_msgs"].append(dict(role="assistant", text=text, runs=new_runs))

if ss["ask_msgs"]:
    transcript = dict(saved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), model=model,
                      data=ws.source, messages=ss["ask_msgs"])
    st.download_button("Download this chat (.json)", json.dumps(transcript, indent=2, default=str),
                       file_name=f"ask_{datetime.now(timezone.utc):%Y%m%d_%H%M}.json")
