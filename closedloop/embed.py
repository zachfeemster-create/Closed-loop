"""Show a self-contained HTML view (the globes) on any Streamlit version: st.iframe where it exists, the older
components.html otherwise."""
from __future__ import annotations

import streamlit as st


def show(html: str, height: int) -> None:
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=height)


def safe_json(s: str) -> str:
    """JSON going inside a <script> tag: names from other people's data can't close the tag early."""
    return s.replace("</", "<\\/")
