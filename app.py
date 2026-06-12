"""
Streamlit demo UI for the Multi-Modal AI Embedding System.

Run with:
    streamlit run app.py

Requires the FastAPI backend running on http://localhost:8000:
    uvicorn src.api:app --reload
"""

from typing import Optional

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

# --- Page config ------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Modal Search",
    page_icon="🔎",
    layout="wide",
)

# --- Presentation -----------------------------------------------------------
MODALITY_ICONS = {"text": "📄", "audio": "🎵", "video": "🎬"}
MODALITY_COLORS = {"text": "#3b82f6", "audio": "#10b981", "video": "#f59e0b"}


def modality_badge(modality: str, sub_type: Optional[str] = None) -> str:
    icon = MODALITY_ICONS.get(modality, "•")
    color = MODALITY_COLORS.get(modality, "#6b7280")
    label = modality.upper()
    if sub_type:
        label += f" · {sub_type}"
    return (
        f'<span style="background:{color};color:white;padding:3px 10px;'
        f'border-radius:4px;font-size:0.78rem;font-weight:600;">'
        f"{icon} {label}</span>"
    )


# --- Backend calls ----------------------------------------------------------
def get_stats() -> Optional[dict]:
    try:
        r = requests.get(f"{API_BASE}/stats", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def search_text(query: str, k: int, modality: Optional[str], use_visual: bool) -> dict:
    body = {
        "query": query,
        "k": k,
        "modality": modality,
        "use_visual": use_visual,
        "debug": False,
    }
    r = requests.post(f"{API_BASE}/search", json=body, timeout=60)
    r.raise_for_status()
    return r.json()


def search_audio(audio_file, k: int, modality: Optional[str], use_visual: bool) -> dict:
    files = {
        "audio": (
            audio_file.name,
            audio_file.getvalue(),
            audio_file.type or "application/octet-stream",
        )
    }
    data = {
        "k": str(k),
        "use_visual": str(use_visual).lower(),
        "debug": "false",
    }
    if modality:
        data["modality"] = modality
    r = requests.post(f"{API_BASE}/search/audio", files=files, data=data, timeout=180)
    r.raise_for_status()
    return r.json()


def ingest_file(endpoint: str, file) -> dict:
    files = {
        "file": (
            file.name,
            file.getvalue(),
            file.type or "application/octet-stream",
        )
    }
    r = requests.post(f"{API_BASE}/{endpoint}", files=files, timeout=600)
    r.raise_for_status()
    return r.json()


# --- Result rendering -------------------------------------------------------
def render_results(results: list) -> None:
    if not results:
        st.info("No results.")
        return
    for r in results:
        with st.container(border=True):
            cols = st.columns([1, 5, 2])
            with cols[0]:
                st.markdown(f"### #{r['rank']}")
            with cols[1]:
                st.markdown(
                    modality_badge(r["modality"], r.get("type")),
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Source:** `{r['source']}`")
                if r.get("timestamp_sec") is not None:
                    st.caption(f"Timestamp: {r['timestamp_sec']}s")
            with cols[2]:
                st.metric("RRF score", f"{r['rrf_score']:.4f}")
            doc = r.get("document", "")
            if doc:
                preview = doc if len(doc) <= 500 else doc[:500] + "..."
                st.markdown(f"> {preview}")


# --- Sidebar ----------------------------------------------------------------
with st.sidebar:
    st.title("🔎 Multi-Modal Search")
    st.caption("Cross-modal semantic search over text, audio, and video.")
    st.divider()
    st.subheader("Collection stats")
    stats = get_stats()
    if stats is None:
        st.error(
            "Backend offline.\n\nStart it with:\n\n`uvicorn src.api:app --reload`"
        )
    else:
        c1, c2 = st.columns(2)
        c1.metric("Main", stats.get("main_collection", "?"))
        c2.metric("Visual", stats.get("visual_collection", "?"))
        st.caption(
            "Main: text + audio transcripts + video transcripts + BLIP captions "
            "(384d MiniLM). Visual: video frame CLIP embeddings (512d)."
        )
    if st.button("↻ Refresh stats", use_container_width=True):
        st.rerun()


# --- Main -------------------------------------------------------------------
st.title("Multi-Modal AI Embedding System")
st.caption(
    "Demo UI · FastAPI backend on localhost:8000 · ChromaDB persistent store · "
    "RRF fusion across MiniLM + CLIP collections"
)

tab_text, tab_audio, tab_ingest = st.tabs(
    ["🔍 Search by text", "🎤 Search by audio", "📥 Ingest files"]
)

# --- Tab 1: Text query ------------------------------------------------------
with tab_text:
    st.subheader("Search by text")
    st.caption(
        "Natural-language query. Searches text, audio transcripts, video transcripts, "
        "BLIP captions, and CLIP frame embeddings; results fused via RRF."
    )
    query = st.text_input(
        "Query",
        placeholder="e.g. how do I make spaghetti",
        key="text_query",
    )
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        k = st.slider("Top-k", 1, 20, 5, key="text_k")
    with c2:
        modality_choice = st.selectbox(
            "Filter by modality",
            ["(all)", "text", "audio", "video"],
            key="text_modality",
        )
    with c3:
        use_visual = st.checkbox(
            "Include visual search (CLIP)", value=True, key="text_visual"
        )
    if st.button("Search", type="primary", key="text_search"):
        if not query.strip():
            st.warning("Enter a query.")
        else:
            modality = None if modality_choice == "(all)" else modality_choice
            try:
                with st.spinner("Searching..."):
                    resp = search_text(query, k, modality, use_visual)
                st.success(f"{resp['n_results']} result(s)")
                render_results(resp["results"])
            except Exception as e:
                st.error(f"Search failed: {e}")

# --- Tab 2: Audio query -----------------------------------------------------
with tab_audio:
    st.subheader("Search by audio")
    st.caption(
        "Upload a short audio file. Whisper transcribes it, then the transcript is "
        "used as a text query."
    )
    audio_file = st.file_uploader(
        "Audio file",
        type=["mp3", "mp4", "m4a", "wav", "ogg", "flac"],
        key="audio_upload",
    )
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        k_a = st.slider("Top-k", 1, 20, 5, key="audio_k")
    with c2:
        modality_choice_a = st.selectbox(
            "Filter by modality",
            ["(all)", "text", "audio", "video"],
            key="audio_modality",
        )
    with c3:
        use_visual_a = st.checkbox(
            "Include visual search (CLIP)", value=True, key="audio_visual"
        )
    if st.button("Transcribe & search", type="primary", key="audio_search"):
        if audio_file is None:
            st.warning("Upload an audio file first.")
        else:
            modality = None if modality_choice_a == "(all)" else modality_choice_a
            try:
                with st.spinner("Transcribing with Whisper, then searching..."):
                    resp = search_audio(audio_file, k_a, modality, use_visual_a)
                st.markdown(f"**Transcript:** _{resp['transcript']}_")
                search_resp = resp["search"]
                st.success(f"{search_resp['n_results']} result(s)")
                render_results(search_resp["results"])
            except Exception as e:
                st.error(f"Audio search failed: {e}")

# --- Tab 3: Ingest ----------------------------------------------------------
with tab_ingest:
    st.subheader("Ingest new files")
    st.caption(
        "Add content to the searchable collections. Uploads land in "
        "data/{text,audio,video} and are embedded into ChromaDB."
    )
    st.warning(
        "Each upload adds new documents — uploading the same file twice duplicates "
        "entries.",
        icon="⚠️",
    )
    ing_text, ing_audio, ing_video = st.columns(3)
    with ing_text:
        st.markdown("**📄 Text (.txt)**")
        f_text = st.file_uploader("text", type=["txt"], key="ing_text_up", label_visibility="collapsed")
        if st.button("Ingest text", key="ing_text_btn", disabled=(f_text is None)):
            try:
                with st.spinner("Ingesting..."):
                    out = ingest_file("ingest/text", f_text)
                st.success(
                    f"Ingested. Main: {out['counts']['main_collection']}, "
                    f"Visual: {out['counts']['visual_collection']}"
                )
            except Exception as e:
                st.error(f"Failed: {e}")
    with ing_audio:
        st.markdown("**🎵 Audio**")
        f_audio = st.file_uploader(
            "audio",
            type=["mp3", "mp4", "m4a", "wav", "ogg", "flac"],
            key="ing_audio_up",
            label_visibility="collapsed",
        )
        if st.button("Ingest audio", key="ing_audio_btn", disabled=(f_audio is None)):
            try:
                with st.spinner("Transcribing & embedding..."):
                    out = ingest_file("ingest/audio", f_audio)
                st.success(
                    f"Ingested. Main: {out['counts']['main_collection']}, "
                    f"Visual: {out['counts']['visual_collection']}"
                )
            except Exception as e:
                st.error(f"Failed: {e}")
    with ing_video:
        st.markdown("**🎬 Video**")
        f_video = st.file_uploader(
            "video",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            key="ing_video_up",
            label_visibility="collapsed",
        )
        if st.button("Ingest video", key="ing_video_btn", disabled=(f_video is None)):
            try:
                with st.spinner("Extracting frames + transcript + captions (slow)..."):
                    out = ingest_file("ingest/video", f_video)
                st.success(
                    f"Ingested. Main: {out['counts']['main_collection']}, "
                    f"Visual: {out['counts']['visual_collection']}"
                )
            except Exception as e:
                st.error(f"Failed: {e}")
