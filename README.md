# Multi-Modal AI Embedding System

A proof-of-concept that ingests text, audio, and video files, generates vector embeddings, and enables natural-language semantic search across all modalities.

Built as part of an SPS internship task — see [the project report](docs/REPORT.md) (coming soon) for the full architecture writeup.

## What it does

You can drop in `.txt`, `.mp3`/`.wav`/`.mp4` audio, or `.mp4` video files. The system extracts and indexes:

- **Text files** → embedded directly with sentence-transformers MiniLM (384d)
- **Audio files** → transcribed by Whisper, then embedded
- **Video files** → three independent signals per video:
  1. **Audio transcript** (Whisper on the audio track)
  2. **Visual frames** (CLIP ViT-B-32 image embeddings, 512d)
  3. **Frame captions** (BLIP image captioning → embedded as text)

All embeddings persist in ChromaDB. Search queries can come back with hits from any modality.

## Architecture

Two ChromaDB collections live side-by-side:

| Collection | Dimension | Contains |
|---|---|---|
| `multimodal` | 384 | Text chunks, audio transcripts, video transcripts, BLIP captions |
| `multimodal_visual` | 512 | CLIP frame embeddings |

A text query is embedded twice — once with MiniLM (to search the main collection) and once with CLIP's text encoder (to search the visual collection). Results are merged using **Reciprocal Rank Fusion (RRF)**, with a CLIP distance threshold to drop weakly-matched frames before fusion.

This means one query like *"a person at a desk with a laptop"* can retrieve the same video via three different paths:
- BLIP caption match (text similarity)
- Whisper transcript match (text similarity)
- Raw CLIP visual match (image-text joint embedding)

## Tech stack

- Python 3.11
- ChromaDB (persistent vector store)
- sentence-transformers `all-MiniLM-L6-v2` — text embeddings
- OpenAI Whisper `base` — audio transcription
- open_clip `ViT-B-32` (openai weights) — visual embeddings
- Salesforce BLIP base — image captioning
- OpenCV — video frame sampling
- FFmpeg — audio track extraction
- FastAPI + Streamlit (coming next) — API + demo UI

Everything is free and runs locally. No paid APIs.

## Status

- [x] Text ingestion + semantic search
- [x] Audio ingestion via Whisper, cross-modal text↔audio search
- [x] Video ingestion: Whisper transcript + CLIP frames + BLIP captions
- [x] RRF-fused multi-collection search with CLIP distance gating
- [ ] Audio-as-query (transcribe spoken query, then search)
- [ ] FastAPI endpoints
- [ ] Streamlit demo UI
- [ ] Project report + walkthrough video

## Quickstart

```powershell
# Create venv and install
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Ingest the included sample files
python -m scripts.ingest_text_test
python -m scripts.ingest_audio_test
python -m scripts.ingest_video_test

# Search
python -m scripts.search "climate change and greenhouse gases"
python -m scripts.search "a person at a desk with a laptop" --debug
```

## Notes on sample data

`data/text/` contains three short articles (~1 KB each) on AI history, climate change, and cooking pasta.

`data/audio/recording_climate.mp4` and `data/video/workstation_ai.mp4` are NOT checked in (gitignored), since they're media files. The audio is a voice recording of the climate change script; the video is an AI-generated clip (Google Gemini) of a person at a workstation discussing transformers and CLIP. Both are aligned topically with the text articles so cross-modal retrieval demos cleanly show one query returning results from multiple modalities.

If you want to reproduce the demo, you can drop your own short audio/video files into `data/audio/` and `data/video/` and run the ingest scripts.

## License

MIT (or to be decided before final submission).
