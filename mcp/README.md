# Scenema Audio — MCP Server

A thin [Model Context Protocol](https://modelcontextprotocol.io/) server that lets an AI agent
generate podcasts through Scenema's REST API. It runs **on the agent's machine**, not on the GPU
box, and proxies tool calls to Scenema over your private network (stdio transport — the agent
launches it as a subprocess; no port is exposed).

```
 Agent host (always-on)                          GPU host (Scenema, not always-on)
 ┌──────────────────────────┐   private net      ┌──────────────────────────────────┐
 │ Agent (e.g. Hermes)      │ ─────────────────▶ │ Scenema REST (FastAPI, :8000)     │
 │  └─ scenema-audio MCP    │  POST /podcast     │  ├─ /podcast (async jobs)         │
 │     (this server, stdio) │  GET  /podcast/{id}│  ├─ /podcast/{id}/audio.mp3       │
 │  └─ downloads the MP3 ───│◀── audio.mp3 ──────│  └─ /voices                       │
 │  └─ delivers it          │                    └──────────────────────────────────┘
 └──────────────────────────┘
```

The author's setup, for concreteness: Scenema runs in Docker on a Windows gaming PC with an
RTX 5080 ("Gigatron"); the agent (Hermes) runs on a small always-on Ubuntu box ("Mini"); the two
reach each other over a Tailscale tailnet. There is **no auth** — Tailscale is the trust boundary.
Adapt the hostnames to your own network.

## Tools exposed

| Tool | What it does |
|------|--------------|
| `create_podcast(transcript, speakers, format, title, language, scene, seed)` | Submit a transcript; returns `{job_id, status_url, audio_url}` immediately. |
| `get_podcast_status(job_id)` | Poll job status; returns progress and (when done) `audio_url` + `duration_s`. |
| `list_voices()` | List saved voice presets usable by `voice_id`. |
| `create_voice(name, description, gender)` | Create a description-only voice preset. |

Cloning from a **reference clip** is intentionally not done through MCP (don't ship audio bytes
through tool calls). Upload the clip directly to Scenema's REST `POST /voices` (multipart `file`
or base64 `reference_b64`), then reference the resulting `voice_id` in `create_podcast`.

## Install (on the agent host)

```bash
git clone https://github.com/ScenemaAI/scenema-audio.git
cd scenema-audio
python3 -m venv .venv && . .venv/bin/activate
pip install -r mcp/requirements-mcp.txt
```

(You only need `src/mcp_server.py` and `mcp/requirements-mcp.txt` — the rest of the repo is the
GPU server and isn't required here.)

## Run

```bash
SCENEMA_API_URL=http://gigatron:8000 PYTHONPATH=src python -m mcp_server
```

| Env var | Default | Description |
|---------|---------|-------------|
| `SCENEMA_API_URL` | `http://localhost:8000` | Base URL of the Scenema REST API (your GPU host's private-network address — a MagicDNS name like `http://gigatron:8000`, or `http://100.x.x.x:8000`). |
| `SCENEMA_TIMEOUT_S` | `60` | Per-request HTTP timeout in seconds. |

The server speaks MCP over stdio, so normally you don't run it by hand — the agent spawns it.

## Register with the agent

Add it to your MCP client config. Generic shape (Claude Desktop / most MCP hosts):

```json
{
  "mcpServers": {
    "scenema-audio": {
      "command": "python",
      "args": ["-m", "mcp_server"],
      "env": {
        "PYTHONPATH": "/path/to/scenema-audio/src",
        "SCENEMA_API_URL": "http://gigatron:8000"
      }
    }
  }
}
```

For [Hermes Agent](https://hermes-agent.nousresearch.com/docs/), register the same command/env in
its MCP server configuration. Hermes can then call the tools directly.

## End-to-end flow

1. Agent calls `create_podcast(transcript=..., speakers=..., title=...)` → gets `job_id` + `audio_url`.
2. Agent polls `get_podcast_status(job_id)` until `status == "succeeded"`.
3. Agent **downloads `audio_url`** itself over the private network (e.g. `httpx.get`).
4. Agent delivers the file however it likes. For Telegram, the audio URL is private-network-only,
   so the agent uploads the bytes via `sendAudio` (bot upload limit is 50 MB; a 20-minute
   128 kbps MP3 is ~19 MB). Pass `title` and `duration` (from `duration_s`) for nicer playback.

## Notes

- The GPU host may not be always-on. If it's down, tool calls return an `{error, detail}` dict
  rather than raising — the agent can retry later.
- A job left `running` when Scenema restarts becomes `failed` ("interrupted by restart"); there
  is no auto-resume. Resubmit it.
