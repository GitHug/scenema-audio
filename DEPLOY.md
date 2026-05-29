# Deploying the Podcast Pipeline (Hermes → MCP → Scenema → Telegram)

End-to-end runbook for the split deployment this repo's podcast feature was built for:

- **Gigatron** — GPU box (Windows + RTX 5080), runs Scenema in Docker. Not always-on.
- **Mini** — small always-on Ubuntu box, runs the [Hermes](https://hermes-agent.nousresearch.com/docs/) agent and the Scenema MCP server.
- The two reach each other over a **Tailscale** tailnet (no auth — Tailscale is the trust boundary).

Goal: ask Hermes for a podcast → it generates the transcript, drives Scenema over MCP, then
**downloads the finished MP3 and sends it to you on Telegram**, so it's playable offline even when
Gigatron is powered off.

> PowerShell blocks run on **Gigatron** (Windows); bash blocks run on **Mini** (Ubuntu).
> Wherever `gigatron` appears, substitute the Tailscale `100.x` address if MagicDNS doesn't resolve.

---

## Phase 0 — Tailscale (one-time, both machines)

1. Install Tailscale on both machines and log into the same tailnet.
2. Enable **MagicDNS** (Tailscale admin console → DNS) so `gigatron` resolves as a hostname.
3. From Mini, confirm reachability:
   ```bash
   tailscale status        # should list "gigatron"
   ping -c1 gigatron
   ```

---

## Phase 1 — Gigatron: configure & run Scenema

1. **Get a HuggingFace token** — Gemma 3 12B is gated. Accept the license at
   <https://huggingface.co/google/gemma-3-12b-it>, then create a read token.

2. **Clone the repo and create `.env`** (PowerShell):
   ```powershell
   git clone https://github.com/GitHug/scenema-audio.git
   cd scenema-audio
   copy .env.example .env
   ```
   Edit `.env`:
   ```
   HF_TOKEN=hf_your_token_here
   PUBLIC_BASE_URL=http://gigatron:8000
   ```
   `.env` is git-ignored. `PUBLIC_BASE_URL` is what makes the returned `audio_url` point at
   `gigatron` instead of `localhost` — essential, since Mini is the one downloading it.

3. **Confirm Docker can see the GPU** — Docker Desktop → WSL2 backend + GPU support enabled;
   `nvidia-smi` works inside WSL.

4. **Build & run** (first run downloads ~38 GB of models — long wait):
   ```powershell
   docker compose up --build
   ```
   Wait for the log line showing the server listening on `:8000`.

5. **Open the port through Windows Firewall** (PowerShell as admin, one-time) so Mini can reach
   it over Tailscale:
   ```powershell
   New-NetFirewallRule -DisplayName "Scenema 8000" -Direction Inbound -Port 8000 -Protocol TCP -Action Allow -Profile Any
   ```

6. **Forward the port into WSL2** — Docker Desktop on WSL2 binds to `127.0.0.1`, not `0.0.0.0`,
   so external clients (Mini over Tailscale) can't reach it even with the firewall open. Create a
   Windows port proxy that forwards to the WSL2 VM's IP (PowerShell as admin):
   ```powershell
   # Get the WSL2 VM's current IP
   wsl hostname -I
   # e.g. 172.17.169.100 ...  (first IP is the one you want)

   netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=8000 connectaddress=<WSL2_IP> connectport=8000
   ```
   Verify:
   ```powershell
   netsh interface portproxy show all
   ```
   > **Note:** WSL2 IPs change on reboot. If connectivity breaks after a restart, re-check with
   > `wsl hostname -I` and update the proxy:
   > ```powershell
   > netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=8000
   > netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=8000 connectaddress=<NEW_WSL2_IP> connectport=8000
   > ```

---

## Phase 2 — Verify the API from Mini

Prove Mini → Scenema works before involving Hermes:

```bash
curl http://gigatron:8000/health

curl -X POST http://gigatron:8000/podcast \
  -H 'Content-Type: application/json' \
  -d '{
    "format": "multi",
    "title": "Smoke Test",
    "transcript": "HOST: Welcome to the test.\nGUEST: Glad to be here.",
    "speakers": {
      "HOST":  {"description": "A warm male host in his 40s", "gender": "male"},
      "GUEST": {"description": "A bright female guest",        "gender": "female"}
    }
  }'
# -> {"job_id":"...","status":"queued","audio_url":"http://gigatron:8000/podcast/.../audio.mp3"}
```

Poll until done, then download:
```bash
JOB=<job_id>
curl http://gigatron:8000/podcast/$JOB          # queued -> running -> succeeded
curl -o test.mp3 http://gigatron:8000/podcast/$JOB/audio.mp3
```

If `test.mp3` plays, the core works. A real 20-minute podcast is many turns and takes several
minutes — poll on an interval; don't expect instant.

---

## Phase 3 — (optional) Save a reusable voice

Upload a clean 10–20s, single-speaker clip once to clone a voice:
```bash
curl -X POST http://gigatron:8000/voices \
  -F name="My Voice" \
  -F description="Calm narrator, mid 30s" \
  -F gender=male \
  -F file=@/path/to/my_clip.wav
# -> {"voice_id":"my-voice", ...}
```
Reference it later with `{"voice_id": "my-voice"}`. To clone from YouTube, extract the audio first
— see [`mcp/README.md`](mcp/README.md).

---

## Phase 4 — Mini: deploy the MCP server

```bash
git clone https://github.com/GitHug/scenema-audio.git
cd scenema-audio
python3 -m venv .venv && . .venv/bin/activate
pip install -r mcp/requirements-mcp.txt

# Smoke test it starts (Gigatron must be up). It waits silently on stdio — Ctrl-C to exit.
SCENEMA_API_URL=http://gigatron:8000 PYTHONPATH=src python -m mcp_server
```
No error = good. Record the absolute path (`pwd`) for the next step.

---

## Phase 5 — Register the MCP server with Hermes

Add to Hermes's MCP configuration (`~/.hermes/config.yaml`). Point `command` at the **venv's**
python so the `mcp`/`httpx` deps resolve, and use your absolute paths:

```yaml
# In ~/.hermes/config.yaml, add under (or create) the mcp_servers: key:
mcp_servers:
  scenema-audio:
    command: /home/youruser/scenema-audio/.venv/bin/python
    args: ["-m", "mcp_server"]
    env:
      PYTHONPATH: /home/youruser/scenema-audio/src
      SCENEMA_API_URL: http://gigatron:8000
```

Run `/reload-mcp` in Hermes (or restart it). It should now expose four tools: `create_podcast`,
`get_podcast_status`, `list_voices`, `create_voice`.

---

## Phase 6 — Drive it from Hermes

Talk to Hermes over Telegram, e.g.:

> "Write a 15-minute two-host podcast explaining how a steam engine works, then generate it with
> Scenema and send me the audio."

Hermes writes the transcript, calls `create_podcast(...)`, gets a `job_id` + `audio_url`, and polls
`get_podcast_status(job_id)` until `status == "succeeded"`.

> Tell Hermes explicitly to **poll until succeeded** and that it may take several minutes, so it
> doesn't give up early. If Gigatron is asleep, the tool returns an `{error, detail}` dict — wake
> Gigatron and have Hermes retry.

---

## Phase 7 — Listen (offline-capable, via Telegram)

Telegram's servers can't reach a Tailscale URL, so Hermes **downloads the bytes itself and uploads
them** — the result lands in your Telegram and plays anytime, even when Gigatron is off.

Instruct Hermes that, once the job succeeds, it should:
1. `GET` the `audio_url` to fetch the MP3 bytes (it has web/`execute_code` access over Tailscale).
2. Send via Telegram `sendAudio` with the file plus `title` and `duration` (use `duration_s` from
   the status response) for nicer playback metadata.

A 20-minute @128 kbps MP3 is ~19 MB — well under Telegram's 50 MB bot upload limit. Hermes keeps
the pulled MP3 in its own library on Mini, so your podcast collection persists independently of
Gigatron.

**Quick alternative (Gigatron must be on):** any tailnet device can stream directly by opening
`http://gigatron:8000/podcast/<job_id>/audio.mp3` in a browser or podcast app.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `audio_url` says `localhost` | `PUBLIC_BASE_URL` not set in Gigatron's `.env` |
| Mini can't reach `gigatron:8000` (timeout) | MagicDNS off (use the `100.x` IP), Windows Firewall rule missing, or port proxy not set |
| Mini connects but gets empty reply | Port proxy points to `127.0.0.1` instead of the WSL2 IP — update with `netsh interface portproxy` |
| Connectivity breaks after Gigatron reboot | WSL2 IP changed — re-check with `wsl hostname -I` and update the port proxy |
| Server won't start, Gemma error | `HF_TOKEN` missing or model license not accepted |
| Hermes reports "tool not found" | venv path wrong in MCP config, or Hermes not restarted |
| Job goes `running` → `failed` after a reboot | Gigatron powered off mid-job — no auto-resume by design; resubmit |
| Cloned voice sounds off | reference clip too short / noisy / multi-speaker — use clean 10–20s solo speech |
| Telegram upload rejected | file > 50 MB — lower `MP3_BITRATE` or shorten the podcast |
