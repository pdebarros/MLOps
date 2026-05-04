# cursor-gitai-bridge

Small CLI next to the `gitai` folder that:

1. **Watches** `cursor-proposed-changes.ndjson` (output of `track_cursor_gitai.py`) and appends each parsed record to `data/git-ai-prompts.jsonl`.
2. **Asks Cursor** via `@cursor/sdk` (local runtime): one-shot `ask`, or multi-turn `session` with optional resume using the saved agent id.

This does **not** talk to the Cursor IDE “Ask” panel. It uses the **Cursor Cloud Agents API** with a local working directory (`local.cwd`), same as the official TypeScript SDK.

## Setup

```bash
cd cursor-gitai-bridge
npm install
cp .env.example .env   # add CURSOR_API_KEY
npm run build
```

### HTTP API (FastAPI)

Runs the same Node CLI via subprocess. Install Python deps and start uvicorn from this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export CURSOR_API_KEY=…     # required for /v1/ask and /v1/session
uvicorn app.main:app --host 127.0.0.1 --port 8765
```

Open **http://127.0.0.1:8765/docs** for interactive OpenAPI.

| Method | Path | Body / notes |
|--------|------|----------------|
| GET | `/health` | Node + built `dist/cli.js` present |
| POST | `/v1/ask` | `{"cwd": "/path/to/repo", "question": "..."}` |
| POST | `/v1/session` | `{"cwd": "...", "question": "...", "new": false, "agent_id": null, "data_dir": "data"}` |
| POST | `/v1/watch/start` | `{"file": "/path/to/cursor-proposed-changes.ndjson", "out_dir": "data", "from_start": false}` |
| POST | `/v1/watch/stop` | Stop background watch |
| GET | `/v1/watch/status` | Whether watch child process is running |

`data_dir` for session is resolved like the CLI (paths relative to the bridge folder work).

## Usage

```bash
export CURSOR_API_KEY=…   # or: source .env in your shell if you use direnv

# Tail git-ai NDJSON (default file name; override with --file)
node dist/cli.js watch --file ../gitai/cursor-proposed-changes.ndjson

# One-shot question against a repo checkout
node dist/cli.js ask --cwd ../gitai "What is tracker_worker.py?"

# Multi-turn: first message creates an agent; id stored under data/last-agent.json
node dist/cli.js session --cwd ../gitai "Summarize the desktop app"

# Next message continues that agent (reads last-agent.json)
node dist/cli.js session --cwd ../gitai "Focus on file tracking"

# Explicit resume or force new session
node dist/cli.js session --cwd ../gitai --agent-id bc-xxxx "Follow-up"
node dist/cli.js session --cwd ../gitai --new "Fresh thread"
```

Development without building:

```bash
npm run dev -- ask --cwd ../gitai "hello"
```

## Data layout

| Output | Purpose |
|--------|---------|
| `data/git-ai-prompts.jsonl` | One wrapper object per NDJSON line (`record` + metadata), written by `watch` |
| `data/last-agent.json` | Last Cursor agent id from `session` (for resume) |
