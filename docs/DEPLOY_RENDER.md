# Deploying a preview on Render

ChainTrace is designed to run offline. This deploys a **hosted preview on synthetic data** for demos; it is not the air-gapped setup.

## Steps
1. Push the repo to GitHub.
2. Render dashboard -> **New -> Blueprint** -> select the repo. `render.yaml` configures a Docker web service.
3. Wait for the build (several minutes: installs Python packages and builds the React UI).
4. Open the service URL. The first request after an idle spell is slow (free instances sleep, then the Forensics Lab dataset rebuilds, about 5 to 60 s depending on CPU).

## What the config does
* `PORT` is assigned by Render; `backend/main.py` reads it (falls back to `CHAINTRACE_PORT`, then 8000).
* `CHAINTRACE_DEFAULT_TX=8000` keeps the startup dataset small. Measured peak memory: about 240 MB at 8,000 transactions and
  280 MB at 20,000 (Windows measurement, process total). The free tier has 512 MB.
* Health check: `/health`.

## Limits to know
* **Untested on Render.** The Dockerfile has not been built locally (no Docker on the dev machine); expect to fix a small thing on first build.
* **Free tier CPU is tiny.** Model training uses all cores; on a fraction of a CPU, loading and "Retrain" can take minutes. Use a paid plan for live demos.
* **Ephemeral disk.** Analyst verdicts and cases (SQLite) are lost on restart or redeploy. Add a paid persistent disk mounted at `/app/state` to keep them.
* **No authentication.** Anyone with the URL can load data, submit verdicts and retrain. Keep it synthetic-only, or put the service behind a login/IP allow-list.
* **Larger requests can exhaust memory.** `POST /forensics/load` accepts up to 400,000 synthetic transactions. On a small instance that will crash it.
* **Elliptic++ files are not deployed** (large, git-ignored). The demo pages and Forensics Lab do not need them.
