# Deployment

This project is deployed via [Dokploy](https://dokploy.com) onto a single-GPU host. All routing, TLS, and basic-auth are configured in the Dokploy UI — there are no Traefik labels in `docker-compose.yml`, per [ADR-0003](./DECISIONS/0003-dokploy-ui-for-domains-and-auth.md).

## 1. Prerequisites

- Dokploy is already installed and pointing at a host with **at least one 24 GB NVIDIA GPU** (RTX 3090 / RTX 4090 / A5000 or better).
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is installed on the host so containers can see the GPU.
- A Cloudflare (or equivalent) DNS **A-record** for `tribev2.ws.coursebite.ai` is pointed at the host's public IP.
- A Hugging Face access token with read scope, and **LLaMA-3.2-3B access approved** on <https://huggingface.co/meta-llama/Llama-3.2-3B> for the same account. Without this, text input cannot work and TRIBE v2 itself cannot load.

## 2. Create the application

In the Dokploy UI:

1. **New Application** → **Docker Compose**.
2. Connect this repo's git URL (`<TODO: URL after publish>`).
3. Branch: `main`. Compose file: `docker-compose.yml`.
4. Confirm Dokploy detects a single service (`tribev2`) with a GPU device reservation.

## 3. Environment variables

In the application's **Environment** tab in the Dokploy UI, set:

| Variable          | Value                                                                                 | Notes                                                              |
| ----------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `HF_TOKEN`        | `hf_...`                                                                              | Personal Hugging Face token with read scope.                       |
| `ENABLE_TEXT`     | `true` (default) or `false`                                                           | `false` disables the text tab and avoids loading LLaMA.            |
| `BASIC_AUTH_USER` | e.g. `harsh`                                                                          | Used by the Dokploy middleware (step 5), not the container itself. |
| `BASIC_AUTH_PASS` | strong password                                                                       | Same.                                                              |

Do **not** put these in the repo's `.env.example` with real values.

## 4. Domain

In the application's **Domains** tab in the Dokploy UI:

1. Click **Add Domain**.
2. Host: `tribev2.ws.coursebite.ai`.
3. Container port: `7860` (Gradio default).
4. Path: `/`.
5. **Enable Let's Encrypt**.

Dokploy will inject the Traefik labels for the certificate challenge and router. Wait for the cert to be issued before testing HTTPS.

## 5. Basic-auth middleware

This deployment is single-user. Add a Dokploy `basicAuth` middleware bound to the domain so that public unauthenticated access is blocked:

1. **Middlewares** tab → **Add Middleware** → **basicAuth**.
2. Generate an `htpasswd`-style line offline (do NOT put plaintext credentials in the UI):

   ```bash
   htpasswd -nbB <user> <pass>
   ```

3. Paste the resulting `user:$2y$05$...` line into the middleware's `users` field.
4. Attach the middleware to the `tribev2.ws.coursebite.ai` router.

## 6. Deploy

Click **Deploy** in the Dokploy UI. The build will:

- Pull the CUDA-12.4 base image.
- Install Python deps from `pyproject.toml` (including `tribev2[plotting]` from git).
- Bring the container up with GPU device passthrough.

On the **first request**, the container downloads the TRIBE v2 checkpoint into `/app/cache` and (if `ENABLE_TEXT=true`) downloads LLaMA-3.2-3B. This is slow — tens of GB and several minutes — and is single-threaded. Subsequent requests reuse the cache.

## 7. Smoke test

After the deploy finishes, run the smoke test from a workstation with access to `https://tribev2.ws.coursebite.ai/`:

```bash
python scripts/smoke_test.py \
  --url https://tribev2.ws.coursebite.ai/ \
  --user "$BASIC_AUTH_USER" \
  --pass "$BASIC_AUTH_PASS" \
  --clip path/to/face_clip.mp4
```

Expected result: the top regions table should include **L_FFC** and/or **R_FFC** (fusiform face complex). If not, see [`docs/RISKS.md`](./RISKS.md) row "`CreateVideosFromImages` quality for V-JEPA2".

## 8. Claude Code automation playbook (Dokploy MCP)

Most steps above can be driven by an LLM agent through the Dokploy MCP tools, if available in your harness. A canonical sequence:

1. `mcp__dokploy__project_create` — create the project from the repo URL.
2. `mcp__dokploy__set_env` — set `HF_TOKEN`, `ENABLE_TEXT`, `BASIC_AUTH_USER`, `BASIC_AUTH_PASS`.
3. `mcp__dokploy__domain_add` — add `tribev2.ws.coursebite.ai` on port 7860 with Let's Encrypt.
4. `mcp__dokploy__middleware_basicauth_add` — paste the precomputed `htpasswd` line and bind to the domain router.
5. `mcp__dokploy__app_deploy` — kick off the build.
6. Poll `mcp__dokploy__app_status` until the deployment reports `running`.
7. Issue a plain `HTTP GET` to `https://tribev2.ws.coursebite.ai/` with basic-auth and confirm HTTP 200 plus a Gradio HTML title.

If the Dokploy MCP tools are not available in your harness, fall back to the UI steps in §2–§6 above. There is no scripted CLI fallback at scaffold time.
