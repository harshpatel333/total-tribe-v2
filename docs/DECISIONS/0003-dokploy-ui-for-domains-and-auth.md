# ADR-0003: Dokploy UI owns domains, TLS, and basic-auth

- **Status:** Accepted
- **Date:** 2026-05-11

## Context

The host that serves this UI runs [Dokploy](https://dokploy.com) as its container orchestrator. Dokploy itself ships with Traefik and an in-app domain/middleware management UI. There are two reasonable places to define the public URL, TLS cert, and basic-auth middleware:

1. In the project's `docker-compose.yml`, via Traefik labels on the service.
2. In the Dokploy UI, via its **Domains** and **Middlewares** tabs.

Dokploy generates and injects Traefik labels at deploy time based on what the UI says. Any hand-written Traefik labels in `docker-compose.yml` either collide silently with Dokploy's injected labels, or are overwritten depending on Dokploy version. Either way, the result is unpredictable and undebuggable from the repo's perspective.

## Decision

All routing, TLS termination, and basic-auth middleware are configured in the **Dokploy UI**, not in the compose file. `docker-compose.yml` defines only the service, its GPU device reservation, its volumes, and its port mapping. There are **no Traefik labels** in the compose file.

The walkthrough in `docs/DEPLOYMENT.md` documents:

- Adding `tribev2.ws.coursebite.ai` as a domain bound to port 7860 with Let's Encrypt.
- Adding a `basicAuth` middleware with a precomputed `htpasswd` line, bound to the domain's router.

The MCP playbook in §8 of the deployment doc uses the corresponding `mcp__dokploy__*` tools to do the same thing programmatically.

## Consequences

**Positive:**

- The repo stays clean of Traefik-specific configuration. A reader looking at `docker-compose.yml` sees only application-relevant settings.
- Dokploy's own UI remains the single source of truth for routing. There is no risk of UI and repo disagreeing.
- TLS rotation, middleware swaps, and domain changes are operator-level concerns that do not require a code change or redeploy.

**Negative:**

- Production routing is **not reproducible from the repo alone**. Recreating the deployment on a fresh Dokploy instance requires the operator to follow `docs/DEPLOYMENT.md` (or run the MCP playbook). Acceptable for v1 because this is a single deployment.
- New operators must know that the compose file is intentionally label-free; the comment in `docker-compose.yml` notes this.

## Alternatives considered

- **Bare Traefik with manual labels in `docker-compose.yml`.** Rejected — more ops surface area (Traefik must be installed and managed by the operator), and the labels would conflict with Dokploy's injected labels on this host.
- **Caddy sidecar with automatic HTTPS.** Rejected — redundant with Dokploy's existing Traefik. Two reverse proxies in series for no benefit.
- **Nginx + certbot.** Rejected — no auto-Let's Encrypt integration in this setup; would require a separate cert-renewal cron and a writable `/etc/letsencrypt` volume. More moving parts than the UI flow.
