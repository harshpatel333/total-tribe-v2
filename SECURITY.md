# Security Policy

## Reporting a vulnerability

Please report security issues via GitHub private security advisory: `<TODO: repo URL once published>`.

Do **not** open a public issue for security-sensitive reports.

## Acknowledgment window

We will acknowledge receipt of your report within **7 days**. We do not commit to a specific remediation timeline beyond that — this is a side project and turnaround depends on severity and complexity.

## Scope

This policy covers vulnerabilities in **this project's code only** (`src/`, `scripts/`, `Dockerfile`, `docker-compose.yml`, and our packaged configuration).

It explicitly does **not** cover:

- Upstream `tribev2` (Meta FAIR) — report those at <https://github.com/facebookresearch/tribev2>.
- The TRIBE v2 model weights, LLaMA-3.2-3B weights, V-JEPA2 weights, W2v-BERT weights, or any other third-party model artifacts.
- User-supplied media content (text, audio, video, image inputs). We do not validate or moderate user-uploaded content beyond size/duration limits.

## Bug bounty

There is no bug bounty. Reports are appreciated but unfunded.

## Out of scope

The following are explicitly out of scope and will be closed as `informational` if reported:

- Vulnerabilities in third-party dependencies (`torch`, `gradio`, `nilearn`, etc.) unless we ship a code path that is uniquely vulnerable because of how we use them.
- Denial-of-service via oversized input files — we enforce a ≤30 s input duration cap upstream of any model invocation, and Dokploy's reverse proxy enforces additional limits.
- Issues that require an authenticated session as the single configured user (this is a single-user deployment behind Dokploy basic-auth by design).
- Social engineering of the maintainer.
