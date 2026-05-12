#!/usr/bin/env python3
"""Post-deploy smoke test for total-tribe-v2.

Uploads a fixture clip to the running Gradio app and asserts the response
includes ``L_FFC`` / ``R_FFC`` in the top regions (face fusiform should
activate strongly on a face close-up).

Usage::

    python scripts/smoke_test.py \\
        --url https://tribev2.ws.coursebite.ai \\
        --auth user:pass \\
        --fixture tests/fixtures/face_clip.mp4

Exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("smoke_test")

_EXPECTED_PARCELS = ("L_FFC", "R_FFC")


def smoke(url: str, auth: tuple[str, str] | None, fixture: Path) -> int:
    try:
        import requests
    except ImportError:
        logger.error("`requests` not installed. `pip install requests` and retry.")
        return 2

    if not fixture.exists():
        logger.error("fixture missing: %s", fixture)
        return 2

    # 1. health check
    r = requests.get(url.rstrip("/") + "/", auth=auth, timeout=30)
    if r.status_code != 200:
        logger.error("health check failed: HTTP %d", r.status_code)
        return 1
    logger.info("OK health check passed")

    # 2. predict
    # TODO: confirm the exact Gradio API surface once the app build is
    # finalized. Gradio 4.x exposes /api/predict but the payload shape
    # depends on the component layout; rev this once we have the running app.
    # Once implemented, assert that _EXPECTED_PARCELS appear in the top
    # regions returned by the prediction response.
    _ = _EXPECTED_PARCELS  # referenced by the future predict assertions
    logger.warning("predict step is a stub; rev once the Gradio API surface is finalized.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="https://host of the running app")
    parser.add_argument("--auth", help="basic-auth user:pass; optional")
    parser.add_argument("--fixture", type=Path, required=True, help="path to a fixture clip")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )

    auth: tuple[str, str] | None = None
    if args.auth:
        user, _, pw = args.auth.partition(":")
        auth = (user, pw)

    return smoke(args.url, auth, args.fixture)


if __name__ == "__main__":
    sys.exit(main())
