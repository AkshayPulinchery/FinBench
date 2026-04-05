"""Validator-compatible server entrypoint."""

from __future__ import annotations

import os

import uvicorn

from app import app


def main(host: str | None = None, port: int | None = None) -> None:
    resolved_host = host or os.getenv("HOST", "0.0.0.0")
    resolved_port = port or int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host=resolved_host, port=resolved_port)


if __name__ == "__main__":
    main()
