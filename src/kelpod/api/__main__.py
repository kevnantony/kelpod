"""``python -m kelpod.api [--host 127.0.0.1] [--port 8000]``: serve the API with uvicorn."""

from __future__ import annotations

import argparse
import os


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m kelpod.api", description=__doc__)
    p.add_argument("--host", default="127.0.0.1", help="Bind address (0.0.0.0 for all)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--jobs-dir", help="Keep jobs here instead of a temp directory")
    p.add_argument("--max-upload-mb", type=float, help="Upload size limit (default 50)")
    p.add_argument("--log-level", default="info")
    args = p.parse_args(argv)
    if args.jobs_dir:
        os.environ["KELPOD_JOBS_DIR"] = args.jobs_dir
    if args.max_upload_mb is not None:
        os.environ["KELPOD_MAX_UPLOAD_MB"] = str(args.max_upload_mb)

    import uvicorn

    from kelpod.api import app_from_env

    uvicorn.run(app_from_env(), host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
