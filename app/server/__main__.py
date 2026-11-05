"""Run the playground: `python -m app.server`."""
from __future__ import annotations

import argparse
import webbrowser


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="auto-reload on edits")
    ap.add_argument("--no-open", action="store_true", help="do not open a browser")
    args = ap.parse_args()

    import uvicorn

    url = f"http://{args.host}:{args.port}"
    print(f"Stereo playground -> {url}   (API docs at {url}/docs)")
    if not args.no_open and not args.reload:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    uvicorn.run(
        "app.server.main:app", host=args.host, port=args.port,
        reload=args.reload, log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
