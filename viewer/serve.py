from __future__ import annotations

import argparse
import http.server
import json
import socketserver
from pathlib import Path
from urllib.parse import urlsplit


def _to_web_path(path: Path, root: Path) -> str:
    return "/" + path.relative_to(root).as_posix()


def _find_model_bundles(root: Path) -> list[dict]:
    bundles: list[dict] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if not child.name.startswith("out"):
            continue
        # Collect model/report pairs in each out* directory:
        # model.obj <-> report.json
        # model_backup.obj <-> report_backup.json
        # model_xxx.obj <-> report_xxx.json
        for model in sorted(child.glob("model*.obj")):
            suffix = model.stem[len("model") :]  # "", "_backup", ...
            report = child / f"report{suffix}.json"
            if not report.exists():
                continue
            mtime = max(model.stat().st_mtime, report.stat().st_mtime)
            if child.name == "out_v2" and suffix == "":
                name = "out_v2 (추천)"
            elif child.name == "out" and suffix == "":
                name = "out (구버전)"
            else:
                name = f"{child.name}{suffix}"
            bundles.append(
                {
                    "name": name,
                    "model": _to_web_path(model, root),
                    "report": _to_web_path(report, root),
                    "mtime": mtime,
                }
            )

    def _folder_priority(model_web_path: str) -> int:
        p = str(model_web_path)
        if p.startswith("/out_v2/"):
            return 0
        if p.startswith("/out/"):
            return 1
        return 2

    bundles.sort(
        key=lambda x: (
            _folder_priority(x.get("model", "")),
            -x["mtime"],
            x["name"],
        )
    )
    return bundles


def _make_handler(root: Path):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            super().end_headers()

        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.path == "/api/models":
                payload = {"models": _find_model_bundles(root)}
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            return super().do_GET()

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve repo root so viewer and out are both reachable")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()

    root = Path(args.root).resolve()
    handler = _make_handler(root)

    with socketserver.TCPServer(("", args.port), handler) as httpd:
        print(f"Serving {root} at http://127.0.0.1:{args.port}/viewer/")
        import os
        os.chdir(root)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
