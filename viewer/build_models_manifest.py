from __future__ import annotations

import argparse
import json
from pathlib import Path


def _to_web_path(path: Path, root: Path) -> str:
    return "/" + path.relative_to(root).as_posix()


def _find_model_bundles(root: Path) -> list[dict]:
    bundles: list[dict] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not child.name.startswith("out"):
            continue
        for model in sorted(child.glob("model*.obj")):
            suffix = model.stem[len("model") :]
            report = child / f"report{suffix}.json"
            if not report.exists():
                continue
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
                    "mtime": max(model.stat().st_mtime, report.stat().st_mtime),
                }
            )

    def _folder_priority(model_web_path: str) -> int:
        if model_web_path.startswith("/out_v2/"):
            return 0
        if model_web_path.startswith("/out/"):
            return 1
        return 2

    bundles.sort(key=lambda x: (_folder_priority(str(x["model"])), -float(x["mtime"]), str(x["name"])))
    return bundles


def main() -> None:
    parser = argparse.ArgumentParser(description="Build viewer/models.json for static hosting")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_path = Path(args.out).resolve() if args.out else (root / "viewer" / "models.json")
    bundles = _find_model_bundles(root)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"models": [{"name": b["name"], "model": b["model"], "report": b["report"]} for b in bundles]}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} (models={len(payload['models'])})")


if __name__ == "__main__":
    main()
