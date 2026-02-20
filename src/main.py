from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

from .geometry_report import serialize_faces, write_faces_csv, write_report_json
from .step_loader import load_step_shape
from .tessellate_export import export_obj_by_face
from .topology_extract import annotate_contact_metrics, count_topology, extract_faces, global_bbox, total_area, total_volume


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="STEP face extraction + face-mapped OBJ export")
    p.add_argument("--input", required=True, help="Input STEP file path (.stp/.step)")
    p.add_argument("--out", default="out", help="Output directory")
    p.add_argument(
        "--output-stem",
        default=None,
        help="Output name stem (default: input filename stem). e.g. 'sample' -> model_sample.obj/report_sample.json/faces_sample.csv",
    )
    p.add_argument("--linear-deflection", type=float, default=0.2, help="Mesh linear deflection")
    p.add_argument("--angular-deflection", type=float, default=0.3, help="Mesh angular deflection (radian)")
    p.add_argument("--compute-contact", action="store_true", help="Compute face-face contact candidates (heavier)")
    p.add_argument("--contact-tolerance", type=float, default=1e-4, help="Distance tolerance for contact detection")
    p.add_argument("--contact-max-pairs", type=int, default=20000, help="Max candidate face pairs to evaluate for contact")
    p.add_argument("--contact-top-k", type=int, default=5, help="Max contact pair entries kept per face")
    p.add_argument("--log-level", default="INFO", help="DEBUG/INFO/WARN/ERROR")
    return p


def _safe_stem(raw: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    return s or "model"


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    logger = logging.getLogger("step-poc")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_stem = Path(args.input).stem
    stem = _safe_stem(args.output_stem or input_stem)
    out_obj = out_dir / f"model_{stem}.obj"
    out_csv = out_dir / f"faces_{stem}.csv"
    out_report = out_dir / f"report_{stem}.json"

    try:
        logger.info("Loading STEP: %s", args.input)
        shape, header, visuals = load_step_shape(args.input)

        logger.info("Extracting topology and face metadata")
        counts = count_topology(shape)
        bbox = global_bbox(shape)
        faces_raw = extract_faces(shape, include_normal=True, visual_context=visuals)
        if args.compute_contact:
            logger.info("Computing contact candidates (tol=%s, max_pairs=%s)", args.contact_tolerance, args.contact_max_pairs)
            contact_summary = annotate_contact_metrics(
                faces=faces_raw,
                tolerance=float(args.contact_tolerance),
                max_pair_checks=int(args.contact_max_pairs),
                top_k_per_face=int(args.contact_top_k),
            )
            header.setdefault("derived_analysis", {})
            header["derived_analysis"]["contact"] = contact_summary
        faces_serialized = serialize_faces(faces_raw)

        logger.info("Exporting OBJ grouped by FaceN")
        export_obj_by_face(
            shape=shape,
            faces=faces_raw,
            out_obj=out_obj,
            linear_deflection=args.linear_deflection,
            angular_deflection=args.angular_deflection,
        )

        logger.info("Writing reports")
        write_faces_csv(faces_serialized, out_csv)
        write_report_json(
            report_path=out_report,
            header_info=header,
            counts=counts,
            global_bbox=bbox,
            total_area=total_area(shape),
            total_volume=total_volume(shape),
            faces_serialized=faces_serialized,
        )

        logger.info("Done. outputs: %s, %s, %s", out_obj.resolve(), out_report.resolve(), out_csv.resolve())
        return 0

    except Exception:
        logger.exception("Failed to process STEP")
        return 1


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
