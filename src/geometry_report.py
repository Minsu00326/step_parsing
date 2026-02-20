from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _round_vec(v: List[float], ndigits: int = 6) -> List[float]:
    return [round(float(x), ndigits) for x in v]


def serialize_faces(faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for f in faces:
        d = {k: v for k, v in f.items() if not k.startswith("_")}
        if "center_of_mass" in d:
            d["center_of_mass"] = _round_vec(d["center_of_mass"])
        if "bbox_min" in d:
            d["bbox_min"] = _round_vec(d["bbox_min"])
        if "bbox_max" in d:
            d["bbox_max"] = _round_vec(d["bbox_max"])
        if "uv_bounds" in d and d["uv_bounds"] is not None:
            d["uv_bounds"] = _round_vec(d["uv_bounds"])
        if "local_origin" in d and d["local_origin"] is not None:
            d["local_origin"] = _round_vec(d["local_origin"])
        if "local_x_dir" in d and d["local_x_dir"] is not None:
            d["local_x_dir"] = _round_vec(d["local_x_dir"])
        if "local_y_dir" in d and d["local_y_dir"] is not None:
            d["local_y_dir"] = _round_vec(d["local_y_dir"])
        if "local_z_dir" in d and d["local_z_dir"] is not None:
            d["local_z_dir"] = _round_vec(d["local_z_dir"])
        if "face_span_local_xyz" in d and d["face_span_local_xyz"] is not None:
            d["face_span_local_xyz"] = _round_vec(d["face_span_local_xyz"])
        if "face_span_global_xyz" in d and d["face_span_global_xyz"] is not None:
            d["face_span_global_xyz"] = _round_vec(d["face_span_global_xyz"])
        if "axis_origin" in d and d["axis_origin"] is not None:
            d["axis_origin"] = _round_vec(d["axis_origin"])
        if "axis_direction" in d and d["axis_direction"] is not None:
            d["axis_direction"] = _round_vec(d["axis_direction"])
        if "cyl_axis_origin_est" in d and d["cyl_axis_origin_est"] is not None:
            d["cyl_axis_origin_est"] = _round_vec(d["cyl_axis_origin_est"])
        if "cyl_axis_direction_est" in d and d["cyl_axis_direction_est"] is not None:
            d["cyl_axis_direction_est"] = _round_vec(d["cyl_axis_direction_est"])
        if "normal_midpoint" in d and d["normal_midpoint"] is not None:
            d["normal_midpoint"] = _round_vec(d["normal_midpoint"])
        if "area" in d:
            d["area"] = round(float(d["area"]), 8)
        for key in [
            "face_length",
            "face_width",
            "face_thickness_est",
            "cyl_radius_est",
            "cyl_diameter_est",
            "cyl_height_est",
            "cyl_sweep_angle_rad_est",
            "cyl_sweep_angle_deg_est",
            "cyl_arc_length_est",
            "cyl_circumference_est",
        ]:
            if key in d and d[key] is not None:
                d[key] = round(float(d[key]), 8)
        if "edges" in d and isinstance(d["edges"], list):
            edges_out = []
            for e in d["edges"]:
                if not isinstance(e, dict):
                    continue
                ee = dict(e)
                if "length" in ee and ee["length"] is not None:
                    ee["length"] = round(float(ee["length"]), 8)
                if "radius" in ee and ee["radius"] is not None:
                    ee["radius"] = round(float(ee["radius"]), 8)
                for k in [
                    "start_point",
                    "end_point",
                    "mid_point",
                    "bbox_min",
                    "bbox_max",
                    "circle_center",
                    "circle_axis_direction",
                ]:
                    if isinstance(ee.get(k), list) and len(ee[k]) >= 3:
                        ee[k] = _round_vec([ee[k][0], ee[k][1], ee[k][2]], 6)
                if isinstance(ee.get("polyline"), list):
                    pts = []
                    for p in ee["polyline"]:
                        if isinstance(p, list) and len(p) >= 3:
                            pts.append(_round_vec([p[0], p[1], p[2]], 6))
                    ee["polyline"] = pts
                edges_out.append(ee)
            d["edges"] = edges_out
        out.append(d)
    return out


def write_faces_csv(faces_serialized: List[Dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    base_fields = [
        "face_id",
        "display_label",
        "surface_type",
        "surface_type_ko",
        "surface_desc_ko",
        "area",
        "center_of_mass",
        "bbox_min",
        "bbox_max",
        "uv_bounds",
        "edge_count",
        "dominant_edge_type",
        "dominant_edge_type_ko",
        "edge_type_counts",
        "edge_type_counts_ko",
        "edges_brief",
        "wire_count",
        "circle_loop_count",
        "wire_edge_counts",
        "edge_loop_type_counts",
        "edge_loop_type_counts_ko",
        "edge_loop_type_counts_step_entity",
        "edge_mix_summary_ko",
        "easy_hint_ko",
        "source_color_rgb",
        "source_color_hex",
        "source_color_alpha",
        "source_color_source",
        "length_unit_name",
        "length_unit_symbol",
        "length_unit_to_m",
        "angle_unit_name",
        "angle_unit_symbol",
        "angle_unit_to_rad",
        "source_layer_name",
        "source_layer_description",
        "source_layer_note",
        "source_layer_names",
        "source_layer_assignment_ids",
        "source_layer_ref_ids",
        "source_part_name",
        "source_part_name_raw",
        "source_part_is_guid",
        "source_part_names",
        "source_part_candidate_count",
        "source_part_note",
        "source_solid_id",
        "source_solid_ids",
        "source_solid_count",
        "step_advanced_face_id",
        "step_advanced_face_line",
        "step_advanced_face_expr",
        "step_surface_ref_id",
        "step_surface_entity_raw",
        "step_surface_entity_line",
        "step_surface_entity_expr",
        "step_surface_placement_ref_id",
        "step_surface_placement_entity_raw",
        "step_surface_placement_line",
        "step_surface_placement_expr",
        "step_surface_local_origin_ref_id",
        "step_surface_local_axis_ref_id",
        "step_surface_local_refdir_ref_id",
        "step_shell_id",
        "step_shell_ids",
        "step_shell_entity_raw",
        "step_manifold_solid_brep_id",
        "step_manifold_solid_brep_ids",
        "step_manifold_solid_brep_name",
        "step_manifold_solid_brep_names",
        "step_candidate_advanced_face_ids",
        "step_candidate_count",
        "step_transfer_surface_entity_raw",
        "step_transfer_surface_signature",
        "step_entity_hierarchy",
        "step_ref_mapping",
        "step_surface_mapping_consistent",
        "local_origin",
        "local_x_dir",
        "local_y_dir",
        "local_z_dir",
        "local_frame_status",
        "axis_tilt_to_global_z_deg",
        "face_length",
        "face_width",
        "face_thickness_est",
        "face_thickness_source",
        "face_size_basis",
        "face_span_local_xyz",
        "face_span_global_xyz",
        "cylindrical_like",
        "cylindrical_mode",
        "cyl_radius_est",
        "cyl_diameter_est",
        "cyl_height_est",
        "cyl_sweep_angle_rad_est",
        "cyl_sweep_angle_deg_est",
        "cyl_arc_length_est",
        "cyl_circumference_est",
        "cyl_axis_origin_est",
        "cyl_axis_direction_est",
        "cyl_circle_edge_count",
        "cyl_metric_source",
        "orientation_reversed",
        "normal_midpoint",
        "surface_step_entity",
        "mean_curvature",
        "gaussian_curvature",
        "min_curvature",
        "max_curvature",
        "curvature_eval_uv",
        "curvature_status",
        "contact_candidate_count",
        "contact_area_total",
        "contact_length_total",
        "contact_pairs_top",
        "contact_status",
        "metric_sources",
        "radius",
        "ref_radius",
        "semi_angle",
        "semi_angle_deg",
        "major_radius",
        "minor_radius",
        "axis_origin",
        "axis_direction",
        "center",
        "dominant_edge_step_entity",
        "edge_type_counts_step_entity",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=base_fields)
        w.writeheader()
        for row in faces_serialized:
            item = {k: row.get(k) for k in base_fields}
            for key in [
                "center_of_mass",
                "bbox_min",
                "bbox_max",
                "uv_bounds",
                "normal_midpoint",
                "edge_type_counts",
                "edge_type_counts_ko",
                "edges_brief",
                "wire_edge_counts",
                "edge_loop_type_counts",
                "edge_loop_type_counts_ko",
                "edge_loop_type_counts_step_entity",
                "source_color_rgb",
                "source_layer_names",
                "source_layer_assignment_ids",
                "source_layer_ref_ids",
                "source_part_names",
                "source_solid_ids",
                "step_shell_ids",
                "step_manifold_solid_brep_ids",
                "step_manifold_solid_brep_names",
                "step_candidate_advanced_face_ids",
                "step_transfer_surface_signature",
                "local_origin",
                "local_x_dir",
                "local_y_dir",
                "local_z_dir",
                "face_span_local_xyz",
                "face_span_global_xyz",
                "cyl_axis_origin_est",
                "cyl_axis_direction_est",
                "axis_origin",
                "axis_direction",
                "center",
                "edge_type_counts_step_entity",
                "curvature_eval_uv",
                "contact_pairs_top",
                "metric_sources",
            ]:
                if item.get(key) is not None:
                    item[key] = json.dumps(item[key])
            w.writerow(item)


def write_report_json(
    report_path: Path,
    header_info: Dict[str, Any],
    counts: Dict[str, int],
    global_bbox: Dict[str, List[float]],
    total_area: float,
    total_volume: float | None,
    faces_serialized: List[Dict[str, Any]],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "step_header": header_info,
        "counts": counts,
        "global_bounding_box": {
            "min": _round_vec(global_bbox["min"]),
            "max": _round_vec(global_bbox["max"]),
        },
        "total_surface_area": round(float(total_area), 8),
        "total_volume": None if total_volume is None else round(float(total_volume), 8),
        "faces": faces_serialized,
        "faces_by_id": {f["face_id"]: f for f in faces_serialized},
    }

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
