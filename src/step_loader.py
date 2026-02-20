from __future__ import annotations

from collections import defaultdict
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, Tuple

LOGGER = logging.getLogger(__name__)

USING_OCP = False

try:
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader

    USING_OCP = True
except Exception:
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Reader


def _named_entity_stats(text: str, entity_keyword: str) -> Dict[str, Any]:
    pattern = rf"\b{entity_keyword}\s*\(\s*'([^']*)'"
    raw_names = re.findall(pattern, text, flags=re.IGNORECASE)
    cleaned = [name.strip() for name in raw_names]
    named = [name for name in cleaned if name and name.upper() != "NONE"]
    return {
        "total": len(cleaned),
        "named": len(named),
        "none_or_empty": len(cleaned) - len(named),
        "sample_names": sorted(set(named))[:20],
    }


def _to_hex_channel(v: float) -> int:
    return max(0, min(255, int(round(v * 255.0))))


def _is_guid_like(name: str) -> bool:
    if re.fullmatch(r"[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12,}", name):
        return True
    if re.fullmatch(r"[0-9A-Fa-f\-]{24,}", name):
        return True
    return False


def _decode_step_string(raw: str | None) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    s = s.replace("''", "'")

    def _decode_x2(m: re.Match[str]) -> str:
        hex_data = re.sub(r"[^0-9A-Fa-f]", "", m.group(1))
        if not hex_data or len(hex_data) % 4 != 0:
            return m.group(0)
        try:
            return "".join(chr(int(hex_data[i : i + 4], 16)) for i in range(0, len(hex_data), 4))
        except Exception:
            return m.group(0)

    def _decode_x4(m: re.Match[str]) -> str:
        hex_data = re.sub(r"[^0-9A-Fa-f]", "", m.group(1))
        if not hex_data or len(hex_data) % 8 != 0:
            return m.group(0)
        try:
            return "".join(chr(int(hex_data[i : i + 8], 16)) for i in range(0, len(hex_data), 8))
        except Exception:
            return m.group(0)

    s = re.sub(r"\\X2\\(.*?)\\X0\\", _decode_x2, s, flags=re.IGNORECASE | re.DOTALL)
    s = re.sub(r"\\X4\\(.*?)\\X0\\", _decode_x4, s, flags=re.IGNORECASE | re.DOTALL)
    return s


def _parse_colour_palette(text: str) -> Dict[str, Any]:
    matches = re.findall(
        r"COLOUR_RGB\s*\(\s*'([^']*)'\s*,\s*([0-9.Ee+\-]+)\s*,\s*([0-9.Ee+\-]+)\s*,\s*([0-9.Ee+\-]+)\s*\)",
        text,
        flags=re.IGNORECASE,
    )
    counts: Dict[tuple[float, float, float], int] = {}
    for _, r, g, b in matches:
        key = (round(float(r), 9), round(float(g), 9), round(float(b), 9))
        counts[key] = counts.get(key, 0) + 1

    palette = []
    for (r, g, b), cnt in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        rr = _to_hex_channel(r)
        gg = _to_hex_channel(g)
        bb = _to_hex_channel(b)
        palette.append(
            {
                "rgb_0_1": [r, g, b],
                "rgb_255": [rr, gg, bb],
                "hex": f"#{rr:02X}{gg:02X}{bb:02X}",
                "count": cnt,
            }
        )

    return {
        "colour_rgb_total": len(matches),
        "colour_rgb_unique": len(palette),
        "palette_top": palette[:20],
    }


def _parse_layer_stats(text: str) -> Dict[str, Any]:
    entries = _parse_layer_assignments_with_face_refs(text)
    named = []
    for item in entries:
        n = str(item.get("name") or "").strip()
        d = str(item.get("description") or "").strip()
        if n or d:
            named.append(
                {
                    "assignment_id": item.get("assignment_id"),
                    "name": n,
                    "description": d,
                    "item_ref_count": int(item.get("item_ref_count") or 0),
                    "face_ref_count": int(item.get("face_ref_count") or 0),
                    "manifold_names": (item.get("manifold_names") or [])[:10],
                    "item_ref_types": item.get("item_ref_types") or {},
                    "edge_like_ref_count": int(item.get("edge_like_ref_count") or 0),
                    "edge_like_ref_ids": (item.get("edge_like_ref_ids") or [])[:20],
                    "item_refs_sample": (item.get("item_refs") or [])[:20],
                    "attachment_counts": item.get("attachment_counts") or {},
                    "attachment_scope": item.get("attachment_scope") or [],
                }
            )
    return {
        "layer_assignment_total": len(entries),
        "layer_assignment_named": len(named),
        "layer_named_samples": named[:20],
    }


def _parse_layer_assignments(text: str) -> list[Dict[str, Any]]:
    entries: list[Dict[str, Any]] = []
    layer_re = re.compile(
        r"(#\d+)\s*=\s*PRESENTATION_LAYER_ASSIGNMENT\s*\(\s*'([^']*)'\s*,\s*'([^']*)'\s*,\s*\((.*?)\)\s*\)\s*;",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for m in layer_re.finditer(text):
        assignment_id = m.group(1)
        name = _decode_step_string(m.group(2))
        description = _decode_step_string(m.group(3))
        refs_blob = m.group(4)
        item_refs = re.findall(r"#\d+", refs_blob)
        entries.append(
            {
                "assignment_id": assignment_id,
                "name": name,
                "description": description,
                "item_refs": item_refs,
                "item_ref_count": len(item_refs),
                "line": int(text.count("\n", 0, m.start()) + 1),
            }
        )
    return entries


def _parse_entity_type_map(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for m in re.finditer(r"(#\d+)\s*=\s*([A-Z0-9_]+)\s*\(", text, flags=re.IGNORECASE):
        out[m.group(1)] = m.group(2).upper()

    # Complex entity instances (e.g. B_SPLINE_SURFACE + RATIONAL_B_SPLINE_SURFACE)
    # are serialized as: #id = ( ... );
    for m in re.finditer(r"(#\d+)\s*=\s*\((.*?)\)\s*;", text, flags=re.IGNORECASE | re.DOTALL):
        rid = m.group(1)
        if rid in out:
            continue
        body_up = m.group(2).upper()
        if "B_SPLINE_SURFACE(" in body_up:
            out[rid] = "B_SPLINE_SURFACE"
        elif "B_SPLINE_CURVE(" in body_up:
            out[rid] = "B_SPLINE_CURVE"
        elif "BEZIER_SURFACE(" in body_up:
            out[rid] = "BEZIER_SURFACE"
        elif "BEZIER_CURVE(" in body_up:
            out[rid] = "BEZIER_CURVE"
        elif "SURFACE_OF_REVOLUTION(" in body_up:
            out[rid] = "SURFACE_OF_REVOLUTION"
        elif "SURFACE_OF_LINEAR_EXTRUSION(" in body_up:
            out[rid] = "SURFACE_OF_LINEAR_EXTRUSION"
        elif "CYLINDRICAL_SURFACE(" in body_up:
            out[rid] = "CYLINDRICAL_SURFACE"
        elif "CONICAL_SURFACE(" in body_up:
            out[rid] = "CONICAL_SURFACE"
        elif "SPHERICAL_SURFACE(" in body_up:
            out[rid] = "SPHERICAL_SURFACE"
        elif "TOROIDAL_SURFACE(" in body_up:
            out[rid] = "TOROIDAL_SURFACE"
        elif "PLANE(" in body_up:
            out[rid] = "PLANE"
    return out


def _parse_entity_statement_map(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for m in re.finditer(r"(#\d+)\s*=\s*(.*?)\s*;", text, flags=re.IGNORECASE | re.DOTALL):
        out[m.group(1)] = m.group(2)
    return out


def _parse_bspline_surface_signature_from_stmt(surface_stmt: str) -> Dict[str, int] | None:
    up = surface_stmt.upper()
    if "B_SPLINE_SURFACE(" not in up:
        return None

    m_deg = re.search(r"B_SPLINE_SURFACE\s*\(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*,", surface_stmt, flags=re.IGNORECASE | re.DOTALL)
    if not m_deg:
        return None
    u_degree = int(m_deg.group(1))
    v_degree = int(m_deg.group(2))

    cp_rows = 0
    cp_cols = 0
    m_cp = re.search(
        r"B_SPLINE_SURFACE\s*\(\s*[+-]?\d+\s*,\s*[+-]?\d+\s*,\s*\((.*?)\)\s*,\s*\.\w+\.",
        surface_stmt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m_cp:
        rows = re.findall(r"\(([^()]*#\d+[^()]*)\)", m_cp.group(1), flags=re.DOTALL)
        cp_rows = len(rows)
        cp_cols = max((len(re.findall(r"#\d+", r)) for r in rows), default=0)

    u_knot_count = 0
    v_knot_count = 0
    u_mult_count = 0
    v_mult_count = 0
    m_kn = re.search(
        r"B_SPLINE_SURFACE_WITH_KNOTS\s*\(\s*\((.*?)\)\s*,\s*\((.*?)\)\s*,\s*\((.*?)\)\s*,\s*\((.*?)\)",
        surface_stmt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m_kn:
        u_mult_count = len([x for x in re.split(r"\s*,\s*", m_kn.group(1).strip()) if x])
        v_mult_count = len([x for x in re.split(r"\s*,\s*", m_kn.group(2).strip()) if x])
        u_knot_count = len([x for x in re.split(r"\s*,\s*", m_kn.group(3).strip()) if x])
        v_knot_count = len([x for x in re.split(r"\s*,\s*", m_kn.group(4).strip()) if x])

    return {
        "u_degree": u_degree,
        "v_degree": v_degree,
        "cp_rows": cp_rows,
        "cp_cols": cp_cols,
        "u_knot_count": u_knot_count,
        "v_knot_count": v_knot_count,
        "u_mult_count": u_mult_count,
        "v_mult_count": v_mult_count,
    }


def _transient_surface_entity_name(surface_obj: Any) -> str | None:
    if surface_obj is None:
        return None
    tname = type(surface_obj).__name__.upper()
    if "BSPLINESURFACE" in tname:
        return "B_SPLINE_SURFACE"
    if "BEZIERSURFACE" in tname:
        return "BEZIER_SURFACE"
    if "CYLINDRICALSURFACE" in tname:
        return "CYLINDRICAL_SURFACE"
    if "CONICALSURFACE" in tname:
        return "CONICAL_SURFACE"
    if "SPHERICALSURFACE" in tname:
        return "SPHERICAL_SURFACE"
    if "TOROIDALSURFACE" in tname:
        return "TOROIDAL_SURFACE"
    if "SURFACEOFREVOLUTION" in tname:
        return "SURFACE_OF_REVOLUTION"
    if "SURFACEOFLINEAREXTRUSION" in tname:
        return "SURFACE_OF_LINEAR_EXTRUSION"
    if tname.endswith("_PLANE") or tname.endswith("PLANE"):
        return "PLANE"
    return None


def _transient_surface_signature(surface_obj: Any, surface_entity: str | None) -> Dict[str, int] | None:
    if surface_obj is None:
        return None
    if str(surface_entity or "").upper() != "B_SPLINE_SURFACE":
        return None
    try:
        return {
            "u_degree": int(surface_obj.UDegree()),
            "v_degree": int(surface_obj.VDegree()),
            "cp_rows": int(surface_obj.NbControlPointsListI()),
            "cp_cols": int(surface_obj.NbControlPointsListJ()),
            "u_knot_count": int(surface_obj.NbUKnots()),
            "v_knot_count": int(surface_obj.NbVKnots()),
            "u_mult_count": int(surface_obj.NbUMultiplicities()),
            "v_mult_count": int(surface_obj.NbVMultiplicities()),
        }
    except Exception:
        return None


def _transient_entity_name(ent: Any) -> str | None:
    if ent is None:
        return None
    try:
        n = ent.Name()
        if hasattr(n, "ToCString"):
            return _decode_step_string(n.ToCString())
    except Exception:
        return None
    return None


def _parse_shell_face_refs(text: str, entity_types: Dict[str, str]) -> Dict[str, list[str]]:
    out: Dict[str, list[str]] = {}
    shell_re = re.compile(r"(#\d+)\s*=\s*(CLOSED_SHELL|OPEN_SHELL)\s*\((.*?)\)\s*;", flags=re.IGNORECASE | re.DOTALL)
    for m in shell_re.finditer(text):
        shell_id = m.group(1)
        refs_all = re.findall(r"#\d+", m.group(3))
        face_refs = [rid for rid in refs_all if entity_types.get(rid) == "ADVANCED_FACE"]
        out[shell_id] = face_refs
    return out


def _parse_manifold_map(text: str) -> tuple[Dict[str, str], Dict[str, str]]:
    manifold_to_shell: Dict[str, str] = {}
    manifold_name: Dict[str, str] = {}
    re_manifold = re.compile(
        r"(#\d+)\s*=\s*MANIFOLD_SOLID_BREP\s*\(\s*'([^']*)'\s*,\s*(#\d+)\s*\)\s*;",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for m in re_manifold.finditer(text):
        mid = m.group(1)
        manifold_name[mid] = _decode_step_string(m.group(2))
        manifold_to_shell[mid] = m.group(3)
    return manifold_to_shell, manifold_name


def _parse_styled_item_targets(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    re_styled = re.compile(
        r"(#\d+)\s*=\s*STYLED_ITEM\s*\(\s*'[^']*'\s*,\s*\((?:.*?)\)\s*,\s*(#\d+)\s*\)\s*;",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for m in re_styled.finditer(text):
        out[m.group(1)] = m.group(2)
    return out


def _expand_ref_to_advanced_faces(
    ref_id: str,
    entity_types: Dict[str, str],
    shell_face_refs: Dict[str, list[str]],
    manifold_to_shell: Dict[str, str],
    styled_targets: Dict[str, str],
    visited: set[str] | None = None,
) -> set[str]:
    if not ref_id:
        return set()
    visited = visited or set()
    if ref_id in visited:
        return set()
    visited.add(ref_id)

    kind = entity_types.get(ref_id)
    if kind == "ADVANCED_FACE":
        return {ref_id}
    if kind in {"CLOSED_SHELL", "OPEN_SHELL"}:
        return set(shell_face_refs.get(ref_id, []))
    if kind == "MANIFOLD_SOLID_BREP":
        shell_id = manifold_to_shell.get(ref_id)
        if shell_id:
            return _expand_ref_to_advanced_faces(
                shell_id,
                entity_types=entity_types,
                shell_face_refs=shell_face_refs,
                manifold_to_shell=manifold_to_shell,
                styled_targets=styled_targets,
                visited=visited,
            )
        return set()
    if kind == "STYLED_ITEM":
        target_id = styled_targets.get(ref_id)
        if target_id:
            return _expand_ref_to_advanced_faces(
                target_id,
                entity_types=entity_types,
                shell_face_refs=shell_face_refs,
                manifold_to_shell=manifold_to_shell,
                styled_targets=styled_targets,
                visited=visited,
            )
        return set()
    return set()


def _sort_ref_ids(ids: set[str] | list[str]) -> list[str]:
    def _k(v: str) -> tuple[int, str]:
        m = re.match(r"#(\d+)$", str(v).strip())
        if m:
            return (0, f"{int(m.group(1)):012d}")
        return (1, str(v))

    return sorted(set(ids), key=_k)


def _build_face_to_shell_and_manifold_maps(
    text: str,
) -> tuple[Dict[str, list[str]], Dict[str, list[str]], Dict[str, str], Dict[str, str]]:
    entity_types = _parse_entity_type_map(text)
    shell_face_refs = _parse_shell_face_refs(text, entity_types)
    manifold_to_shell, manifold_name = _parse_manifold_map(text)

    face_to_shell: Dict[str, set[str]] = defaultdict(set)
    for shell_id, faces in shell_face_refs.items():
        for fid in faces:
            face_to_shell[fid].add(shell_id)

    shell_to_manifold: Dict[str, set[str]] = defaultdict(set)
    for mid, sid in manifold_to_shell.items():
        shell_to_manifold[sid].add(mid)

    face_to_shell_sorted: Dict[str, list[str]] = {
        fid: _sort_ref_ids(sids) for fid, sids in face_to_shell.items()
    }
    face_to_manifold_sorted: Dict[str, list[str]] = {}
    for fid, shell_ids in face_to_shell_sorted.items():
        mids: set[str] = set()
        for sid in shell_ids:
            mids.update(shell_to_manifold.get(sid, set()))
        face_to_manifold_sorted[fid] = _sort_ref_ids(mids)

    return face_to_shell_sorted, face_to_manifold_sorted, manifold_name, entity_types


def _part_candidate_score(candidate: Dict[str, Any]) -> tuple[int, int, str]:
    display = str(candidate.get("display_name") or "").strip()
    raw = str(candidate.get("raw_name") or "").strip()
    is_guid = bool(candidate.get("is_guid"))

    score = 0
    if is_guid:
        score += 100
    if not display:
        score += 30
    if display.startswith("Component"):
        score += 20
    if display.startswith("=>[") or "=>[" in display:
        score += 12
    if raw and _is_guid_like(raw):
        score += 40
    if raw.startswith("=>[") or "=>[" in raw:
        score += 8
    if re.search(r"[A-Za-z가-힣]", display):
        score -= 4
    return (score, len(display), display)


def _classify_layer_ref_type(kind_label: str) -> str:
    k = str(kind_label or "").upper()
    if "->" in k:
        k = k.split("->", 1)[1]
    if "MANIFOLD_SOLID_BREP" in k or k in {"CLOSED_SHELL", "OPEN_SHELL"}:
        return "part_or_solid"
    if "ADVANCED_FACE" in k:
        return "face"
    edge_curve_tokens = (
        "EDGE",
        "CURVE",
        "POLYLINE",
        "LINE",
        "CIRCLE",
        "ELLIPSE",
        "HYPERBOLA",
        "PARABOLA",
        "B_SPLINE",
        "BEZIER",
        "COMPOSITE_CURVE",
        "SURFACE_CURVE",
        "SEAM_CURVE",
        "VERTEX_POINT",
    )
    if any(tok in k for tok in edge_curve_tokens):
        return "edge_or_curve"
    if "STYLED_ITEM" in k:
        return "styled_item"
    return "other"


def _summarize_layer_attachment_counts(item_ref_types: Dict[str, int]) -> Dict[str, int]:
    out = {
        "part_or_solid": 0,
        "face": 0,
        "edge_or_curve": 0,
        "styled_item": 0,
        "other": 0,
    }
    for kind, cnt in (item_ref_types or {}).items():
        try:
            c = int(cnt)
        except Exception:
            continue
        bucket = _classify_layer_ref_type(kind)
        out[bucket] = int(out.get(bucket, 0)) + max(0, c)
    return out


def _parse_layer_assignments_with_face_refs(text: str) -> list[Dict[str, Any]]:
    layer_entries = _parse_layer_assignments(text)
    if len(layer_entries) == 0:
        return []

    entity_types = _parse_entity_type_map(text)
    shell_face_refs = _parse_shell_face_refs(text, entity_types)
    manifold_to_shell, manifold_name = _parse_manifold_map(text)
    styled_targets = _parse_styled_item_targets(text)

    out: list[Dict[str, Any]] = []
    edge_like_types = {
        "EDGE_CURVE",
        "ORIENTED_EDGE",
        "EDGE_LOOP",
        "TRIMMED_CURVE",
        "POLYLINE",
        "LINE",
        "CIRCLE",
        "ELLIPSE",
        "HYPERBOLA",
        "PARABOLA",
        "B_SPLINE_CURVE",
        "B_SPLINE_CURVE_WITH_KNOTS",
        "BEZIER_CURVE",
        "COMPOSITE_CURVE",
        "SURFACE_CURVE",
        "SEAM_CURVE",
        "VERTEX_POINT",
    }
    for item in layer_entries:
        item_refs = [str(x) for x in (item.get("item_refs") or [])]
        face_refs: set[str] = set()
        manifolds: set[str] = set()
        ref_type_counts: Dict[str, int] = {}
        edge_like_ref_ids: set[str] = set()
        for rid in item_refs:
            kind = entity_types.get(rid)
            kind_label = kind or "UNKNOWN"
            if kind in edge_like_types:
                edge_like_ref_ids.add(rid)
            if kind == "MANIFOLD_SOLID_BREP":
                nm = str(manifold_name.get(rid) or "").strip()
                if nm and nm.upper() != "NONE":
                    manifolds.add(nm)
            if kind == "STYLED_ITEM":
                target = styled_targets.get(rid)
                if target:
                    target_kind = entity_types.get(target)
                    kind_label = f"STYLED_ITEM->{target_kind or 'UNKNOWN'}"
                    if target_kind in edge_like_types:
                        edge_like_ref_ids.add(target)
                if target and entity_types.get(target) == "MANIFOLD_SOLID_BREP":
                    nm = str(manifold_name.get(target) or "").strip()
                    if nm and nm.upper() != "NONE":
                        manifolds.add(nm)
            ref_type_counts[kind_label] = int(ref_type_counts.get(kind_label, 0)) + 1

            face_refs.update(
                _expand_ref_to_advanced_faces(
                    rid,
                    entity_types=entity_types,
                    shell_face_refs=shell_face_refs,
                    manifold_to_shell=manifold_to_shell,
                    styled_targets=styled_targets,
                    visited=set(),
                )
            )

        row = dict(item)
        row["face_refs"] = _sort_ref_ids(face_refs)
        row["face_ref_count"] = len(row["face_refs"])
        row["manifold_names"] = sorted(manifolds)[:20]
        row["item_ref_types"] = {k: ref_type_counts[k] for k in sorted(ref_type_counts.keys())}
        row["edge_like_ref_ids"] = _sort_ref_ids(edge_like_ref_ids)
        row["edge_like_ref_count"] = len(row["edge_like_ref_ids"])
        row["attachment_counts"] = _summarize_layer_attachment_counts(row["item_ref_types"])
        row["attachment_scope"] = [
            k
            for k in ("part_or_solid", "face", "edge_or_curve", "styled_item", "other")
            if int(row["attachment_counts"].get(k, 0)) > 0
        ]
        out.append(row)

    return out


def _parse_styled_item_stats(text: str) -> Dict[str, Any]:
    names = re.findall(r"STYLED_ITEM\s*\(\s*'([^']*)'", text, flags=re.IGNORECASE)
    named = sorted(set([n.strip() for n in names if n.strip() and n.strip().upper() != "NONE"]))
    return {
        "styled_item_total": len(names),
        "styled_item_named": len(named),
        "styled_item_named_samples": named[:20],
    }


def _si_prefix_scale(prefix_raw: str | None) -> float:
    p = (prefix_raw or "$").strip().upper()
    mapping = {
        "$": 1.0,
        ".EXA.": 1e18,
        ".PETA.": 1e15,
        ".TERA.": 1e12,
        ".GIGA.": 1e9,
        ".MEGA.": 1e6,
        ".KILO.": 1e3,
        ".HECTO.": 1e2,
        ".DECA.": 1e1,
        ".DECI.": 1e-1,
        ".CENTI.": 1e-2,
        ".MILLI.": 1e-3,
        ".MICRO.": 1e-6,
        ".NANO.": 1e-9,
        ".PICO.": 1e-12,
        ".FEMTO.": 1e-15,
        ".ATTO.": 1e-18,
    }
    return float(mapping.get(p, 1.0))


def _parse_unit_system(text: str) -> Dict[str, Any]:
    length: Dict[str, Any] = {
        "name": "metre",
        "symbol": "m",
        "to_m": 1.0,
        "source": "DEFAULT",
        "line": None,
        "expr": None,
    }
    angle: Dict[str, Any] = {
        "name": "radian",
        "symbol": "rad",
        "to_rad": 1.0,
        "source": "DEFAULT",
        "line": None,
        "expr": None,
    }

    stmt_re = re.compile(r"(#\d+\s*=\s*\(.*?\)\s*;)", flags=re.IGNORECASE | re.DOTALL)
    for m in stmt_re.finditer(text):
        stmt = m.group(1)
        up = stmt.upper()

        if "LENGTH_UNIT" in up and "SI_UNIT" in up and ".METRE." in up:
            mp = re.search(r"SI_UNIT\s*\(\s*([^,\)]+)\s*,\s*\.METRE\.\s*\)", stmt, flags=re.IGNORECASE | re.DOTALL)
            prefix = mp.group(1).strip() if mp else "$"
            scale = _si_prefix_scale(prefix)
            if abs(scale - 1e-3) < 1e-18:
                name, symbol = "millimetre", "mm"
            elif abs(scale - 1e-2) < 1e-18:
                name, symbol = "centimetre", "cm"
            elif abs(scale - 1.0) < 1e-18:
                name, symbol = "metre", "m"
            else:
                name, symbol = f"{prefix} metre", "m"
            expr = re.sub(r"\s+", " ", stmt).strip()
            if len(expr) > 260:
                expr = expr[:257] + "..."
            length = {
                "name": name,
                "symbol": symbol,
                "to_m": float(scale),
                "source": "SI_UNIT_LENGTH",
                "line": int(text.count("\n", 0, m.start()) + 1),
                "expr": expr,
            }
            break

    for m in stmt_re.finditer(text):
        stmt = m.group(1)
        up = stmt.upper()

        if "PLANE_ANGLE_UNIT" in up and "SI_UNIT" in up and ".RADIAN." in up:
            expr = re.sub(r"\s+", " ", stmt).strip()
            if len(expr) > 260:
                expr = expr[:257] + "..."
            angle = {
                "name": "radian",
                "symbol": "rad",
                "to_rad": 1.0,
                "source": "SI_UNIT_ANGLE",
                "line": int(text.count("\n", 0, m.start()) + 1),
                "expr": expr,
            }
            break
        if "PLANE_ANGLE_UNIT" in up and "CONVERSION_BASED_UNIT" in up:
            mn = re.search(r"CONVERSION_BASED_UNIT\s*\(\s*'([^']+)'", stmt, flags=re.IGNORECASE | re.DOTALL)
            nm = (mn.group(1).strip() if mn else "conversion_angle")
            nup = nm.upper()
            to_rad = math.pi / 180.0 if ("DEG" in nup) else 1.0
            symbol = "deg" if ("DEG" in nup) else nm
            expr = re.sub(r"\s+", " ", stmt).strip()
            if len(expr) > 260:
                expr = expr[:257] + "..."
            angle = {
                "name": nm,
                "symbol": symbol,
                "to_rad": float(to_rad),
                "source": "CONVERSION_BASED_ANGLE",
                "line": int(text.count("\n", 0, m.start()) + 1),
                "expr": expr,
            }
            break

    return {"length": length, "angle": angle}


def _parse_advanced_face_refs(text: str) -> list[Dict[str, Any]]:
    entries: list[Dict[str, Any]] = []
    entity_types = _parse_entity_type_map(text)
    entity_statements = _parse_entity_statement_map(text)
    adv_re = re.compile(r"(#\d+)\s*=\s*ADVANCED_FACE\s*\(([^;]*)\)\s*;", flags=re.IGNORECASE | re.DOTALL)
    for m in adv_re.finditer(text):
        adv_id = m.group(1)
        body = m.group(2).strip()
        adv_line = int(text.count("\n", 0, m.start()) + 1)
        adv_expr = re.sub(r"\s+", " ", m.group(0)).strip()
        if len(adv_expr) > 260:
            adv_expr = adv_expr[:257] + "..."
        m_surf = re.search(r",\s*(#\d+)\s*,\s*\.(T|F)\.\s*$", body, flags=re.IGNORECASE | re.DOTALL)
        surface_ref = m_surf.group(1) if m_surf else None
        same_sense = True if not m_surf else (m_surf.group(2).upper() == "T")
        m_bounds = re.match(r"\s*'[^']*'\s*,\s*\((.*?)\)\s*,", body, flags=re.IGNORECASE | re.DOTALL)
        bounds_count = len(re.findall(r"#\d+", m_bounds.group(1))) if m_bounds else None
        surface_entity = None
        surface_signature = None
        surface_expr = None
        surface_line = None
        surface_placement_ref_id = None
        surface_placement_entity_raw = None
        surface_placement_line = None
        surface_placement_expr = None
        surface_local_origin_ref_id = None
        surface_local_axis_ref_id = None
        surface_local_refdir_ref_id = None
        if surface_ref:
            surface_entity = entity_types.get(surface_ref)
            stmt_body = entity_statements.get(surface_ref)
            if stmt_body:
                surface_stmt = f"{surface_ref}={stmt_body};"
                surface_expr = re.sub(r"\s+", " ", surface_stmt).strip()
                if len(surface_expr) > 260:
                    surface_expr = surface_expr[:257] + "..."
                m_stmt_loc = re.search(
                    rf"{re.escape(surface_ref)}\s*=\s*(?:\(.*?\)|[A-Z0-9_]+\s*\(.*?\))\s*;",
                    text,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                if m_stmt_loc:
                    surface_line = int(text.count("\n", 0, m_stmt_loc.start()) + 1)

                if str(surface_entity or "").upper() == "B_SPLINE_SURFACE":
                    surface_signature = _parse_bspline_surface_signature_from_stmt(stmt_body)

                m_place_ref = re.search(
                    r"=\s*[A-Z0-9_]+\s*\(\s*'[^']*'\s*,\s*(#\d+)",
                    stmt_body,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                if m_place_ref:
                    surface_placement_ref_id = m_place_ref.group(1)
                    surface_placement_entity_raw = entity_types.get(surface_placement_ref_id)
                    place_stmt_body = entity_statements.get(surface_placement_ref_id)
                    if place_stmt_body:
                        place_stmt = f"{surface_placement_ref_id}={place_stmt_body};"
                        surface_placement_expr = re.sub(r"\s+", " ", place_stmt).strip()
                        if len(surface_placement_expr) > 260:
                            surface_placement_expr = surface_placement_expr[:257] + "..."
                        m_place_loc = re.search(
                            rf"{re.escape(surface_placement_ref_id)}\s*=\s*(?:\(.*?\)|[A-Z0-9_]+\s*\(.*?\))\s*;",
                            text,
                            flags=re.IGNORECASE | re.DOTALL,
                        )
                        if m_place_loc:
                            surface_placement_line = int(text.count("\n", 0, m_place_loc.start()) + 1)
                        m_ax = re.search(
                            r"AXIS2_PLACEMENT_3D\s*\(\s*'[^']*'\s*,\s*(#\d+)\s*,\s*(#\d+)\s*,\s*(#\d+)\s*\)",
                            place_stmt_body,
                            flags=re.IGNORECASE | re.DOTALL,
                        )
                        if m_ax:
                            surface_local_origin_ref_id = m_ax.group(1)
                            surface_local_axis_ref_id = m_ax.group(2)
                            surface_local_refdir_ref_id = m_ax.group(3)
        entries.append(
            {
                "advanced_face_id": adv_id,
                "advanced_face_line": adv_line,
                "advanced_face_expr": adv_expr,
                "surface_ref_id": surface_ref,
                "same_sense": same_sense,
                "bounds_count": bounds_count,
                "surface_entity_raw": surface_entity,
                "surface_signature": surface_signature,
                "surface_entity_line": surface_line,
                "surface_entity_expr": surface_expr,
                "surface_placement_ref_id": surface_placement_ref_id,
                "surface_placement_entity_raw": surface_placement_entity_raw,
                "surface_placement_line": surface_placement_line,
                "surface_placement_expr": surface_placement_expr,
                "surface_local_origin_ref_id": surface_local_origin_ref_id,
                "surface_local_axis_ref_id": surface_local_axis_ref_id,
                "surface_local_refdir_ref_id": surface_local_refdir_ref_id,
            }
        )
    return entries


def _extract_human_named_tokens(text: str) -> Dict[str, Any]:
    def _uniq_keep_order(seq: list[str]) -> list[str]:
        seen = set()
        out: list[str] = []
        for item in seq:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    def _entity_names(keyword: str) -> list[str]:
        raw = re.findall(rf"\b{keyword}\s*\(\s*'([^']*)'", text, flags=re.IGNORECASE)
        clean = [x.strip() for x in raw]
        return _uniq_keep_order([x for x in clean if x and x.upper() != "NONE"])

    product_names = _entity_names("PRODUCT")
    solid_names = _entity_names("MANIFOLD_SOLID_BREP")
    rep_names = _entity_names("SHAPE_REPRESENTATION")
    # NEXT_ASSEMBLY_USAGE_OCCURRENCE(id, name, description, ...)
    nauo_names_raw = re.findall(
        r"\bNEXT_ASSEMBLY_USAGE_OCCURRENCE\s*\(\s*'[^']*'\s*,\s*'([^']*)'",
        text,
        flags=re.IGNORECASE,
    )
    assembly_instance_names = _uniq_keep_order([x.strip() for x in nauo_names_raw if x.strip() and x.strip().upper() != "NONE"])

    candidates = _uniq_keep_order(product_names + solid_names + rep_names + assembly_instance_names)
    human_candidates = [x for x in candidates if not _is_guid_like(x)]
    weld_pat = re.compile(r"(WELD|WELDING|SEAM|BEAD|FILLET|JOINT)(?:\b|_)", flags=re.IGNORECASE)
    weld_hits = [x for x in human_candidates if weld_pat.search(x)]

    return {
        "product_names": product_names[:200],
        "manifold_solid_brep_names": solid_names[:200],
        "shape_representation_names": rep_names[:200],
        "assembly_instance_names": assembly_instance_names[:200],
        "human_name_candidates": human_candidates[:200],
        "weld_related_name_hits": weld_hits[:200],
    }


def _extract_face_visuals_with_xcaf(step_path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "faces_by_id": {},
        "meta": {
            "color_source": None,
        },
    }

    try:
        from OCP.STEPCAFControl import STEPCAFControl_Reader
        from OCP.TCollection import TCollection_ExtendedString
        from OCP.TDataStd import TDataStd_Name
        from OCP.TDF import TDF_Label, TDF_LabelSequence
        from OCP.TDocStd import TDocStd_Document
        from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS
        from OCP.XCAFDoc import XCAFDoc_ColorCurv, XCAFDoc_ColorGen, XCAFDoc_ColorSurf, XCAFDoc_DocumentTool
        from OCP.Quantity import Quantity_Color
    except Exception:
        try:
            from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
            from OCC.Core.TCollection import TCollection_ExtendedString
            from OCC.Core.TDataStd import TDataStd_Name
            from OCC.Core.TDF import TDF_Label, TDF_LabelSequence
            from OCC.Core.TDocStd import TDocStd_Document
            from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_SOLID
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopoDS import topods
            from OCC.Core.XCAFDoc import XCAFDoc_ColorCurv, XCAFDoc_ColorGen, XCAFDoc_ColorSurf, XCAFDoc_DocumentTool
            from OCC.Core.Quantity import Quantity_Color
        except Exception:
            return result

        def _to_face(shape: Any) -> Any:
            return topods.Face(shape)

    else:
        def _to_face(shape: Any) -> Any:
            return TopoDS.Face_s(shape)

    def _label_name(label: Any) -> str | None:
        try:
            name_attr = TDataStd_Name()
            if label.FindAttribute(TDataStd_Name.GetID_s(), name_attr):
                raw = name_attr.Get().ToExtString().strip()
                if raw and raw.upper() != "NONE":
                    return raw
        except Exception:
            return None
        return None

    try:
        step_text = step_path.read_text(encoding="utf-8", errors="ignore")
        step_face_refs = _parse_advanced_face_refs(step_text)
        layer_entries = _parse_layer_assignments_with_face_refs(step_text)
        face_to_shell_ids, face_to_manifold_ids, manifold_name_map, entity_types_map = _build_face_to_shell_and_manifold_maps(step_text)
        for ref in step_face_refs:
            aid = str(ref.get("advanced_face_id") or "").strip()
            mids = face_to_manifold_ids.get(aid, [])
            mnames: list[str] = []
            for mid in mids:
                n = str(manifold_name_map.get(mid) or "").strip()
                if n and n.upper() != "NONE" and n not in mnames:
                    mnames.append(n)
            ref["manifold_ids"] = mids[:20]
            ref["manifold_names"] = mnames[:20]
        layers_by_face_ref: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
        for lay in layer_entries:
            for face_ref in lay.get("face_refs", []) or []:
                layers_by_face_ref[face_ref].append(lay)

        reader = STEPCAFControl_Reader()
        try:
            reader.SetColorMode(True)
            reader.SetNameMode(True)
            reader.SetLayerMode(True)
            reader.SetPropsMode(True)
        except Exception:
            pass
        status = reader.ReadFile(str(step_path))
        if status != IFSelect_RetDone:
            return result

        doc = TDocStd_Document(TCollection_ExtendedString("XmlXCAF"))
        if not reader.Transfer(doc):
            return result

        main = doc.Main()
        shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(main)
        color_tool = XCAFDoc_DocumentTool.ColorTool_s(main)
        one_shape = shape_tool.GetOneShape()
        if one_shape is None:
            return result
        transfer_reader = reader.Reader().WS().TransferReader()

        solid_step_name_rows: list[dict[str, Any]] = []
        sx_sol = TopExp_Explorer(one_shape, TopAbs_SOLID)
        while sx_sol.More():
            solid = sx_sol.Current()
            transient_solid = transfer_reader.EntityFromShapeResult(solid, 1)
            step_solid_name = _transient_entity_name(transient_solid)
            solid_faces: list[Any] = []
            sx_sf = TopExp_Explorer(solid, TopAbs_FACE)
            while sx_sf.More():
                solid_faces.append(_to_face(sx_sf.Current()))
                sx_sf.Next()
            solid_step_name_rows.append({"step_name": step_solid_name, "faces": solid_faces})
            sx_sol.Next()

        def _step_solid_name_for_face(face_shape: Any) -> str | None:
            for row in solid_step_name_rows:
                sname = str(row.get("step_name") or "").strip()
                if not sname:
                    continue
                for sf in row.get("faces", []):
                    try:
                        if face_shape.IsSame(sf):
                            return sname
                    except Exception:
                        continue
            return None

        def _shape_color(shape: Any) -> Dict[str, Any] | None:
            qc = Quantity_Color()
            source = None
            if color_tool.GetColor(shape, XCAFDoc_ColorSurf, qc):
                source = "XCAF_ColorSurf"
            elif color_tool.GetColor(shape, XCAFDoc_ColorGen, qc):
                source = "XCAF_ColorGen"
            elif color_tool.GetColor(shape, XCAFDoc_ColorCurv, qc):
                source = "XCAF_ColorCurv"
            elif color_tool.GetInstanceColor(shape, XCAFDoc_ColorSurf, qc):
                source = "XCAF_InstanceColorSurf"
            elif color_tool.GetInstanceColor(shape, XCAFDoc_ColorGen, qc):
                source = "XCAF_InstanceColorGen"
            elif color_tool.GetInstanceColor(shape, XCAFDoc_ColorCurv, qc):
                source = "XCAF_InstanceColorCurv"
            if source is None:
                return None
            r = float(qc.Red())
            g = float(qc.Green())
            b = float(qc.Blue())
            rr = _to_hex_channel(r)
            gg = _to_hex_channel(g)
            bb = _to_hex_channel(b)
            return {
                "source_color_rgb": [round(r, 9), round(g, 9), round(b, 9)],
                "source_color_hex": f"#{rr:02X}{gg:02X}{bb:02X}",
                "source_color_alpha": 1.0,
                "source_color_source": source,
            }

        ex = TopExp_Explorer(one_shape, TopAbs_FACE)
        idx = 1
        color_hit_face_direct = 0
        face_order: list[tuple[str, Any]] = []
        while ex.More():
            face = _to_face(ex.Current())
            face_id = f"Face{idx}"
            idx += 1
            face_order.append((face_id, face))
            row = result["faces_by_id"].setdefault(face_id, {})

            step_idx = idx - 2
            row["step_candidate_advanced_face_ids"] = None
            row["step_candidate_count"] = 0
            row["step_transfer_surface_entity_raw"] = None
            row["step_transfer_surface_signature"] = None

            ref = None
            mapping_mode = None
            transfer_surface_entity = None

            # Prefer transfer-entity signature matching (works for B-spline surfaces).
            try:
                transient_face_ent = transfer_reader.EntityFromShapeResult(face, 1)
            except Exception:
                transient_face_ent = None
            if transient_face_ent is not None:
                t_same_sense = None
                t_bounds_count = None
                t_surface_sig = None
                try:
                    t_same_sense = bool(transient_face_ent.SameSense())
                except Exception:
                    pass
                try:
                    t_bounds_count = int(transient_face_ent.NbBounds())
                except Exception:
                    pass
                try:
                    t_surface = transient_face_ent.FaceGeometry()
                except Exception:
                    t_surface = None
                transfer_surface_entity = _transient_surface_entity_name(t_surface)
                t_surface_sig = _transient_surface_signature(t_surface, transfer_surface_entity)
                row["step_transfer_surface_entity_raw"] = transfer_surface_entity
                row["step_transfer_surface_signature"] = t_surface_sig

                if transfer_surface_entity:
                    cands = [
                        x
                        for x in step_face_refs
                        if str(x.get("surface_entity_raw") or "").upper() == str(transfer_surface_entity).upper()
                    ]
                    if t_same_sense is not None:
                        cands = [x for x in cands if bool(x.get("same_sense")) == bool(t_same_sense)]
                    if t_bounds_count is not None:
                        cands = [x for x in cands if int(x.get("bounds_count") or 0) == int(t_bounds_count)]
                    if isinstance(t_surface_sig, dict):
                        cands_sig = [x for x in cands if isinstance(x.get("surface_signature"), dict) and x.get("surface_signature") == t_surface_sig]
                        if len(cands_sig) > 0:
                            cands = cands_sig
                    step_solid_name = _step_solid_name_for_face(face)
                    if step_solid_name:
                        cands_by_name = [x for x in cands if step_solid_name in (x.get("manifold_names") or [])]
                        if len(cands_by_name) > 0:
                            cands = cands_by_name

                    row["step_candidate_advanced_face_ids"] = [x.get("advanced_face_id") for x in cands[:30] if x.get("advanced_face_id")]
                    row["step_candidate_count"] = len(cands)

                    if len(cands) == 1:
                        ref = cands[0]
                        mapping_mode = "TRANSFER_SIGNATURE_MATCH"
                    elif len(cands) > 1:
                        mapping_mode = "TRANSFER_SIGNATURE_AMBIGUOUS"

            allow_order_fallback = not (
                mapping_mode == "TRANSFER_SIGNATURE_AMBIGUOUS" and str(transfer_surface_entity or "").upper() == "B_SPLINE_SURFACE"
            )
            if ref is None and allow_order_fallback and 0 <= step_idx < len(step_face_refs):
                ref = step_face_refs[step_idx]
                mapping_mode = "ORDER_BASED_BEST_EFFORT"

            if ref is not None:
                row["step_advanced_face_id"] = ref.get("advanced_face_id")
                row["step_advanced_face_line"] = ref.get("advanced_face_line")
                row["step_advanced_face_expr"] = ref.get("advanced_face_expr")
                row["step_surface_ref_id"] = ref.get("surface_ref_id")
                row["step_surface_entity_raw"] = ref.get("surface_entity_raw")
                row["step_surface_entity_line"] = ref.get("surface_entity_line")
                row["step_surface_entity_expr"] = ref.get("surface_entity_expr")
                row["step_surface_placement_ref_id"] = ref.get("surface_placement_ref_id")
                row["step_surface_placement_entity_raw"] = ref.get("surface_placement_entity_raw")
                row["step_surface_placement_line"] = ref.get("surface_placement_line")
                row["step_surface_placement_expr"] = ref.get("surface_placement_expr")
                row["step_surface_local_origin_ref_id"] = ref.get("surface_local_origin_ref_id")
                row["step_surface_local_axis_ref_id"] = ref.get("surface_local_axis_ref_id")
                row["step_surface_local_refdir_ref_id"] = ref.get("surface_local_refdir_ref_id")
                row["step_ref_mapping"] = mapping_mode

                adv_face_ref = ref.get("advanced_face_id")
                if isinstance(adv_face_ref, str) and adv_face_ref:
                    shell_ids = face_to_shell_ids.get(adv_face_ref, [])
                    manifold_ids = face_to_manifold_ids.get(adv_face_ref, [])
                    manifold_names: list[str] = []
                    for mid in manifold_ids:
                        mname = str(manifold_name_map.get(mid) or "").strip()
                        if mname and mname.upper() != "NONE" and mname not in manifold_names:
                            manifold_names.append(mname)

                    row["step_shell_ids"] = shell_ids[:20]
                    row["step_shell_id"] = shell_ids[0] if shell_ids else None
                    row["step_shell_entity_raw"] = (
                        entity_types_map.get(row["step_shell_id"]) if row.get("step_shell_id") else None
                    )
                    row["step_manifold_solid_brep_ids"] = manifold_ids[:20]
                    row["step_manifold_solid_brep_id"] = manifold_ids[0] if manifold_ids else None
                    row["step_manifold_solid_brep_names"] = manifold_names[:20]
                    row["step_manifold_solid_brep_name"] = manifold_names[0] if manifold_names else None

                    parts: list[str] = []
                    if row.get("step_manifold_solid_brep_id"):
                        mid = str(row["step_manifold_solid_brep_id"])
                        mname = str(row.get("step_manifold_solid_brep_name") or "").strip()
                        parts.append(f"{mid} MANIFOLD_SOLID_BREP('{mname}')" if mname else f"{mid} MANIFOLD_SOLID_BREP")
                    if row.get("step_shell_id"):
                        sid = str(row["step_shell_id"])
                        skind = str(row.get("step_shell_entity_raw") or "").strip()
                        parts.append(f"{sid} {skind}".strip())
                    parts.append(f"{adv_face_ref} ADVANCED_FACE")

                    sref = str(row.get("step_surface_ref_id") or "").strip()
                    sent = str(row.get("step_surface_entity_raw") or "").strip()
                    if sref:
                        parts.append(f"{sref} {sent}".strip())

                    pref = str(row.get("step_surface_placement_ref_id") or "").strip()
                    pent = str(row.get("step_surface_placement_entity_raw") or "").strip()
                    if pref:
                        parts.append(f"{pref} {pent}".strip())

                    row["step_entity_hierarchy"] = " -> ".join([p for p in parts if p])

                    layers = layers_by_face_ref.get(adv_face_ref, [])
                    if len(layers) > 0:
                        layer_names: list[str] = []
                        layer_assignment_ids: list[str] = []
                        layer_ref_ids: set[str] = set()
                        for layer in layers:
                            n = str(layer.get("name") or "").strip()
                            d = str(layer.get("description") or "").strip()
                            if n and d:
                                label = f"{n}|{d}"
                            elif n:
                                label = n
                            elif d:
                                label = d
                            else:
                                label = str(layer.get("assignment_id") or "(unnamed)")
                            if label not in layer_names:
                                layer_names.append(label)

                            aid = str(layer.get("assignment_id") or "").strip()
                            if aid and aid not in layer_assignment_ids:
                                layer_assignment_ids.append(aid)

                            for rid in layer.get("item_refs", []) or []:
                                if isinstance(rid, str):
                                    layer_ref_ids.add(rid)

                        first = layers[0]
                        first_name = str(first.get("name") or "").strip()
                        first_desc = str(first.get("description") or "").strip()
                        row["source_layer_name"] = first_name or None
                        row["source_layer_description"] = first_desc or None
                        row["source_layer_names"] = layer_names[:30]
                        row["source_layer_assignment_ids"] = layer_assignment_ids[:30]
                        row["source_layer_ref_ids"] = _sort_ref_ids(layer_ref_ids)[:100]
                        row["source_layer_note"] = (
                            "LAYER_FACE_MULTI_MAPPED"
                            if len(layer_names) > 1
                            else ("LAYER_FACE_MAPPED" if (first_name or first_desc) else "LAYER_FACE_MAPPED_NAME_EMPTY")
                        )
            elif mapping_mode:
                row["step_ref_mapping"] = mapping_mode

            face_color = _shape_color(face)
            if face_color is not None:
                row.update(face_color)
                color_hit_face_direct += 1

            ex.Next()

        # Face -> Part(=assembly component name) mapping for user-friendly traceability.
        # Use component instance shapes so IsSame(face) can match the OneShape faces.
        comp_labels = TDF_LabelSequence()
        free_labels = TDF_LabelSequence()
        shape_tool.GetFreeShapes(free_labels)
        if free_labels.Length() == 0:
            shape_tool.GetShapes(comp_labels)
        else:
            for fi in range(1, free_labels.Length() + 1):
                free_label = free_labels.Value(fi)
                comps = TDF_LabelSequence()
                shape_tool.GetComponents_s(free_label, comps)
                if comps.Length() == 0:
                    comp_labels.Append(free_label)
                else:
                    for ci in range(1, comps.Length() + 1):
                        comp_labels.Append(comps.Value(ci))

        solid_entries: list[dict[str, Any]] = []
        for li in range(1, comp_labels.Length() + 1):
            comp_label = comp_labels.Value(li)
            comp_shape = shape_tool.GetShape_s(comp_label)

            ref_label = TDF_Label()
            ref_shape = None
            ref_name = None
            if shape_tool.GetReferredShape_s(comp_label, ref_label) and not ref_label.IsNull():
                ref_shape = shape_tool.GetShape_s(ref_label)
                ref_name = _label_name(ref_label)

            if comp_shape is None or comp_shape.IsNull():
                comp_shape = ref_shape
            if comp_shape is None or comp_shape.IsNull():
                continue

            comp_color = _shape_color(comp_shape)
            if comp_color is None and ref_shape is not None and not ref_shape.IsNull():
                comp_color = _shape_color(ref_shape)

            raw_name = _label_name(comp_label) or ref_name
            is_guid = bool(raw_name and _is_guid_like(raw_name))
            if raw_name:
                display_name = raw_name
            else:
                display_name = f"Component{li}"

            sx_sol = TopExp_Explorer(comp_shape, TopAbs_SOLID)
            has_solid = False
            while sx_sol.More():
                has_solid = True
                solid_shape = sx_sol.Current()
                solid_color = _shape_color(solid_shape) or comp_color
                solid_faces: list[Any] = []
                sx = TopExp_Explorer(solid_shape, TopAbs_FACE)
                while sx.More():
                    solid_faces.append(_to_face(sx.Current()))
                    sx.Next()

                solid_entries.append(
                    {
                        "display_name": display_name,
                        "raw_name": raw_name,
                        "is_guid": is_guid,
                        "color": solid_color,
                        "faces": solid_faces,
                    }
                )
                sx_sol.Next()

            if has_solid:
                continue

            solid_faces = []
            sx = TopExp_Explorer(comp_shape, TopAbs_FACE)
            while sx.More():
                solid_faces.append(_to_face(sx.Current()))
                sx.Next()
            if solid_faces:
                solid_entries.append(
                    {
                        "display_name": display_name,
                        "raw_name": raw_name,
                        "is_guid": is_guid,
                        "color": comp_color,
                        "faces": solid_faces,
                    }
                )

        face_part_candidates: Dict[str, list[dict[str, Any]]] = defaultdict(list)
        for solid in solid_entries:
            for sface in solid["faces"]:
                for face_id, gface in face_order:
                    if gface.IsSame(sface):
                        bucket = face_part_candidates[face_id]
                        key = (solid["display_name"], solid["raw_name"], solid["is_guid"])
                        if all((x["display_name"], x["raw_name"], x["is_guid"]) != key for x in bucket):
                            bucket.append(
                                {
                                    "display_name": solid["display_name"],
                                    "raw_name": solid["raw_name"],
                                    "is_guid": solid["is_guid"],
                                    "color": solid.get("color"),
                                }
                            )
                        break

        part_names_all: list[str] = []
        part_names_human: list[str] = []
        for face_id, _ in face_order:
            candidates = face_part_candidates.get(face_id, [])
            if not candidates:
                continue
            best = sorted(candidates, key=_part_candidate_score)[0]
            part_names = [str(c["display_name"]) for c in candidates]
            result["faces_by_id"].setdefault(face_id, {})
            result["faces_by_id"][face_id]["source_part_name"] = best["display_name"]
            result["faces_by_id"][face_id]["source_part_name_raw"] = best["raw_name"]
            result["faces_by_id"][face_id]["source_part_is_guid"] = bool(best["is_guid"])
            result["faces_by_id"][face_id]["source_part_names"] = part_names[:20]
            result["faces_by_id"][face_id]["source_part_candidate_count"] = len(candidates)
            if not result["faces_by_id"][face_id].get("source_color_hex"):
                color_best = next(
                    (c for c in candidates if isinstance(c.get("color"), dict) and not c["is_guid"]),
                    None,
                )
                if color_best is None:
                    color_best = next((c for c in candidates if isinstance(c.get("color"), dict)), None)
                if color_best is not None:
                    color_data = dict(color_best["color"])
                    color_data["source_color_source"] = f"{color_data.get('source_color_source')}|PartFallback"
                    result["faces_by_id"][face_id].update(color_data)
            part_names_all.extend(part_names)
            part_names_human.extend([str(c["display_name"]) for c in candidates if not c["is_guid"]])

        result["meta"]["color_source"] = "STEPCAF"
        face_total = max(0, idx - 1)
        face_color_hit = sum(1 for _, row in result["faces_by_id"].items() if row.get("source_color_hex"))
        result["meta"]["face_color_total"] = face_total
        result["meta"]["face_color_hit"] = face_color_hit
        result["meta"]["face_color_hit_face_direct"] = color_hit_face_direct
        result["meta"]["face_color_hit_ratio"] = None if face_total == 0 else round(face_color_hit / face_total, 6)
        result["meta"]["part_solid_total"] = len(solid_entries)
        result["meta"]["face_part_mapped_total"] = len(face_part_candidates)
        result["meta"]["face_part_multi_candidate_total"] = sum(1 for _, v in face_part_candidates.items() if len(v) > 1)
        uniq_all = list(dict.fromkeys(part_names_all))
        uniq_human = list(dict.fromkeys(part_names_human))
        result["meta"]["part_name_samples"] = uniq_all[:30]
        result["meta"]["part_name_human_samples"] = uniq_human[:30]
        result["meta"]["step_advanced_face_ref_total"] = len(step_face_refs)
        result["meta"]["step_advanced_face_ref_mapped"] = min(len(step_face_refs), max(0, idx - 1))
        result["meta"]["layer_assignment_total"] = len(layer_entries)
        result["meta"]["layer_assignment_named"] = sum(
            1 for x in layer_entries if str(x.get("name") or "").strip() or str(x.get("description") or "").strip()
        )
        result["meta"]["layer_face_mapped_total"] = sum(
            1 for _, row in result["faces_by_id"].items() if isinstance(row.get("source_layer_assignment_ids"), list)
        )
        result["meta"]["layer_named_samples"] = [
            {
                "assignment_id": x.get("assignment_id"),
                "name": x.get("name"),
                "description": x.get("description"),
                "item_ref_count": x.get("item_ref_count"),
                "face_ref_count": x.get("face_ref_count"),
                "manifold_names": (x.get("manifold_names") or [])[:10],
                "item_ref_types": x.get("item_ref_types") or {},
                "edge_like_ref_count": x.get("edge_like_ref_count"),
                "edge_like_ref_ids": (x.get("edge_like_ref_ids") or [])[:20],
                "item_refs_sample": (x.get("item_refs") or [])[:20],
                "attachment_counts": x.get("attachment_counts") or {},
                "attachment_scope": x.get("attachment_scope") or [],
            }
            for x in layer_entries
            if str(x.get("name") or "").strip() or str(x.get("description") or "").strip()
        ][:30]
        if len(face_part_candidates) == 0:
            result["meta"]["part_note"] = "PART_MAPPING_NONE"
        elif len(uniq_human) == 0:
            result["meta"]["part_note"] = "PART_MAPPING_GUID_ONLY"
        else:
            result["meta"]["part_note"] = "PART_MAPPING_OK"
        return result
    except Exception:
        LOGGER.exception("Failed to extract face visuals with STEPCAF")
        return result


def parse_step_header_text(step_path: Path) -> Dict[str, Any]:
    """Best-effort STEP header parser independent of OCCT internals."""
    text = step_path.read_text(encoding="utf-8", errors="ignore")

    file_schema = None
    file_name = None
    file_description = None

    m_schema = re.search(r"FILE_SCHEMA\s*\(\s*\((.*?)\)\s*\)", text, flags=re.IGNORECASE | re.DOTALL)
    if m_schema:
        file_schema = m_schema.group(1).strip()

    m_name = re.search(r"FILE_NAME\s*\((.*?)\)\s*;", text, flags=re.IGNORECASE | re.DOTALL)
    if m_name:
        file_name = m_name.group(1).strip()

    m_desc = re.search(r"FILE_DESCRIPTION\s*\((.*?)\)\s*;", text, flags=re.IGNORECASE | re.DOTALL)
    if m_desc:
        file_description = m_desc.group(1).strip()

    units = sorted(set(re.findall(r"\b(SI_UNIT\s*\(.*?\))", text, flags=re.IGNORECASE)))
    units_parsed = _parse_unit_system(text)

    return {
        "file_schema": file_schema,
        "file_name": file_name,
        "file_description": file_description,
        "units_raw": units[:20],
        "units_parsed": units_parsed,
        "entity_name_stats": {
            "advanced_face": _named_entity_stats(text, "ADVANCED_FACE"),
            "edge_curve": _named_entity_stats(text, "EDGE_CURVE"),
            "manifold_solid_brep": _named_entity_stats(text, "MANIFOLD_SOLID_BREP"),
            "product": _named_entity_stats(text, "PRODUCT"),
        },
        "presentation_stats": {
            **_parse_colour_palette(text),
            **_parse_layer_stats(text),
            **_parse_styled_item_stats(text),
        },
        "human_named_signals": _extract_human_named_tokens(text),
    }


def load_step_shape(step_path: str) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    path = Path(step_path)
    if not path.exists():
        raise FileNotFoundError(f"STEP file not found: {path}")

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP read failed with status: {int(status)}")

    transfer_result = reader.TransferRoots()
    LOGGER.info("TransferRoots result: %s", transfer_result)

    shape = reader.OneShape()
    if shape is None:
        raise RuntimeError("Reader returned empty shape")

    header = parse_step_header_text(path)
    visuals = _extract_face_visuals_with_xcaf(path)
    pstats = header.get("presentation_stats", {})
    layer_total = int(pstats.get("layer_assignment_total", 0))
    layer_named = int(pstats.get("layer_assignment_named", 0))
    layer_samples = pstats.get("layer_named_samples", [])
    layer_name = None
    layer_desc = None
    layer_note = "LAYER_NONE"
    if layer_total > 0 and layer_named == 0:
        layer_note = "LAYER_NAME_EMPTY"
    elif layer_named > 0:
        one = layer_samples[0] if layer_samples else {}
        layer_name = one.get("name") or None
        layer_desc = one.get("description") or None
        layer_note = "LAYER_NAMED"
    visuals.setdefault("meta", {})
    visuals["meta"]["layer_name"] = layer_name
    visuals["meta"]["layer_description"] = layer_desc
    visuals["meta"]["layer_note"] = layer_note
    units_parsed = header.get("units_parsed", {}) if isinstance(header, dict) else {}
    ul = units_parsed.get("length", {}) if isinstance(units_parsed, dict) else {}
    ua = units_parsed.get("angle", {}) if isinstance(units_parsed, dict) else {}
    visuals["meta"]["length_unit_name"] = ul.get("name")
    visuals["meta"]["length_unit_symbol"] = ul.get("symbol")
    visuals["meta"]["length_unit_to_m"] = ul.get("to_m")
    visuals["meta"]["angle_unit_name"] = ua.get("name")
    visuals["meta"]["angle_unit_symbol"] = ua.get("symbol")
    visuals["meta"]["angle_unit_to_rad"] = ua.get("to_rad")
    visuals["meta"]["part_note"] = visuals["meta"].get("part_note") or "PART_UNKNOWN"
    header.setdefault("presentation_stats", {})
    header["presentation_stats"]["face_color_total_from_xcaf"] = int(visuals.get("meta", {}).get("face_color_total", 0))
    header["presentation_stats"]["face_color_hit_from_xcaf"] = int(visuals.get("meta", {}).get("face_color_hit", 0))
    header["presentation_stats"]["face_color_hit_ratio_from_xcaf"] = visuals.get("meta", {}).get("face_color_hit_ratio")
    header["presentation_stats"]["face_color_source"] = visuals.get("meta", {}).get("color_source")
    header["presentation_stats"]["face_part_mapped_total"] = int(visuals.get("meta", {}).get("face_part_mapped_total", 0))
    header["presentation_stats"]["face_part_multi_candidate_total"] = int(
        visuals.get("meta", {}).get("face_part_multi_candidate_total", 0)
    )
    header["presentation_stats"]["part_solid_total"] = int(visuals.get("meta", {}).get("part_solid_total", 0))
    header["presentation_stats"]["part_note"] = visuals.get("meta", {}).get("part_note")
    header["presentation_stats"]["part_name_samples"] = visuals.get("meta", {}).get("part_name_samples", [])
    header["presentation_stats"]["part_name_human_samples"] = visuals.get("meta", {}).get("part_name_human_samples", [])
    header["presentation_stats"]["face_layer_mapped_total"] = int(visuals.get("meta", {}).get("layer_face_mapped_total", 0))
    if not header["presentation_stats"].get("layer_named_samples"):
        header["presentation_stats"]["layer_named_samples"] = visuals.get("meta", {}).get("layer_named_samples", [])
    header["occt_binding"] = "OCP(cadquery-ocp)" if USING_OCP else "pythonocc-core"
    header["transfer_mode"] = "STEPControl_AsIs"

    return shape, header, visuals
