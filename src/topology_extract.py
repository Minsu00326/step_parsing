from __future__ import annotations

import math
from typing import Any, Dict, List

USING_OCP = False

try:
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.BRepGProp import BRepGProp
    from OCP.BRepLProp import BRepLProp_SLProps
    from OCP.BRepTools import BRepTools
    from OCP.Bnd import Bnd_Box
    from OCP.GProp import GProp_GProps
    from OCP.GeomAbs import (
        GeomAbs_BSplineSurface,
        GeomAbs_BezierSurface,
        GeomAbs_BSplineCurve,
        GeomAbs_BezierCurve,
        GeomAbs_Circle,
        GeomAbs_Cone,
        GeomAbs_Cylinder,
        GeomAbs_Ellipse,
        GeomAbs_Hyperbola,
        GeomAbs_Line,
        GeomAbs_OffsetSurface,
        GeomAbs_OffsetCurve,
        GeomAbs_OtherCurve,
        GeomAbs_OtherSurface,
        GeomAbs_Parabola,
        GeomAbs_Plane,
        GeomAbs_Sphere,
        GeomAbs_SurfaceOfExtrusion,
        GeomAbs_SurfaceOfRevolution,
        GeomAbs_Torus,
    )
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED, TopAbs_SHELL, TopAbs_SOLID, TopAbs_VERTEX, TopAbs_WIRE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    USING_OCP = True
except Exception:
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepBndLib import brepbndlib_Add
    from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
    from OCC.Core.BRepGProp import brepgprop_LinearProperties, brepgprop_SurfaceProperties, brepgprop_VolumeProperties
    from OCC.Core.BRepLProp import BRepLProp_SLProps
    from OCC.Core.BRepTools import breptools_UVBounds
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.GeomAbs import (
        GeomAbs_BSplineSurface,
        GeomAbs_BezierSurface,
        GeomAbs_BSplineCurve,
        GeomAbs_BezierCurve,
        GeomAbs_Circle,
        GeomAbs_Cone,
        GeomAbs_Cylinder,
        GeomAbs_Ellipse,
        GeomAbs_Hyperbola,
        GeomAbs_Line,
        GeomAbs_OffsetSurface,
        GeomAbs_OffsetCurve,
        GeomAbs_OtherCurve,
        GeomAbs_OtherSurface,
        GeomAbs_Parabola,
        GeomAbs_Plane,
        GeomAbs_Sphere,
        GeomAbs_SurfaceOfExtrusion,
        GeomAbs_SurfaceOfRevolution,
        GeomAbs_Torus,
    )
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED, TopAbs_SHELL, TopAbs_SOLID, TopAbs_VERTEX, TopAbs_WIRE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods


THICKNESS_SOURCE_OPPOSITE_PARALLEL = (
    "OCCT:TopExp_Explorer(TopAbs_SOLID/TopAbs_FACE)+BRepAdaptor_Surface.GetType(GeomAbs_Plane)"
    "+Dot(local_z_dir)|opposite_parallel_plane_distance"
)
THICKNESS_SOURCE_PARALLEL = (
    "OCCT:TopExp_Explorer(TopAbs_SOLID/TopAbs_FACE)+BRepAdaptor_Surface.GetType(GeomAbs_Plane)"
    "+Dot(local_z_dir)|parallel_plane_distance"
)
THICKNESS_SOURCE_LOCAL_Z_SPAN = "OCCT:Geom_Surface.Position(Ax3)+edge_polyline_projection(local_z_span)"
THICKNESS_SOURCE_GLOBAL_BBOX = "OCCT:Bnd_Box.Get(global_xyz_span_min)"
FACE_SIZE_BASIS_LOCAL_FRAME = "OCCT:Geom_Surface.Position(Ax3)+edge_polyline_projection(local_xy)"
FACE_SIZE_BASIS_GLOBAL_BBOX = "OCCT:Bnd_Box.Get(global_xyz_span)"


def _to_face(shape: Any) -> Any:
    if USING_OCP:
        return TopoDS.Face_s(shape)
    return topods.Face(shape)


def _to_edge(shape: Any) -> Any:
    if USING_OCP:
        return TopoDS.Edge_s(shape)
    return topods.Edge(shape)


def _to_wire(shape: Any) -> Any:
    if USING_OCP:
        return TopoDS.Wire_s(shape)
    return topods.Wire(shape)


def _bbox_minmax(box: Any) -> Dict[str, List[float]]:
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return {
        "min": [float(xmin), float(ymin), float(zmin)],
        "max": [float(xmax), float(ymax), float(zmax)],
    }


def _uv_bounds(face: Any) -> tuple[float, float, float, float] | None:
    try:
        if USING_OCP:
            umin, umax, vmin, vmax = BRepTools.UVBounds_s(face)
        else:
            umin, umax, vmin, vmax = breptools_UVBounds(face)
        return float(umin), float(umax), float(vmin), float(vmax)
    except Exception:
        return None


def _face_curvature_mid(face: Any) -> Dict[str, Any]:
    uv = _uv_bounds(face)
    if uv is None:
        return {"curvature_status": "UV_BOUNDS_UNAVAILABLE"}
    umin, umax, vmin, vmax = uv
    if not all(math.isfinite(x) for x in [umin, umax, vmin, vmax]):
        return {"curvature_status": "UV_BOUNDS_NONFINITE"}
    u = 0.5 * (umin + umax)
    v = 0.5 * (vmin + vmax)
    try:
        props = BRepLProp_SLProps(BRepAdaptor_Surface(face, True), u, v, 2, 1e-6)
        if not props.IsCurvatureDefined():
            return {
                "curvature_status": "CURVATURE_UNDEFINED",
                "curvature_eval_uv": [float(u), float(v)],
            }
        return {
            "curvature_status": "OK",
            "curvature_eval_uv": [float(u), float(v)],
            "mean_curvature": float(props.MeanCurvature()),
            "gaussian_curvature": float(props.GaussianCurvature()),
            "min_curvature": float(props.MinCurvature()),
            "max_curvature": float(props.MaxCurvature()),
        }
    except Exception:
        return {
            "curvature_status": "CURVATURE_ERROR",
            "curvature_eval_uv": [float(u), float(v)],
        }


def _pnt_xyz(p: Any) -> list[float]:
    return [float(p.X()), float(p.Y()), float(p.Z())]


def _dir_xyz(d: Any) -> list[float]:
    return [float(d.X()), float(d.Y()), float(d.Z())]


def _frame_from_ax3(ax3: Any) -> Dict[str, Any]:
    return {
        "local_origin": _pnt_xyz(ax3.Location()),
        "local_x_dir": _dir_xyz(ax3.XDirection()),
        "local_y_dir": _dir_xyz(ax3.YDirection()),
        "local_z_dir": _dir_xyz(ax3.Direction()),
        "local_frame_status": "OK",
    }


def _angle_between_deg(a: list[float], b: list[float]) -> float | None:
    if len(a) != 3 or len(b) != 3:
        return None
    ax, ay, az = float(a[0]), float(a[1]), float(a[2])
    bx, by, bz = float(b[0]), float(b[1]), float(b[2])
    na = math.sqrt(ax * ax + ay * ay + az * az)
    nb = math.sqrt(bx * bx + by * by + bz * bz)
    if na <= 1e-15 or nb <= 1e-15:
        return None
    c = (ax * bx + ay * by + az * bz) / (na * nb)
    c = max(-1.0, min(1.0, c))
    return float(math.degrees(math.acos(c)))


def _surface_local_frame(adaptor: Any, stype: int) -> Dict[str, Any]:
    try:
        if stype == GeomAbs_Plane:
            return _frame_from_ax3(adaptor.Plane().Position())
        if stype == GeomAbs_Cylinder:
            return _frame_from_ax3(adaptor.Cylinder().Position())
        if stype == GeomAbs_Cone:
            return _frame_from_ax3(adaptor.Cone().Position())
        if stype == GeomAbs_Sphere:
            return _frame_from_ax3(adaptor.Sphere().Position())
        if stype == GeomAbs_Torus:
            return _frame_from_ax3(adaptor.Torus().Position())
    except Exception:
        return {"local_frame_status": "ERROR"}
    return {"local_frame_status": "UNAVAILABLE"}


def _bbox_overlap_with_tol(
    a_min: list[float], a_max: list[float], b_min: list[float], b_max: list[float], tol: float
) -> bool:
    return not (
        a_max[0] < (b_min[0] - tol)
        or b_max[0] < (a_min[0] - tol)
        or a_max[1] < (b_min[1] - tol)
        or b_max[1] < (a_min[1] - tol)
        or a_max[2] < (b_min[2] - tol)
        or b_max[2] < (a_min[2] - tol)
    )


def _shape_surface_area(shape: Any) -> float:
    props = GProp_GProps()
    try:
        if USING_OCP:
            BRepGProp.SurfaceProperties_s(shape, props)
        else:
            brepgprop_SurfaceProperties(shape, props)
        return float(props.Mass())
    except Exception:
        return 0.0


def _shape_linear_length(shape: Any) -> float:
    props = GProp_GProps()
    try:
        if USING_OCP:
            BRepGProp.LinearProperties_s(shape, props)
        else:
            brepgprop_LinearProperties(shape, props)
        return float(props.Mass())
    except Exception:
        return 0.0


def _edge_length(edge: Any) -> float:
    return _shape_linear_length(edge)


def _edge_polyline(edge: Any, curve: Any | None = None, max_points: int = 48) -> list[list[float]]:
    try:
        c = curve if curve is not None else BRepAdaptor_Curve(edge)
        u0 = float(c.FirstParameter())
        u1 = float(c.LastParameter())
        if not (math.isfinite(u0) and math.isfinite(u1)):
            return []
        if abs(u1 - u0) <= 1e-12:
            p = c.Value(u0)
            return [[float(p.X()), float(p.Y()), float(p.Z())]]
        # Keep size bounded; enough points to make the edge shape recognizable.
        n = 24
        pts: list[list[float]] = []
        for i in range(n):
            t = i / float(n - 1)
            u = u0 + (u1 - u0) * t
            p = c.Value(u)
            pts.append([float(p.X()), float(p.Y()), float(p.Z())])
        return pts
    except Exception:
        return []


def _polyline_bounds(points: list[list[float]]) -> tuple[list[float], list[float]] | None:
    if not points:
        return None
    xs = [float(p[0]) for p in points if isinstance(p, list) and len(p) >= 3]
    ys = [float(p[1]) for p in points if isinstance(p, list) and len(p) >= 3]
    zs = [float(p[2]) for p in points if isinstance(p, list) and len(p) >= 3]
    if not xs or not ys or not zs:
        return None
    return [min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]


def _vec_dot(a: list[float], b: list[float]) -> float:
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def _vec_sub(a: list[float], b: list[float]) -> list[float]:
    return [float(a[0] - b[0]), float(a[1] - b[1]), float(a[2] - b[2])]


def _vec_norm(a: list[float]) -> float:
    return float(math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]))


def _vec_unit(a: list[float]) -> list[float] | None:
    n = _vec_norm(a)
    if n <= 1e-15:
        return None
    return [float(a[0] / n), float(a[1] / n), float(a[2] / n)]


def _axis_span_from_points(points: list[list[float]], axis_dir: list[float] | None) -> float | None:
    u = _vec_unit(axis_dir) if isinstance(axis_dir, list) and len(axis_dir) == 3 else None
    if u is None or len(points) == 0:
        return None
    vals: list[float] = []
    for p in points:
        if not isinstance(p, list) or len(p) < 3:
            continue
        vals.append(_vec_dot([float(p[0]), float(p[1]), float(p[2])], u))
    if len(vals) == 0:
        return None
    span = float(max(vals) - min(vals))
    return span if span > 1e-9 else None


def _average_axis_dir(dirs: list[list[float]]) -> list[float] | None:
    unit_dirs: list[list[float]] = []
    for d in dirs:
        u = _vec_unit(d) if isinstance(d, list) and len(d) == 3 else None
        if u is not None:
            unit_dirs.append(u)
    if len(unit_dirs) == 0:
        return None
    ref = unit_dirs[0]
    sx = sy = sz = 0.0
    for u in unit_dirs:
        sgn = -1.0 if _vec_dot(u, ref) < 0.0 else 1.0
        sx += sgn * u[0]
        sy += sgn * u[1]
        sz += sgn * u[2]
    return _vec_unit([sx, sy, sz])


def _estimate_cylindrical_metrics(
    *,
    surface_type: str,
    area: float,
    points: list[list[float]],
    edge_items: list[Dict[str, Any]],
    surface_radius: float | None,
    surface_axis_dir: list[float] | None,
    surface_axis_origin: list[float] | None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "cylindrical_like": False,
        "cylindrical_mode": None,
        "cyl_radius_est": None,
        "cyl_diameter_est": None,
        "cyl_height_est": None,
        "cyl_sweep_angle_rad_est": None,
        "cyl_sweep_angle_deg_est": None,
        "cyl_arc_length_est": None,
        "cyl_circumference_est": None,
        "cyl_axis_origin_est": None,
        "cyl_axis_direction_est": None,
        "cyl_circle_edge_count": 0,
        "cyl_metric_source": None,
    }

    circle_radii: list[float] = []
    circle_axis_dirs: list[list[float]] = []
    circle_centers: list[list[float]] = []
    for e in edge_items:
        if str(e.get("curve_type") or "") != "Circle":
            continue
        out["cyl_circle_edge_count"] = int(out["cyl_circle_edge_count"]) + 1
        rr = e.get("radius")
        if rr is not None and math.isfinite(float(rr)) and float(rr) > 0.0:
            circle_radii.append(float(rr))
        cdir = e.get("circle_axis_direction")
        if isinstance(cdir, list) and len(cdir) == 3:
            circle_axis_dirs.append([float(cdir[0]), float(cdir[1]), float(cdir[2])])
        cc = e.get("circle_center")
        if isinstance(cc, list) and len(cc) == 3:
            circle_centers.append([float(cc[0]), float(cc[1]), float(cc[2])])

    radius_est = None
    if surface_radius is not None and math.isfinite(float(surface_radius)) and float(surface_radius) > 0.0:
        radius_est = float(surface_radius)
    elif len(circle_radii) > 0:
        radius_est = float(sum(circle_radii) / len(circle_radii))

    axis_dir_est = _vec_unit(surface_axis_dir) if isinstance(surface_axis_dir, list) and len(surface_axis_dir) == 3 else None
    if axis_dir_est is None:
        axis_dir_est = _average_axis_dir(circle_axis_dirs)

    axis_origin_est = None
    if isinstance(surface_axis_origin, list) and len(surface_axis_origin) == 3:
        axis_origin_est = [float(surface_axis_origin[0]), float(surface_axis_origin[1]), float(surface_axis_origin[2])]
    elif len(circle_centers) > 0:
        cx = sum(float(c[0]) for c in circle_centers) / len(circle_centers)
        cy = sum(float(c[1]) for c in circle_centers) / len(circle_centers)
        cz = sum(float(c[2]) for c in circle_centers) / len(circle_centers)
        axis_origin_est = [float(cx), float(cy), float(cz)]

    is_exact_cylinder = str(surface_type or "") == "Cylinder"
    is_bspline_cylinder_hint = (
        str(surface_type or "") == "BSplineSurface"
        and int(out["cyl_circle_edge_count"]) >= 2
        and radius_est is not None
        and axis_dir_est is not None
    )
    if not (is_exact_cylinder or is_bspline_cylinder_hint):
        return out

    height_est = _axis_span_from_points(points, axis_dir_est)
    sweep_rad = None
    if radius_est is not None and radius_est > 1e-12 and height_est is not None and height_est > 1e-12:
        sweep = float(area) / (float(radius_est) * float(height_est))
        if math.isfinite(sweep) and sweep > 1e-9:
            if sweep > (2.0 * math.pi * 1.2):
                sweep_rad = None
            else:
                sweep_rad = max(0.0, min(float(sweep), 2.0 * math.pi))

    out["cylindrical_like"] = True
    out["cylindrical_mode"] = "EXACT_CYLINDER" if is_exact_cylinder else "INFERRED_FROM_CIRCLE_EDGES"
    out["cyl_radius_est"] = float(radius_est) if radius_est is not None else None
    out["cyl_diameter_est"] = float(2.0 * radius_est) if radius_est is not None else None
    out["cyl_height_est"] = float(height_est) if height_est is not None else None
    out["cyl_sweep_angle_rad_est"] = sweep_rad
    out["cyl_sweep_angle_deg_est"] = float(math.degrees(sweep_rad)) if sweep_rad is not None else None
    out["cyl_arc_length_est"] = float(radius_est * sweep_rad) if (radius_est is not None and sweep_rad is not None) else None
    out["cyl_circumference_est"] = float(2.0 * math.pi * radius_est) if radius_est is not None else None
    out["cyl_axis_origin_est"] = axis_origin_est
    out["cyl_axis_direction_est"] = axis_dir_est
    out["cyl_metric_source"] = (
        "OCCT:Geom_CylindricalSurface+edge_circle_axis_projection"
        if is_exact_cylinder
        else "OCCT:edge_circle_axis_projection(BSpline_cyl_hint)"
    )
    return out


def _shape_center_of_mass(shape: Any) -> list[float] | None:
    props = GProp_GProps()
    try:
        if USING_OCP:
            BRepGProp.SurfaceProperties_s(shape, props)
        else:
            brepgprop_SurfaceProperties(shape, props)
        p = props.CentreOfMass()
        return [float(p.X()), float(p.Y()), float(p.Z())]
    except Exception:
        return None


def _face_points_from_edges(edge_items: list[Dict[str, Any]]) -> list[list[float]]:
    out: list[list[float]] = []
    seen: set[tuple[int, int, int]] = set()
    for e in edge_items:
        poly = e.get("polyline")
        if not isinstance(poly, list):
            continue
        for p in poly:
            if not isinstance(p, list) or len(p) < 3:
                continue
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            # Quantize so duplicated seam points are merged.
            key = (round(x * 1e6), round(y * 1e6), round(z * 1e6))
            if key in seen:
                continue
            seen.add(key)
            out.append([x, y, z])
    return out


def _face_size_metrics(
    points: list[list[float]],
    local_origin: list[float] | None,
    local_x_dir: list[float] | None,
    local_y_dir: list[float] | None,
    local_z_dir: list[float] | None,
    bbox_min: list[float],
    bbox_max: list[float],
) -> Dict[str, Any]:
    dx = float(bbox_max[0] - bbox_min[0])
    dy = float(bbox_max[1] - bbox_min[1])
    dz = float(bbox_max[2] - bbox_min[2])
    spans_global = [max(0.0, dx), max(0.0, dy), max(0.0, dz)]
    sorted_global = sorted(spans_global, reverse=True)
    length = sorted_global[0] if len(sorted_global) > 0 else None
    width = sorted_global[1] if len(sorted_global) > 1 else None
    thickness = sorted_global[2] if len(sorted_global) > 2 else None
    out: Dict[str, Any] = {
        "face_span_global_xyz": spans_global,
        "face_span_local_xyz": None,
        "face_length": length,
        "face_width": width,
        "face_thickness_est": thickness,
        "face_thickness_source": THICKNESS_SOURCE_GLOBAL_BBOX,
        "face_size_basis": FACE_SIZE_BASIS_GLOBAL_BBOX,
    }

    if not points:
        return out
    if not (isinstance(local_origin, list) and len(local_origin) == 3):
        return out

    ux = _vec_unit(local_x_dir) if isinstance(local_x_dir, list) and len(local_x_dir) == 3 else None
    uy = _vec_unit(local_y_dir) if isinstance(local_y_dir, list) and len(local_y_dir) == 3 else None
    uz = _vec_unit(local_z_dir) if isinstance(local_z_dir, list) and len(local_z_dir) == 3 else None
    if ux is None or uy is None or uz is None:
        return out

    px: list[float] = []
    py: list[float] = []
    pz: list[float] = []
    for p in points:
        d = _vec_sub(p, local_origin)
        px.append(_vec_dot(d, ux))
        py.append(_vec_dot(d, uy))
        pz.append(_vec_dot(d, uz))
    if len(px) == 0 or len(py) == 0 or len(pz) == 0:
        return out

    sx = float(max(px) - min(px))
    sy = float(max(py) - min(py))
    sz = float(max(pz) - min(pz))
    spans_local = [max(0.0, sx), max(0.0, sy), max(0.0, sz)]
    out["face_span_local_xyz"] = spans_local
    out["face_length"] = max(sx, sy)
    out["face_width"] = min(sx, sy)
    out["face_thickness_est"] = sz
    out["face_thickness_source"] = THICKNESS_SOURCE_LOCAL_Z_SPAN
    out["face_size_basis"] = FACE_SIZE_BASIS_LOCAL_FRAME
    return out


def _collect_solids_with_faces(shape: Any) -> list[Dict[str, Any]]:
    solids: list[Dict[str, Any]] = []
    sx_sol = TopExp_Explorer(shape, TopAbs_SOLID)
    idx = 1
    while sx_sol.More():
        solid = sx_sol.Current()
        solid_id = f"Solid{idx}"
        idx += 1
        faces: list[Dict[str, Any]] = []
        sx_face = TopExp_Explorer(solid, TopAbs_FACE)
        while sx_face.More():
            f = _to_face(sx_face.Current())
            stype = None
            normal = None
            try:
                ad = BRepAdaptor_Surface(f, True)
                stype = ad.GetType()
                lf = _surface_local_frame(ad, stype)
                nz = lf.get("local_z_dir")
                if isinstance(nz, list) and len(nz) == 3:
                    normal = [float(nz[0]), float(nz[1]), float(nz[2])]
            except Exception:
                pass
            faces.append(
                {
                    "face": f,
                    "surface_type_raw": stype,
                    "normal": normal,
                    "center": _shape_center_of_mass(f),
                }
            )
            sx_face.Next()
        solids.append({"solid_id": solid_id, "faces": faces})
        sx_sol.Next()
    return solids


def _find_solid_ids_for_face(face: Any, solids: list[Dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for item in solids:
        sid = str(item.get("solid_id") or "").strip()
        for f in item.get("faces", []):
            try:
                if face.IsSame(f.get("face")):
                    out.append(sid)
                    break
            except Exception:
                continue
    return out


def _estimate_planar_face_thickness_in_solid(
    face: Any,
    face_normal: list[float] | None,
    face_center: list[float] | None,
    solid_id: str | None,
    solids: list[Dict[str, Any]],
) -> tuple[float | None, str | None]:
    if not solid_id:
        return None, None
    if not (isinstance(face_normal, list) and len(face_normal) == 3 and isinstance(face_center, list) and len(face_center) == 3):
        return None, None
    n = _vec_unit([float(face_normal[0]), float(face_normal[1]), float(face_normal[2])])
    if n is None:
        return None, None

    solid = next((x for x in solids if x.get("solid_id") == solid_id), None)
    if not solid:
        return None, None

    best_opposite = None
    best_parallel = None
    for other in solid.get("faces", []):
        oface = other.get("face")
        try:
            if face.IsSame(oface):
                continue
        except Exception:
            continue

        on = other.get("normal")
        oc = other.get("center")
        if not (isinstance(on, list) and len(on) == 3 and isinstance(oc, list) and len(oc) == 3):
            continue
        ounit = _vec_unit([float(on[0]), float(on[1]), float(on[2])])
        if ounit is None:
            continue
        cosang = _vec_dot(n, ounit)
        # Near parallel only.
        if abs(cosang) < 0.95:
            continue
        d = abs(_vec_dot(_vec_sub([float(oc[0]), float(oc[1]), float(oc[2])], face_center), n))
        if not math.isfinite(d) or d <= 1e-9:
            continue

        if cosang < -0.95:
            best_opposite = d if best_opposite is None else min(best_opposite, d)
        else:
            best_parallel = d if best_parallel is None else min(best_parallel, d)

    if best_opposite is not None:
        return float(best_opposite), THICKNESS_SOURCE_OPPOSITE_PARALLEL
    if best_parallel is not None:
        return float(best_parallel), THICKNESS_SOURCE_PARALLEL
    return None, None


def _shape_distance(a: Any, b: Any) -> float | None:
    try:
        dss = BRepExtrema_DistShapeShape(a, b)
        dss.Perform()
        if not dss.IsDone():
            return None
        return float(dss.Value())
    except Exception:
        return None


def _common_area_length(a: Any, b: Any) -> tuple[float, float]:
    try:
        common = BRepAlgoAPI_Common(a, b)
        common.Build()
        if not common.IsDone():
            return 0.0, 0.0
        shape = common.Shape()
        return _shape_surface_area(shape), _shape_linear_length(shape)
    except Exception:
        return 0.0, 0.0


def _surface_type_name(stype: int) -> str:
    mapping = {
        GeomAbs_Plane: "Plane",
        GeomAbs_Cylinder: "Cylinder",
        GeomAbs_Cone: "Cone",
        GeomAbs_Sphere: "Sphere",
        GeomAbs_Torus: "Torus",
        GeomAbs_BSplineSurface: "BSplineSurface",
        GeomAbs_BezierSurface: "BezierSurface",
        GeomAbs_SurfaceOfRevolution: "SurfaceOfRevolution",
        GeomAbs_SurfaceOfExtrusion: "SurfaceOfExtrusion",
        GeomAbs_OffsetSurface: "OffsetSurface",
        GeomAbs_OtherSurface: "OtherSurface",
    }
    return mapping.get(stype, f"Unknown({stype})")


def _curve_type_name(ctype: int) -> str:
    mapping = {
        GeomAbs_Line: "Line",
        GeomAbs_Circle: "Circle",
        GeomAbs_Ellipse: "Ellipse",
        GeomAbs_Hyperbola: "Hyperbola",
        GeomAbs_Parabola: "Parabola",
        GeomAbs_BezierCurve: "BezierCurve",
        GeomAbs_BSplineCurve: "BSplineCurve",
        GeomAbs_OffsetCurve: "OffsetCurve",
        GeomAbs_OtherCurve: "OtherCurve",
    }
    return mapping.get(ctype, f"Unknown({ctype})")


def _surface_step_entity(surface_type: str) -> str | None:
    mapping = {
        "Plane": "PLANE",
        "Cylinder": "CYLINDRICAL_SURFACE",
        "Cone": "CONICAL_SURFACE",
        "Sphere": "SPHERICAL_SURFACE",
        "Torus": "TOROIDAL_SURFACE",
        "BSplineSurface": "B_SPLINE_SURFACE",
        "BezierSurface": "BEZIER_SURFACE",
        "SurfaceOfRevolution": "SURFACE_OF_REVOLUTION",
        "SurfaceOfExtrusion": "SURFACE_OF_LINEAR_EXTRUSION",
        "OffsetSurface": "OFFSET_SURFACE",
    }
    return mapping.get(surface_type)


def _surface_step_entity_consistency(expected: str | None, mapped_raw: str | None) -> bool | None:
    exp = str(expected or "").strip().upper()
    raw = str(mapped_raw or "").strip().upper()
    if not exp or not raw:
        return None
    if exp == raw:
        return True

    # STEP complex entities can expose umbrella types even when OCCT resolves
    # to a concrete parametric surface.
    alias_map = {
        "B_SPLINE_SURFACE": {"BOUNDED_SURFACE", "B_SPLINE_SURFACE_WITH_KNOTS", "RATIONAL_B_SPLINE_SURFACE"},
        "BEZIER_SURFACE": {"BOUNDED_SURFACE"},
        "SURFACE_OF_REVOLUTION": {"SWEPT_SURFACE"},
        "SURFACE_OF_LINEAR_EXTRUSION": {"SWEPT_SURFACE"},
    }
    if raw in alias_map.get(exp, set()):
        return True
    return False


def _curve_step_entity(curve_type: str) -> str | None:
    mapping = {
        "Line": "LINE",
        "Circle": "CIRCLE",
        "Ellipse": "ELLIPSE",
        "Hyperbola": "HYPERBOLA",
        "Parabola": "PARABOLA",
        "BezierCurve": "BEZIER_CURVE",
        "BSplineCurve": "B_SPLINE_CURVE",
        "OffsetCurve": "OFFSET_CURVE",
        "MixedLoop": "MIXED_LOOP",
    }
    return mapping.get(curve_type)


def _surface_type_name_ko(surface_type: str) -> str:
    mapping = {
        "Plane": "평면",
        "Cylinder": "원통면",
        "Cone": "원뿔면",
        "Sphere": "구면",
        "Torus": "토러스면(도넛형)",
        "BSplineSurface": "자유곡면(B-Spline)",
        "BezierSurface": "자유곡면(Bezier)",
        "SurfaceOfRevolution": "회전 생성면",
        "SurfaceOfExtrusion": "돌출 생성면",
        "OffsetSurface": "오프셋면",
        "OtherSurface": "기타 곡면",
    }
    return mapping.get(surface_type, f"{surface_type}(미분류)")


def _surface_type_desc_ko(surface_type: str) -> str:
    mapping = {
        "Plane": "평면",
        "Cylinder": "원통면",
        "Cone": "원뿔면",
        "Sphere": "구면",
        "Torus": "토러스면",
        "BSplineSurface": "자유곡면(B-Spline)",
        "BezierSurface": "자유곡면(Bezier)",
        "SurfaceOfRevolution": "회전 생성면",
        "SurfaceOfExtrusion": "돌출 생성면",
        "OffsetSurface": "오프셋면",
        "OtherSurface": "기타 곡면",
    }
    return mapping.get(surface_type, "미분류")


def _curve_type_name_ko(curve_type: str) -> str:
    mapping = {
        "Line": "직선",
        "Circle": "원호/원",
        "Ellipse": "타원",
        "Hyperbola": "쌍곡선",
        "Parabola": "포물선",
        "BezierCurve": "Bezier 곡선",
        "BSplineCurve": "B-Spline 곡선",
        "OffsetCurve": "오프셋 곡선",
        "OtherCurve": "기타 곡선",
        "UnknownCurve": "미분류 곡선",
        "MixedLoop": "혼합 루프",
    }
    return mapping.get(curve_type, curve_type)


def _edge_mix_summary_ko(edge_type_counts: Dict[str, int]) -> str:
    if not edge_type_counts:
        return "경계선 정보 없음"
    parts = [f"{_curve_type_name_ko(k)} {v}" for k, v in sorted(edge_type_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    return ", ".join(parts)


def _edge_loop_type_counts(face: Any) -> Dict[str, Any]:
    wire_count = 0
    loop_counts: Dict[str, int] = {}
    mixed_loop_count = 0
    wire_edge_counts: list[int] = []

    wx = TopExp_Explorer(face, TopAbs_WIRE)
    while wx.More():
        wire = _to_wire(wx.Current())
        wire_count += 1

        ex = TopExp_Explorer(wire, TopAbs_EDGE)
        types: list[str] = []
        edge_n = 0
        while ex.More():
            edge = _to_edge(ex.Current())
            edge_n += 1
            ctype = "UnknownCurve"
            try:
                ctype = _curve_type_name(BRepAdaptor_Curve(edge).GetType())
            except Exception:
                pass
            types.append(ctype)
            ex.Next()
        wire_edge_counts.append(edge_n)

        uniq = sorted(set(types))
        if len(uniq) == 1 and uniq[0] != "UnknownCurve":
            t = uniq[0]
            loop_counts[t] = loop_counts.get(t, 0) + 1
        else:
            mixed_loop_count += 1

        wx.Next()

    if mixed_loop_count > 0:
        loop_counts["MixedLoop"] = mixed_loop_count

    return {
        "wire_count": wire_count,
        "edge_loop_type_counts": loop_counts,
        "circle_loop_count": loop_counts.get("Circle", 0),
        "wire_edge_counts": wire_edge_counts,
    }


def _build_easy_hint_ko(surface_type: str, dominant_edge_type: str | None, special: Dict[str, Any]) -> str:
    dom = _curve_type_name_ko(dominant_edge_type) if dominant_edge_type else "미상"
    if surface_type == "Plane":
        return f"주경계: {dom}"
    if surface_type == "Cylinder":
        radius = special.get("radius")
        if radius is not None:
            return f"R={float(radius):.3f} | 주경계: {dom}"
        return f"주경계: {dom}"
    if surface_type == "Cone":
        rr = special.get("ref_radius")
        if rr is not None:
            return f"Rref={float(rr):.3f} | 주경계: {dom}"
        return f"주경계: {dom}"
    if surface_type == "Sphere":
        radius = special.get("radius")
        if radius is not None:
            return f"R={float(radius):.3f} | 주경계: {dom}"
        return f"주경계: {dom}"
    if surface_type == "Torus":
        rmaj = special.get("major_radius")
        rmin = special.get("minor_radius")
        if rmaj is not None and rmin is not None:
            return f"Rmaj={float(rmaj):.3f}, Rmin={float(rmin):.3f} | 주경계: {dom}"
        return f"주경계: {dom}"
    return f"주경계: {dom}"


def _build_face_label(face_id: str, surface_type: str, area: float, edge_count: int, special: Dict[str, Any]) -> str:
    label_bits = [face_id, surface_type]
    if surface_type in {"Cylinder", "Sphere"} and special.get("radius") is not None:
        label_bits.append(f"R={float(special['radius']):.3f}")
    if surface_type == "Cone" and special.get("ref_radius") is not None:
        label_bits.append(f"Rref={float(special['ref_radius']):.3f}")
    if surface_type == "Torus":
        if special.get("major_radius") is not None:
            label_bits.append(f"Rmaj={float(special['major_radius']):.3f}")
        if special.get("minor_radius") is not None:
            label_bits.append(f"Rmin={float(special['minor_radius']):.3f}")
    label_bits.append(f"A={float(area):.3f}")
    label_bits.append(f"E={edge_count}")
    return " | ".join(label_bits)


def count_topology(shape: Any) -> Dict[str, int]:
    def count(kind: int) -> int:
        ex = TopExp_Explorer(shape, kind)
        n = 0
        while ex.More():
            n += 1
            ex.Next()
        return n

    return {
        "solids": count(TopAbs_SOLID),
        "shells": count(TopAbs_SHELL),
        "faces": count(TopAbs_FACE),
        "edges": count(TopAbs_EDGE),
        "vertices": count(TopAbs_VERTEX),
    }


def global_bbox(shape: Any) -> Dict[str, List[float]]:
    box = Bnd_Box()
    if USING_OCP:
        BRepBndLib.Add_s(shape, box)
    else:
        brepbndlib_Add(shape, box)
    return _bbox_minmax(box)


def extract_faces(shape: Any, include_normal: bool = True, visual_context: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    faces: List[Dict[str, Any]] = []
    visual_context = visual_context or {}
    visual_faces: Dict[str, Dict[str, Any]] = visual_context.get("faces_by_id", {}) if isinstance(visual_context, dict) else {}
    visual_meta: Dict[str, Any] = visual_context.get("meta", {}) if isinstance(visual_context, dict) else {}
    solids_with_faces = _collect_solids_with_faces(shape)

    ex = TopExp_Explorer(shape, TopAbs_FACE)
    idx = 1
    while ex.More():
        occ_face = _to_face(ex.Current())
        face_id = f"Face{idx}"

        adaptor = BRepAdaptor_Surface(occ_face, True)
        stype = adaptor.GetType()

        props = GProp_GProps()
        if USING_OCP:
            BRepGProp.SurfaceProperties_s(occ_face, props)
        else:
            brepgprop_SurfaceProperties(occ_face, props)

        com = props.CentreOfMass()
        area = float(props.Mass())

        box = Bnd_Box()
        if USING_OCP:
            BRepBndLib.Add_s(occ_face, box)
        else:
            brepbndlib_Add(occ_face, box)
        uvb = _uv_bounds(occ_face)
        local_frame = _surface_local_frame(adaptor, stype)

        edge_ex = TopExp_Explorer(occ_face, TopAbs_EDGE)
        edge_count = 0
        edge_type_counts: Dict[str, int] = {}
        edge_items: list[Dict[str, Any]] = []
        while edge_ex.More():
            occ_edge = _to_edge(edge_ex.Current())
            edge_count += 1
            curve_type_name = "UnknownCurve"
            curve = None
            try:
                curve = BRepAdaptor_Curve(occ_edge)
                curve_type_name = _curve_type_name(curve.GetType())
            except Exception:
                pass
            edge_type_counts[curve_type_name] = edge_type_counts.get(curve_type_name, 0) + 1

            radius = None
            circle_center = None
            circle_axis_direction = None
            if curve_type_name == "Circle" and curve is not None:
                try:
                    circ = curve.Circle()
                    radius = float(circ.Radius())
                    cax = circ.Axis()
                    cloc = cax.Location()
                    cdir = cax.Direction()
                    circle_center = [float(cloc.X()), float(cloc.Y()), float(cloc.Z())]
                    circle_axis_direction = [float(cdir.X()), float(cdir.Y()), float(cdir.Z())]
                except Exception:
                    radius = None
                    circle_center = None
                    circle_axis_direction = None
            polyline = _edge_polyline(occ_edge, curve=curve)
            start_point = polyline[0] if len(polyline) >= 1 else None
            end_point = polyline[-1] if len(polyline) >= 2 else (polyline[0] if len(polyline) == 1 else None)
            mid_point = polyline[len(polyline) // 2] if len(polyline) >= 1 else None
            pl_bounds = _polyline_bounds(polyline)
            edge_items.append(
                {
                    "edge_id": f"{face_id}_E{edge_count}",
                    "index": edge_count,
                    "curve_type": curve_type_name,
                    "curve_type_ko": _curve_type_name_ko(curve_type_name),
                    "curve_step_entity": _curve_step_entity(curve_type_name),
                    "length": float(_edge_length(occ_edge)),
                    "radius": radius,
                    "circle_center": circle_center,
                    "circle_axis_direction": circle_axis_direction,
                    "polyline": polyline,
                    "polyline_point_count": len(polyline),
                    "start_point": start_point,
                    "end_point": end_point,
                    "mid_point": mid_point,
                    "bbox_min": pl_bounds[0] if pl_bounds else None,
                    "bbox_max": pl_bounds[1] if pl_bounds else None,
                }
            )
            edge_ex.Next()
        loop_info = _edge_loop_type_counts(occ_face)
        dominant_edge_type = None
        if edge_type_counts:
            dominant_edge_type = sorted(edge_type_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

        special: Dict[str, Any] = {}
        if stype == GeomAbs_Cylinder:
            cyl = adaptor.Cylinder()
            axis = cyl.Axis()
            special = {
                "radius": float(cyl.Radius()),
                "axis_origin": [float(axis.Location().X()), float(axis.Location().Y()), float(axis.Location().Z())],
                "axis_direction": [
                    float(axis.Direction().X()),
                    float(axis.Direction().Y()),
                    float(axis.Direction().Z()),
                ],
            }
        elif stype == GeomAbs_Cone:
            cone = adaptor.Cone()
            axis = cone.Axis()
            special = {
                "ref_radius": float(cone.RefRadius()),
                "semi_angle": float(cone.SemiAngle()),
                "semi_angle_deg": float(math.degrees(cone.SemiAngle())),
                "axis_origin": [float(axis.Location().X()), float(axis.Location().Y()), float(axis.Location().Z())],
                "axis_direction": [
                    float(axis.Direction().X()),
                    float(axis.Direction().Y()),
                    float(axis.Direction().Z()),
                ],
            }
        elif stype == GeomAbs_Sphere:
            sph = adaptor.Sphere()
            pos = sph.Position()
            special = {
                "radius": float(sph.Radius()),
                "center": [float(pos.Location().X()), float(pos.Location().Y()), float(pos.Location().Z())],
            }
        elif stype == GeomAbs_Torus:
            tor = adaptor.Torus()
            axis = tor.Axis()
            special = {
                "major_radius": float(tor.MajorRadius()),
                "minor_radius": float(tor.MinorRadius()),
                "axis_origin": [float(axis.Location().X()), float(axis.Location().Y()), float(axis.Location().Z())],
                "axis_direction": [
                    float(axis.Direction().X()),
                    float(axis.Direction().Y()),
                    float(axis.Direction().Z()),
                ],
            }

        face_info: Dict[str, Any] = {
            "face_id": face_id,
            "surface_type": _surface_type_name(stype),
            "area": area,
            "center_of_mass": [float(com.X()), float(com.Y()), float(com.Z())],
            "bbox_min": _bbox_minmax(box)["min"],
            "bbox_max": _bbox_minmax(box)["max"],
            "uv_bounds": None if uvb is None else [float(uvb[0]), float(uvb[1]), float(uvb[2]), float(uvb[3])],
            "edge_count": edge_count,
            "edge_type_counts": edge_type_counts,
            "edges": edge_items,
            "edges_brief": [
                {
                    "edge_id": e.get("edge_id"),
                    "index": e.get("index"),
                    "curve_type": e.get("curve_type"),
                    "curve_type_ko": e.get("curve_type_ko"),
                    "curve_step_entity": e.get("curve_step_entity"),
                    "length": e.get("length"),
                    "radius": e.get("radius"),
                    "polyline_point_count": e.get("polyline_point_count"),
                    "start_point": e.get("start_point"),
                    "end_point": e.get("end_point"),
                }
                for e in edge_items
            ],
            "wire_count": int(loop_info.get("wire_count", 0)),
            "edge_loop_type_counts": loop_info.get("edge_loop_type_counts", {}),
            "circle_loop_count": int(loop_info.get("circle_loop_count", 0)),
            "wire_edge_counts": loop_info.get("wire_edge_counts", []),
            "dominant_edge_type": dominant_edge_type,
            "orientation_reversed": bool(occ_face.Orientation() == TopAbs_REVERSED),
            "source_color_rgb": None,
            "source_color_hex": None,
            "source_color_alpha": None,
            "source_color_source": None,
            "length_unit_name": visual_meta.get("length_unit_name"),
            "length_unit_symbol": visual_meta.get("length_unit_symbol"),
            "length_unit_to_m": visual_meta.get("length_unit_to_m"),
            "angle_unit_name": visual_meta.get("angle_unit_name"),
            "angle_unit_symbol": visual_meta.get("angle_unit_symbol"),
            "angle_unit_to_rad": visual_meta.get("angle_unit_to_rad"),
            "source_layer_name": visual_meta.get("layer_name"),
            "source_layer_description": visual_meta.get("layer_description"),
            "source_layer_note": visual_meta.get("layer_note"),
            "source_layer_names": None,
            "source_layer_assignment_ids": None,
            "source_layer_ref_ids": None,
            "source_part_name": visual_meta.get("part_name"),
            "source_part_name_raw": visual_meta.get("part_name_raw"),
            "source_part_is_guid": visual_meta.get("part_is_guid"),
            "source_part_names": None,
            "source_part_candidate_count": None,
            "source_part_note": None,
            "source_solid_id": None,
            "source_solid_ids": None,
            "source_solid_count": 0,
            "step_advanced_face_id": None,
            "step_advanced_face_line": None,
            "step_advanced_face_expr": None,
            "step_surface_ref_id": None,
            "step_surface_entity_raw": None,
            "step_surface_entity_line": None,
            "step_surface_entity_expr": None,
            "step_surface_placement_ref_id": None,
            "step_surface_placement_entity_raw": None,
            "step_surface_placement_line": None,
            "step_surface_placement_expr": None,
            "step_surface_local_origin_ref_id": None,
            "step_surface_local_axis_ref_id": None,
            "step_surface_local_refdir_ref_id": None,
            "step_shell_id": None,
            "step_shell_ids": None,
            "step_shell_entity_raw": None,
            "step_manifold_solid_brep_id": None,
            "step_manifold_solid_brep_ids": None,
            "step_manifold_solid_brep_name": None,
            "step_manifold_solid_brep_names": None,
            "step_candidate_advanced_face_ids": None,
            "step_candidate_count": 0,
            "step_transfer_surface_entity_raw": None,
            "step_transfer_surface_signature": None,
            "step_entity_hierarchy": None,
            "step_ref_mapping": None,
            "step_surface_mapping_consistent": None,
            "local_origin": None,
            "local_x_dir": None,
            "local_y_dir": None,
            "local_z_dir": None,
            "local_frame_status": "UNAVAILABLE",
            "axis_tilt_to_global_z_deg": None,
            "face_length": None,
            "face_width": None,
            "face_thickness_est": None,
            "face_thickness_source": None,
            "face_size_basis": None,
            "face_span_local_xyz": None,
            "face_span_global_xyz": None,
            "cylindrical_like": False,
            "cylindrical_mode": None,
            "cyl_radius_est": None,
            "cyl_diameter_est": None,
            "cyl_height_est": None,
            "cyl_sweep_angle_rad_est": None,
            "cyl_sweep_angle_deg_est": None,
            "cyl_arc_length_est": None,
            "cyl_circumference_est": None,
            "cyl_axis_origin_est": None,
            "cyl_axis_direction_est": None,
            "cyl_circle_edge_count": 0,
            "cyl_metric_source": None,
            "mean_curvature": None,
            "gaussian_curvature": None,
            "min_curvature": None,
            "max_curvature": None,
            "curvature_eval_uv": None,
            "curvature_status": None,
            "contact_candidate_count": 0,
            "contact_area_total": 0.0,
            "contact_length_total": 0.0,
            "contact_pairs_top": [],
            "contact_status": "NOT_EVALUATED",
            "metric_sources": {
                "area": "OCCT:BRepGProp.SurfaceProperties",
                "edge_count": "OCCT:TopExp_Explorer(TopAbs_EDGE)",
                "edges": "OCCT:TopExp_Explorer(TopAbs_EDGE)+BRepAdaptor_Curve(+sampled polyline)",
                "edge_loop_count": "OCCT:TopExp_Explorer(TopAbs_WIRE)+per-wire edge type aggregation",
                "surface_type": "OCCT:BRepAdaptor_Surface.GetType",
                "global_coords": "OCCT:GProp_GProps.CentreOfMass + Bnd_Box",
                "units": "STEP_TEXT:SI_UNIT/CONVERSION_BASED_UNIT",
                "local_frame": "OCCT:Geom_Surface.Position(Ax3)",
                "uv_bounds": "OCCT:BRepTools.UVBounds",
                "axis_tilt_angle": "OCCT:local_z_dir vs global Z",
                "curvature": "OCCT:BRepLProp_SLProps@UVmid",
                "step_ref": "STEP_TEXT:ADVANCED_FACE(#id)->surface_ref(#id)",
                "face_size": f"{FACE_SIZE_BASIS_LOCAL_FRAME} (fallback: {FACE_SIZE_BASIS_GLOBAL_BBOX})",
                "face_thickness": (
                    f"{THICKNESS_SOURCE_OPPOSITE_PARALLEL} / {THICKNESS_SOURCE_PARALLEL}"
                    f" (fallback: {THICKNESS_SOURCE_LOCAL_Z_SPAN}, {THICKNESS_SOURCE_GLOBAL_BBOX})"
                ),
                "cylindrical_metrics": (
                    "OCCT:Geom_CylindricalSurface or Circle-edge axis inference"
                    "+axis_projection(height)+A/(R*H)(sweep)"
                ),
                "solid_membership": "OCCT:TopExp_Explorer(TopAbs_SOLID/TopAbs_FACE)+IsSame",
                "contact_area": "OCCT:BRepAlgoAPI_Common + BRepGProp.SurfaceProperties",
                "contact_length": "OCCT:BRepAlgoAPI_Common + BRepGProp.LinearProperties",
            },
            "_occ_face": occ_face,
        }

        if include_normal:
            face_info["normal_midpoint"] = None

        face_info.update(special)
        face_info.update(local_frame)
        lz = face_info.get("local_z_dir")
        if isinstance(lz, list) and len(lz) == 3:
            face_info["axis_tilt_to_global_z_deg"] = _angle_between_deg(lz, [0.0, 0.0, 1.0])

        solid_ids = _find_solid_ids_for_face(occ_face, solids_with_faces)
        face_info["source_solid_ids"] = solid_ids[:20]
        face_info["source_solid_count"] = len(solid_ids)
        face_info["source_solid_id"] = solid_ids[0] if solid_ids else None

        face_points = _face_points_from_edges(edge_items)
        size_metrics = _face_size_metrics(
            points=face_points,
            local_origin=face_info.get("local_origin"),
            local_x_dir=face_info.get("local_x_dir"),
            local_y_dir=face_info.get("local_y_dir"),
            local_z_dir=face_info.get("local_z_dir"),
            bbox_min=face_info.get("bbox_min") or [0.0, 0.0, 0.0],
            bbox_max=face_info.get("bbox_max") or [0.0, 0.0, 0.0],
        )
        face_info.update(size_metrics)

        if stype != GeomAbs_Plane:
            # Length/width/thickness are only stable as planar-face metrics.
            face_info["face_length"] = None
            face_info["face_width"] = None
            face_info["face_thickness_est"] = None
            face_info["face_thickness_source"] = None
            face_info["face_size_basis"] = None

        cyl_metrics = _estimate_cylindrical_metrics(
            surface_type=face_info["surface_type"],
            area=area,
            points=face_points,
            edge_items=edge_items,
            surface_radius=face_info.get("radius"),
            surface_axis_dir=(face_info.get("axis_direction") or face_info.get("local_z_dir")),
            surface_axis_origin=(face_info.get("axis_origin") or face_info.get("local_origin")),
        )
        face_info.update(cyl_metrics)

        if stype == GeomAbs_Plane and len(solid_ids) == 1:
            tval, tsource = _estimate_planar_face_thickness_in_solid(
                face=occ_face,
                face_normal=face_info.get("local_z_dir"),
                face_center=face_info.get("center_of_mass"),
                solid_id=solid_ids[0],
                solids=solids_with_faces,
            )
            if tval is not None:
                face_info["face_thickness_est"] = float(tval)
                face_info["face_thickness_source"] = tsource

        face_info["surface_type_ko"] = _surface_type_name_ko(face_info["surface_type"])
        face_info["surface_desc_ko"] = _surface_type_desc_ko(face_info["surface_type"])
        face_info["surface_step_entity"] = _surface_step_entity(face_info["surface_type"])
        face_info["dominant_edge_type_ko"] = _curve_type_name_ko(dominant_edge_type) if dominant_edge_type else None
        face_info["dominant_edge_step_entity"] = _curve_step_entity(dominant_edge_type) if dominant_edge_type else None
        face_info["edge_type_counts_ko"] = {_curve_type_name_ko(k): v for k, v in edge_type_counts.items()}
        face_info["edge_type_counts_step_entity"] = {
            _curve_step_entity(k) or k: v for k, v in edge_type_counts.items()
        }
        face_info["edge_loop_type_counts_ko"] = {
            _curve_type_name_ko(k): v for k, v in face_info["edge_loop_type_counts"].items()
        }
        face_info["edge_loop_type_counts_step_entity"] = {
            _curve_step_entity(k) or k: v for k, v in face_info["edge_loop_type_counts"].items()
        }
        face_info["edge_mix_summary_ko"] = _edge_mix_summary_ko(edge_type_counts)
        face_info["easy_hint_ko"] = _build_easy_hint_ko(face_info["surface_type"], dominant_edge_type, special)
        face_info.update(_face_curvature_mid(occ_face))

        visual = visual_faces.get(face_id)
        if isinstance(visual, dict):
            for k in [
                "source_color_rgb",
                "source_color_hex",
                "source_color_alpha",
                "source_color_source",
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
            ]:
                if k in visual:
                    face_info[k] = visual.get(k)

        step_surface_consistent = _surface_step_entity_consistency(
            face_info.get("surface_step_entity"),
            face_info.get("step_surface_entity_raw"),
        )
        face_info["step_surface_mapping_consistent"] = step_surface_consistent
        mapping_mode = str(face_info.get("step_ref_mapping") or "").strip().upper()
        if step_surface_consistent is False and mapping_mode.startswith("ORDER_"):
            # Avoid showing wrong STEP entity chain when order-based matching drifts.
            for k in [
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
                "step_entity_hierarchy",
            ]:
                face_info[k] = None
            face_info["step_ref_mapping"] = "ORDER_BASED_MISMATCH_SURFACE_TYPE"

        face_info["display_label"] = _build_face_label(
            face_id=face_id,
            surface_type=face_info["surface_type"],
            area=area,
            edge_count=edge_count,
            special=special,
        )
        faces.append(face_info)

        idx += 1
        ex.Next()

    return faces


def annotate_contact_metrics(
    faces: List[Dict[str, Any]],
    tolerance: float = 1e-4,
    max_pair_checks: int = 20000,
    top_k_per_face: int = 5,
) -> Dict[str, Any]:
    for face in faces:
        face["contact_candidate_count"] = 0
        face["contact_area_total"] = 0.0
        face["contact_length_total"] = 0.0
        face["contact_pairs_top"] = []
        face["contact_status"] = "NOT_EVALUATED"

    n = len(faces)
    checked = 0
    near_pairs = 0
    truncated = False

    for i in range(n):
        fi = faces[i]
        shape_i = fi.get("_occ_face")
        if shape_i is None:
            continue
        part_i = fi.get("source_part_name_raw") or fi.get("source_part_name")

        for j in range(i + 1, n):
            if checked >= max_pair_checks:
                truncated = True
                break

            fj = faces[j]
            shape_j = fj.get("_occ_face")
            if shape_j is None:
                continue
            part_j = fj.get("source_part_name_raw") or fj.get("source_part_name")
            if part_i and part_j and str(part_i) == str(part_j):
                continue

            if not _bbox_overlap_with_tol(
                fi.get("bbox_min", [0.0, 0.0, 0.0]),
                fi.get("bbox_max", [0.0, 0.0, 0.0]),
                fj.get("bbox_min", [0.0, 0.0, 0.0]),
                fj.get("bbox_max", [0.0, 0.0, 0.0]),
                tolerance,
            ):
                continue

            checked += 1
            dist = _shape_distance(shape_i, shape_j)
            if dist is None or dist > tolerance:
                continue

            area, length = _common_area_length(shape_i, shape_j)
            if area <= 1e-12 and length <= 1e-12:
                continue

            near_pairs += 1
            item_i = {
                "other_face_id": fj.get("face_id"),
                "other_part": fj.get("source_part_name"),
                "distance": float(dist),
                "contact_area": float(area),
                "contact_length": float(length),
                "method": "OCCT_BRepAlgoAPI_Common",
            }
            item_j = {
                "other_face_id": fi.get("face_id"),
                "other_part": fi.get("source_part_name"),
                "distance": float(dist),
                "contact_area": float(area),
                "contact_length": float(length),
                "method": "OCCT_BRepAlgoAPI_Common",
            }
            fi["contact_pairs_top"].append(item_i)
            fj["contact_pairs_top"].append(item_j)

        if truncated:
            break

    for face in faces:
        pairs = face.get("contact_pairs_top", [])
        pairs.sort(key=lambda x: (-float(x.get("contact_area", 0.0)), -float(x.get("contact_length", 0.0)), float(x.get("distance", 9e9))))
        if len(pairs) > top_k_per_face:
            del pairs[top_k_per_face:]
        face["contact_candidate_count"] = len(pairs)
        face["contact_area_total"] = float(sum(float(x.get("contact_area", 0.0)) for x in pairs))
        face["contact_length_total"] = float(sum(float(x.get("contact_length", 0.0)) for x in pairs))
        face["contact_status"] = "OK" if pairs else "NO_CONTACT_CANDIDATE"

    return {
        "contact_eval_tolerance": float(tolerance),
        "contact_eval_pair_checks": int(checked),
        "contact_eval_near_pairs": int(near_pairs),
        "contact_eval_truncated": bool(truncated),
        "contact_eval_max_pair_checks": int(max_pair_checks),
    }


def total_area(shape: Any) -> float:
    props = GProp_GProps()
    if USING_OCP:
        BRepGProp.SurfaceProperties_s(shape, props)
    else:
        brepgprop_SurfaceProperties(shape, props)
    return float(props.Mass())


def total_volume(shape: Any) -> float | None:
    props = GProp_GProps()
    try:
        if USING_OCP:
            BRepGProp.VolumeProperties_s(shape, props)
        else:
            brepgprop_VolumeProperties(shape, props)
        vol = float(props.Mass())
        return vol if abs(vol) > 1e-12 else None
    except Exception:
        return None
