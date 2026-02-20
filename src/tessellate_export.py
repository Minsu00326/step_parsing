from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

USING_OCP = False

try:
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_REVERSED
    from OCP.TopLoc import TopLoc_Location

    USING_OCP = True
except Exception:
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopAbs import TopAbs_REVERSED
    from OCC.Core.TopLoc import TopLoc_Location


def _triangle_normal(a: Tuple[float, float, float], b: Tuple[float, float, float], c: Tuple[float, float, float]) -> Tuple[float, float, float]:
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    n = math.sqrt(nx * nx + ny * ny + nz * nz)
    if n <= 1e-12:
        return (0.0, 0.0, 1.0)
    return (nx / n, ny / n, nz / n)


def export_obj_by_face(shape: Any, faces: List[Dict[str, Any]], out_obj: Path, linear_deflection: float = 0.2, angular_deflection: float = 0.3) -> None:
    out_obj.parent.mkdir(parents=True, exist_ok=True)

    # Conservative constructor for cross-binding compatibility.
    mesher = BRepMesh_IncrementalMesh(shape, linear_deflection)
    mesher.Perform()

    v_idx = 1
    n_idx = 1
    lines: List[str] = ["# Face-separated OBJ generated from STEP B-Rep"]

    for face_data in faces:
        face = face_data["_occ_face"]
        face_id = face_data["face_id"]

        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc) if USING_OCP else BRep_Tool.Triangulation(face, loc)
        if tri is None:
            continue

        trsf = loc.Transformation()

        lines.append(f"o {face_id}")

        for i in range(1, tri.NbTriangles() + 1):
            t = tri.Triangle(i) if hasattr(tri, "Triangle") else tri.Triangles().Value(i)
            i1, i2, i3 = t.Get()
            if face.Orientation() == TopAbs_REVERSED:
                i2, i3 = i3, i2

            p1 = (tri.Node(i1) if hasattr(tri, "Node") else tri.Nodes().Value(i1)).Transformed(trsf)
            p2 = (tri.Node(i2) if hasattr(tri, "Node") else tri.Nodes().Value(i2)).Transformed(trsf)
            p3 = (tri.Node(i3) if hasattr(tri, "Node") else tri.Nodes().Value(i3)).Transformed(trsf)

            a = (float(p1.X()), float(p1.Y()), float(p1.Z()))
            b = (float(p2.X()), float(p2.Y()), float(p2.Z()))
            c = (float(p3.X()), float(p3.Y()), float(p3.Z()))

            nx, ny, nz = _triangle_normal(a, b, c)

            lines.append(f"v {a[0]} {a[1]} {a[2]}")
            lines.append(f"v {b[0]} {b[1]} {b[2]}")
            lines.append(f"v {c[0]} {c[1]} {c[2]}")
            lines.append(f"vn {nx} {ny} {nz}")
            lines.append(f"f {v_idx}//{n_idx} {v_idx + 1}//{n_idx} {v_idx + 2}//{n_idx}")

            v_idx += 3
            n_idx += 1

    out_obj.write_text("\n".join(lines) + "\n", encoding="utf-8")
