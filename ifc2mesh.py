from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Dict, List

import ifcopenshell
import ifcopenshell.geom
import open3d as o3d


def _configure_geom_settings() -> ifcopenshell.geom.settings:
    settings = ifcopenshell.geom.settings()
    # IfcOpenShell 新版通常使用字符串设置名（如 use-world-coords）。
    # 为兼容不同版本，这里按“存在即设置”的方式启用关键选项。
    available = set()
    try:
        available = set(settings.setting_names())
    except Exception:
        available = set()

    preferred = [
        ("use-world-coords", True),
        ("disable-opening-subtractions", False),
        ("weld-vertices", True),
        ("unify-shapes", False),
    ]
    for key, value in preferred:
        if not available or key in available:
            try:
                settings.set(key, value)
            except Exception:
                pass

    # 兼容极少数旧版 API（枚举常量方式）
    legacy = [
        ("USE_WORLD_COORDS", True),
        ("APPLY_WORLD_COORDS", True),
        ("DISABLE_OPENING_SUBTRACTIONS", False),
    ]
    for key, value in legacy:
        try:
            settings.set(getattr(settings, key), value)
        except Exception:
            pass
    return settings


def _safe_name(text: str) -> str:
    s = (text or "unnamed").strip()
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", s)
    return s[:120] if len(s) > 120 else s


def _extract_profile_name(elem) -> str:
    # 优先从材质轮廓关系中读取截面名（Tekla/常见 IFC 导出通常在这里）
    for rel in getattr(elem, "HasAssociations", []) or []:
        if not rel.is_a("IfcRelAssociatesMaterial"):
            continue
        mat = getattr(rel, "RelatingMaterial", None)
        if mat is None:
            continue

        if mat.is_a("IfcMaterialProfileSetUsage"):
            pset = getattr(mat, "ForProfileSet", None)
            if pset and getattr(pset, "MaterialProfiles", None):
                names = []
                for mp in pset.MaterialProfiles:
                    p = getattr(mp, "Profile", None)
                    if p:
                        names.append(getattr(p, "ProfileName", None) or p.is_a())
                if names:
                    return "+".join([n for n in names if n])

        if mat.is_a("IfcMaterialProfileSet") and getattr(mat, "MaterialProfiles", None):
            names = []
            for mp in mat.MaterialProfiles:
                p = getattr(mp, "Profile", None)
                if p:
                    names.append(getattr(p, "ProfileName", None) or p.is_a())
            if names:
                return "+".join([n for n in names if n])

    # 回退：从属性集里找与 profile/section 相关字段
    for rel in getattr(elem, "IsDefinedBy", []) or []:
        pset = getattr(rel, "RelatingPropertyDefinition", None)
        if pset is None or not pset.is_a("IfcPropertySet"):
            continue
        for prop in getattr(pset, "HasProperties", []) or []:
            if not prop.is_a("IfcPropertySingleValue"):
                continue
            key = (getattr(prop, "Name", "") or "").lower()
            if "profile" in key or "section" in key:
                val = getattr(prop, "NominalValue", None)
                if val is not None:
                    return str(getattr(val, "wrappedValue", val))

    return "N/A"


def _to_open3d_mesh(verts: List[float], faces: List[int]) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector([
        [verts[i], verts[i + 1], verts[i + 2]] for i in range(0, len(verts), 3)
    ])
    mesh.triangles = o3d.utility.Vector3iVector([
        [faces[i], faces[i + 1], faces[i + 2]] for i in range(0, len(faces), 3)
    ])
    mesh.compute_vertex_normals()
    return mesh


def _merge_open3d_meshes(meshes: List[o3d.geometry.TriangleMesh]) -> o3d.geometry.TriangleMesh:
    merged = o3d.geometry.TriangleMesh()
    vertices: List[List[float]] = []
    triangles: List[List[int]] = []
    vertex_offset = 0

    for mesh in meshes:
        mesh_vertices = [[float(v[0]), float(v[1]), float(v[2])] for v in mesh.vertices]
        mesh_triangles = [[int(t[0]), int(t[1]), int(t[2])] for t in mesh.triangles]
        vertices.extend(mesh_vertices)
        triangles.extend(
            [[a + vertex_offset, b + vertex_offset, c + vertex_offset] for a, b, c in mesh_triangles]
        )
        vertex_offset += len(mesh_vertices)

    if vertices:
        merged.vertices = o3d.utility.Vector3dVector(vertices)
    if triangles:
        merged.triangles = o3d.utility.Vector3iVector(triangles)
        merged.compute_vertex_normals()
    return merged


def _mesh_centroid_and_diag(mesh: o3d.geometry.TriangleMesh):
    vertices = mesh.vertices
    n = len(vertices)
    if n == 0:
        return [0.0, 0.0, 0.0], 0.0

    xs = [float(v[0]) for v in vertices]
    ys = [float(v[1]) for v in vertices]
    zs = [float(v[2]) for v in vertices]

    cx = sum(xs) / n
    cy = sum(ys) / n
    cz = sum(zs) / n

    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    dz = max(zs) - min(zs)
    diag = math.sqrt(dx * dx + dy * dy + dz * dz)

    return [cx, cy, cz], float(diag)


def _validate_topology(records: List[Dict[str, object]]) -> Dict[str, object]:
    ok_records = [r for r in records if r.get("status") == "ok"]
    centroids = [
        r.get("centroid")
        for r in ok_records
        if isinstance(r.get("centroid"), list) and len(r.get("centroid")) == 3
    ]
    diag_values = [
        float(r.get("bbox_diag", 0.0))
        for r in ok_records
        if float(r.get("bbox_diag", 0.0)) > 0
    ]

    if len(centroids) < 2:
        return {
            "status": "insufficient-data",
            "reason": "有效构件数量不足，无法判断拓扑分布",
            "mesh_count": len(centroids),
        }

    xs = [float(c[0]) for c in centroids]
    ys = [float(c[1]) for c in centroids]
    zs = [float(c[2]) for c in centroids]
    spread_xyz = [
        max(xs) - min(xs),
        max(ys) - min(ys),
        max(zs) - min(zs),
    ]

    max_centroid_distance = 0.0
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            dx = float(centroids[i][0]) - float(centroids[j][0])
            dy = float(centroids[i][1]) - float(centroids[j][1])
            dz = float(centroids[i][2]) - float(centroids[j][2])
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if d > max_centroid_distance:
                max_centroid_distance = d

    median_bbox_diag = statistics.median(diag_values) if diag_values else 0.0
    ratio = (max_centroid_distance / median_bbox_diag) if median_bbox_diag > 1e-9 else 0.0

    # 经验阈值：如果构件间最大中心距仅为“单构件中位尺寸”的几倍，可能仍处于局部坐标导出。
    threshold = 5.0
    suspicious_local_coords = ratio < threshold

    return {
        "status": "warn" if suspicious_local_coords else "pass",
        "suspicious_local_coords": suspicious_local_coords,
        "threshold": threshold,
        "mesh_count": len(centroids),
        "spread_xyz": [round(v, 6) for v in spread_xyz],
        "max_centroid_distance": round(max_centroid_distance, 6),
        "median_bbox_diag": round(median_bbox_diag, 6),
        "distance_to_size_ratio": round(ratio, 6),
        "hint": "若为 warn，请确认已启用 use-world-coords 或检查 IFC 变换链",
    }


def convert_ifc_to_mesh(input_ifc: Path, out_dir: Path, element_type: str, validate_topology: bool = False) -> Path:
    if not input_ifc.exists():
        raise FileNotFoundError(f"IFC 文件不存在: {input_ifc}")

    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    ifc_file = ifcopenshell.open(str(input_ifc))
    settings = _configure_geom_settings()

    if element_type.lower() == "all":
        elements = [
            e
            for e in ifc_file.by_type("IfcProduct")
            if hasattr(e, "Representation") and getattr(e, "Representation", None) is not None
        ]
    else:
        elements = ifc_file.by_type(element_type)

    records: List[Dict[str, object]] = []
    ok_meshes: List[o3d.geometry.TriangleMesh] = []
    ok = 0
    for idx, elem in enumerate(elements, start=1):
        guid = getattr(elem, "GlobalId", "") or ""
        name = getattr(elem, "Name", None) or elem.is_a()
        profile = _extract_profile_name(elem)

        try:
            shape = ifcopenshell.geom.create_shape(settings, elem)
            verts = shape.geometry.verts
            faces = shape.geometry.faces
            if len(verts) < 9 or len(faces) < 3:
                raise RuntimeError("空几何或几何过小")

            mesh = _to_open3d_mesh(verts, faces)
            centroid, bbox_diag = _mesh_centroid_and_diag(mesh)
            mesh_name = f"{idx:04d}_{_safe_name(name)}_{guid}.ply"
            mesh_path = mesh_dir / mesh_name
            saved = o3d.io.write_triangle_mesh(str(mesh_path), mesh)
            if not saved:
                raise RuntimeError(f"网格写出失败: {mesh_path}")
            ok += 1
            ok_meshes.append(mesh)

            print(
                f"[{idx}/{len(elements)}] {name} | GUID={guid} | Profile={profile} | "
                f"V={len(verts)//3}, F={len(faces)//3} -> {mesh_path.name}"
            )
            records.append(
                {
                    "index": idx,
                    "ifc_type": elem.is_a(),
                    "name": name,
                    "guid": guid,
                    "profile": profile,
                    "vertex_count": int(len(verts) // 3),
                    "face_count": int(len(faces) // 3),
                    "centroid": [round(float(c), 6) for c in centroid],
                    "bbox_diag": round(float(bbox_diag), 6),
                    "mesh_file": str(mesh_path),
                    "status": "ok",
                }
            )
        except Exception as exc:
            print(f"[{idx}/{len(elements)}] {name} | GUID={guid} | Profile={profile} | 失败: {exc}")
            records.append(
                {
                    "index": idx,
                    "ifc_type": elem.is_a(),
                    "name": name,
                    "guid": guid,
                    "profile": profile,
                    "status": f"failed: {exc}",
                }
            )

    merged_mesh_path = out_dir / "all_ply.ply"
    merged_mesh_written = False
    if ok_meshes:
        merged_mesh = _merge_open3d_meshes(ok_meshes)
        merged_mesh_written = o3d.io.write_triangle_mesh(str(merged_mesh_path), merged_mesh)
        if not merged_mesh_written:
            print(f"合并网格写出失败: {merged_mesh_path}")

    report = {
        "input_ifc": str(input_ifc),
        "element_type": element_type,
        "total_elements": len(elements),
        "converted_ok": ok,
        "failed": int(len(elements) - ok),
        "merged_mesh_file": str(merged_mesh_path) if merged_mesh_written else None,
        "records": records,
    }
    if validate_topology:
        report["topology_validation"] = _validate_topology(records)

    report_path = out_dir / "ifc_mesh_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n转换完成: 成功 {ok}/{len(elements)}")
    if merged_mesh_written:
        print(f"合并网格: {merged_mesh_path}")
    if validate_topology:
        tv = report.get("topology_validation", {})
        print(
            "拓扑校验: "
            f"{tv.get('status', 'unknown').upper()} | "
            f"ratio={tv.get('distance_to_size_ratio', 'N/A')} | "
            f"spread={tv.get('spread_xyz', 'N/A')}"
        )
    print(f"Mesh 目录: {mesh_dir}")
    print(f"报告文件: {report_path}")
    return report_path


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="IFC 构件遍历 + Profile/GUID/Name 提取 + Mesh 导出")
    p.add_argument("--input-ifc", type=str, default=str(here / "out.ifc"))
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录；默认使用 IFC 文件同级的 result 子文件夹",
    )
    p.add_argument(
        "--element-type",
        type=str,
        default="all",
        help="要处理的 IFC 类型（例如 IfcBeam, IfcColumn, IfcBuildingElementProxy），默认 all",
    )
    p.add_argument(
        "--validate-topology",
        action="store_true",
        help="导出后自动校验构件坐标分布，检测是否疑似局部坐标导出",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_ifc = Path(args.input_ifc).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (input_ifc.parent / "result").resolve()
    convert_ifc_to_mesh(
        input_ifc,
        output_dir,
        args.element_type,
        validate_topology=args.validate_topology,
    )


if __name__ == "__main__":
    main()