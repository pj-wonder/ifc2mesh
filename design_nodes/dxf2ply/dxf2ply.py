"""
DXF -> PLY 采样与节点偏差计算工具。

这个脚本面向两类工作流：
1. 将 DXF 中的线结构按固定间距采样，输出为 ASCII PLY，便于后续点云处理。
2. 读取 `line2point.py` 导出的设计节点 CSV，与实测 PLY 做最近点匹配，输出 xyz 偏差 CSV。

为了方便和现有流程联动，偏差输出字段尽量对齐 `line2point.py`。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import ezdxf
import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None


SCRIPT_DIR = Path(__file__).resolve().parent

from design_nodes import line2point as mod


Point3D = Tuple[float, float, float]


def create_timestamped_output_dir(base_dir: Optional[str] = None) -> Path:
    if base_dir is None:
        base_dir = SCRIPT_DIR / "result"
    else:
        base_dir = Path(base_dir).resolve()

    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir.resolve()


def _default_run_name(prefix: str, run_stamp: str, suffix: str) -> str:
    return f"{run_stamp}_{prefix}{suffix}"


def _normalize_point(values: Sequence[Any]) -> Point3D:
    coords = [float(value) for value in values]
    if len(coords) == 2:
        coords.append(0.0)
    return float(coords[0]), float(coords[1]), float(coords[2])


def _segment_length(start_point: Point3D, end_point: Point3D) -> float:
    return float(math.dist(start_point, end_point))


def _extract_axis_segments(entity: Any) -> List[Tuple[Point3D, Point3D]]:
    entity_type = entity.dxftype()

    if entity_type == "LINE":
        start_pt = entity.dxf.start
        end_pt = entity.dxf.end
        return [
            (
                _normalize_point((start_pt.x, start_pt.y, start_pt.z)),
                _normalize_point((end_pt.x, end_pt.y, end_pt.z)),
            )
        ]

    if entity_type == "LWPOLYLINE":
        vertices: List[Point3D] = []
        for point in entity.get_points("xy"):
            vertices.append((float(point[0]), float(point[1]), 0.0))
        return [(vertices[index], vertices[index + 1]) for index in range(len(vertices) - 1)]

    if entity_type == "POLYLINE":
        vertices: List[Point3D] = []
        for vertex in entity.vertices:
            location = vertex.dxf.location
            vertices.append(_normalize_point((location.x, location.y, getattr(location, "z", 0.0))))
        return [(vertices[index], vertices[index + 1]) for index in range(len(vertices) - 1)]

    return []


def _node_key(point: Point3D, tolerance: float) -> Tuple[int, int, int]:
    scale = 1.0 / tolerance if tolerance > 0 else 1000.0
    return (
        round(point[0] * scale),
        round(point[1] * scale),
        round(point[2] * scale),
    )


def _read_point_cloud_safe(ply_path: Path) -> Any:
    if o3d is None:
        raise RuntimeError("open3d 未安装，无法读取 PLY。请先执行: pip install open3d")

    try:
        return o3d.io.read_point_cloud(str(ply_path))
    except UnicodeDecodeError:
        ascii_name = f"pc_input_{ply_path.stem}{ply_path.suffix}"
        with tempfile.TemporaryDirectory(prefix="ply_ascii_") as tmp_dir:
            ascii_path = Path(tmp_dir) / ascii_name
            shutil.copyfile(ply_path, ascii_path)
            return o3d.io.read_point_cloud(str(ascii_path))


def sample_dxf_to_points(
    dxf_path: str,
    spacing: float = 100.0,
    min_axis_length: float = 1000.0,
) -> List[Point3D]:
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    sampled_points: List[Point3D] = []
    for entity in msp:
        segments = _extract_axis_segments(entity)
        if not segments:
            continue

        for start_point, end_point in segments:
            length = _segment_length(start_point, end_point)
            if length < float(min_axis_length):
                continue

            if spacing <= 0 or length <= spacing:
                sampled_points.append(start_point)
                sampled_points.append(end_point)
                continue

            step_count = max(1, int(math.floor(length / float(spacing))))
            for index in range(step_count + 1):
                t = index / float(step_count)
                x = start_point[0] + (end_point[0] - start_point[0]) * t
                y = start_point[1] + (end_point[1] - start_point[1]) * t
                z = start_point[2] + (end_point[2] - start_point[2]) * t
                sampled_points.append((float(x), float(y), float(z)))

    unique_points: Dict[Tuple[int, int, int], Point3D] = {}
    tolerance = 1e-6
    for point in sampled_points:
        unique_points.setdefault(_node_key(point, tolerance), point)

    return list(unique_points.values())


def write_ply(points: List[Point3D], ply_path: str) -> None:
    output_path = Path(ply_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

    print(f"PLY 已保存: {output_path} (points={len(points)})")


def write_node_ply(node_records: List[Dict[str, Any]], ply_path: str) -> None:
    points: List[Point3D] = [
        (float(node["x"]), float(node["y"]), float(node["z"]))
        for node in node_records
    ]
    write_ply(points, ply_path)


def _read_ply_points(ply_path: str) -> List[Point3D]:
    path = Path(ply_path)
    if not path.exists():
        raise FileNotFoundError(f"PLY 不存在: {path}")

    if o3d is not None:
        pcd = _read_point_cloud_safe(path)
        points = np.asarray(pcd.points, dtype=float)
        return [(float(p[0]), float(p[1]), float(p[2])) for p in points]

    points: List[Point3D] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        header_done = False
        for line in f:
            if not header_done:
                if line.strip() == "end_header":
                    header_done = True
                continue

            parts = line.strip().split()
            if len(parts) < 3:
                continue

            try:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
            except ValueError:
                continue
            points.append((x, y, z))

    return points


def _read_design_nodes_csv(design_nodes_csv: str) -> List[Dict[str, Any]]:
    design_path = Path(design_nodes_csv)
    if not design_path.exists():
        raise FileNotFoundError(f"设计节点 CSV 不存在: {design_path}")

    design_rows: List[Dict[str, Any]] = []
    with design_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row.get("x", row.get("design_x", "")))
                y = float(row.get("y", row.get("design_y", "")))
                z = float(row.get("z", row.get("design_z", "")))
            except (TypeError, ValueError):
                continue

            design_rows.append(
                {
                    "node_id": str(row.get("node_id", row.get("node", ""))),
                    "node_role": str(row.get("node_role", "")),
                    "axis_ids": str(row.get("axis_ids", "")),
                    "axis_count": row.get("axis_count", ""),
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )

    if not design_rows:
        raise RuntimeError("未读取到任何设计节点")

    return design_rows


def compare_design_nodes_to_measured_points(
    node_records: List[Dict[str, Any]],
    measured_points: List[Point3D],
    max_match_distance: float = 999999.0,
) -> List[Dict[str, Any]]:
    if not node_records:
        return []
    if not measured_points:
        raise RuntimeError("测量节点点云为空，无法比较偏差")

    design_array = np.asarray(
        [[float(node["x"]), float(node["y"]), float(node["z"])] for node in node_records],
        dtype=float,
    )

    rows: List[Dict[str, Any]] = []
    for measured_idx, measured_point_tuple in enumerate(measured_points, start=1):
        measured_point = np.asarray(measured_point_tuple, dtype=float)
        dist_all = np.linalg.norm(design_array - measured_point, axis=1)
        design_idx = int(np.argmin(dist_all))
        matched_dist = float(dist_all[design_idx])

        if not math.isfinite(matched_dist) or matched_dist > float(max_match_distance):
            rows.append(
                {
                    "measured_index": int(measured_idx),
                    "measured_x": float(measured_point[0]),
                    "measured_y": float(measured_point[1]),
                    "measured_z": float(measured_point[2]),
                    "node_id": "",
                    "node_role": "",
                    "axis_count": "",
                    "axis_ids": "",
                    "design_index": "",
                    "design_x": "",
                    "design_y": "",
                    "design_z": "",
                    "dx": "",
                    "dy": "",
                    "dz": "",
                    "deviation_3d": "",
                    "matched": False,
                }
            )
            continue

        design_node = node_records[design_idx]
        axis_ids = design_node.get("axis_ids", [])
        if isinstance(axis_ids, str):
            axis_ids = [item for item in axis_ids.split(";") if item]

        design_point = design_array[design_idx]
        delta = measured_point - design_point

        rows.append(
            {
                "measured_index": int(measured_idx),
                "measured_x": float(measured_point[0]),
                "measured_y": float(measured_point[1]),
                "measured_z": float(measured_point[2]),
                "node_id": str(design_node.get("node_id", "")),
                "node_role": str(design_node.get("node_role", "")),
                "axis_count": int(len(axis_ids)),
                "axis_ids": ";".join(str(item) for item in axis_ids),
                "axis_pairs": str(design_node.get("axis_pairs", "")),
                "entity_type": str(design_node.get("entity_type", "")),
                "layer": str(design_node.get("layer", "")),
                "design_index": int(design_idx + 1),
                "design_x": float(design_point[0]),
                "design_y": float(design_point[1]),
                "design_z": float(design_point[2]),
                "design_is_intersection_expected": bool(design_node.get("is_intersection_expected", False))
                if str(design_node.get("is_intersection_expected", "")) != ""
                else "",
                "design_solve_method": str(design_node.get("solve_method", "")),
                "design_member_count": design_node.get("member_count", ""),
                "design_on_segment_ratio": design_node.get("on_segment_ratio", ""),
                "dx": float(delta[0]),
                "dy": float(delta[1]),
                "dz": float(delta[2]),
                "deviation_3d": float(np.linalg.norm(delta)),
                "matched": True,
            }
        )

    return rows


def summarize_node_deviation_records(deviation_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    matched_records = [record for record in deviation_records if bool(record.get("matched", False))]
    deviations = [float(record["deviation_3d"]) for record in matched_records]

    summary: Dict[str, Any] = {
        "matched_count": int(len(matched_records)),
        "total_count": int(len(deviation_records)),
        "unmatched_count": int(len(deviation_records) - len(matched_records)),
    }

    if deviations:
        summary.update(
            {
                "mean_deviation_3d": float(np.mean(deviations)),
                "max_deviation_3d": float(np.max(deviations)),
                "min_deviation_3d": float(np.min(deviations)),
                "rmse_deviation_3d": float(np.sqrt(np.mean(np.square(deviations)))),
            }
        )

    return summary


def build_design_node_crosswalk_records(
    node_records: List[Dict[str, Any]],
    deviation_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for deviation in deviation_records:
        rows.append(
            {
                "measured_index": deviation.get("measured_index", ""),
                "measured_x": deviation.get("measured_x", ""),
                "measured_y": deviation.get("measured_y", ""),
                "measured_z": deviation.get("measured_z", ""),
                "node_id": deviation.get("node_id", ""),
                "node_role": deviation.get("node_role", ""),
                "axis_count": deviation.get("axis_count", ""),
                "axis_ids": deviation.get("axis_ids", ""),
                "axis_pairs": deviation.get("axis_pairs", ""),
                "entity_type": deviation.get("entity_type", ""),
                "layer": deviation.get("layer", ""),
                "design_index": deviation.get("design_index", ""),
                "design_x": deviation.get("design_x", ""),
                "design_y": deviation.get("design_y", ""),
                "design_z": deviation.get("design_z", ""),
                "design_is_intersection_expected": deviation.get("design_is_intersection_expected", ""),
                "design_solve_method": deviation.get("design_solve_method", ""),
                "design_member_count": deviation.get("design_member_count", ""),
                "design_on_segment_ratio": deviation.get("design_on_segment_ratio", ""),
                "matched": bool(deviation.get("matched", False)),
                "dx": deviation.get("dx", ""),
                "dy": deviation.get("dy", ""),
                "dz": deviation.get("dz", ""),
                "deviation_3d": deviation.get("deviation_3d", ""),
            }
        )

    return rows


def export_records_to_csv(csv_path: str, fieldnames: List[str], records: List[Dict[str, Any]]) -> None:
    output_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"CSV 已保存: {output_path} (records={len(records)})")


def _load_points_from_cloud(path: str) -> np.ndarray:
    cloud_path = Path(path)
    if not cloud_path.exists():
        raise FileNotFoundError(f"点云文件不存在: {cloud_path}")

    suffix = cloud_path.suffix.lower()
    if suffix == ".ply":
        points = _read_ply_points(str(cloud_path))
        return np.asarray(points, dtype=float)

    pcd = _read_point_cloud_safe(cloud_path)
    return np.asarray(pcd.points, dtype=float)


def _point_cloud_from_points(points: np.ndarray, color: Sequence[float]) -> Any:
    if o3d is None:
        raise RuntimeError("open3d 未安装，无法构建可视化点云")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    if points.shape[0] > 0:
        color_array = np.tile(np.asarray(color, dtype=float)[None, :], (points.shape[0], 1))
        pcd.colors = o3d.utility.Vector3dVector(color_array)
    return pcd


def visualize_points_open3d(points: Sequence[Point3D], window_name: str = "DXF Sampled Points") -> None:
    if o3d is None:
        raise RuntimeError("open3d 未安装，无法可视化。请先执行: pip install open3d")

    point_array = np.asarray(points, dtype=float)
    if point_array.size == 0:
        raise RuntimeError("没有可视化的点")

    geometries = [_point_cloud_from_points(point_array, [0.18, 0.45, 0.95])]
    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_name,
        width=1600,
        height=1000,
    )


def visualize_sampled_points_and_nodes_open3d(
    sampled_points: Sequence[Point3D],
    node_records: Sequence[Dict[str, Any]],
    window_name: str = "DXF Sampled Points and Junction Nodes",
) -> None:
    if o3d is None:
        raise RuntimeError("open3d 未安装，无法可视化。请先执行: pip install open3d")

    geometries: List[Any] = []

    sampled_array = np.asarray(sampled_points, dtype=float)
    if sampled_array.size > 0:
        geometries.append(_point_cloud_from_points(sampled_array, [0.18, 0.45, 0.95]))

    node_points = np.asarray(
        [[float(node["x"]), float(node["y"]), float(node["z"])] for node in node_records],
        dtype=float,
    )
    if node_points.size > 0:
        geometries.append(_point_cloud_from_points(node_points, [0.95, 0.25, 0.18]))

    if not geometries:
        raise RuntimeError("没有可视化的点")

    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_name,
        width=1600,
        height=1000,
    )


def compare_axis_node_plys_with_measured_cloud(
    axis_ply: str,
    expected_nodes_ply: str,
    measured_cloud: str,
    output_dir: Optional[str] = None,
    show_open3d: bool = False,
    measured_downsample_step: int = 1,
) -> Dict[str, Any]:
    if o3d is None:
        raise RuntimeError("open3d 未安装，无法进行可视化比较。请先执行: pip install open3d")

    axis_points = _load_points_from_cloud(axis_ply)
    expected_nodes = _load_points_from_cloud(expected_nodes_ply)
    measured_points = _load_points_from_cloud(measured_cloud)

    if measured_downsample_step > 1 and measured_points.shape[0] > 0:
        measured_points = measured_points[:: int(measured_downsample_step)]

    geometries: List[Any] = []
    if axis_points.shape[0] > 0:
        geometries.append(_point_cloud_from_points(axis_points, [0.15, 0.45, 0.95]))
    if expected_nodes.shape[0] > 0:
        geometries.append(_point_cloud_from_points(expected_nodes, [1.0, 0.0, 0.0]))
    if measured_points.shape[0] > 0:
        geometries.append(_point_cloud_from_points(measured_points, [0.72, 0.72, 0.72]))

    overlay_points: List[np.ndarray] = []
    overlay_colors: List[np.ndarray] = []
    if axis_points.shape[0] > 0:
        overlay_points.append(axis_points)
        overlay_colors.append(np.tile(np.array([[0.15, 0.45, 0.95]], dtype=float), (axis_points.shape[0], 1)))
    if expected_nodes.shape[0] > 0:
        overlay_points.append(expected_nodes)
        overlay_colors.append(np.tile(np.array([[1.0, 0.0, 0.0]], dtype=float), (expected_nodes.shape[0], 1)))
    if measured_points.shape[0] > 0:
        overlay_points.append(measured_points)
        overlay_colors.append(np.tile(np.array([[0.72, 0.72, 0.72]], dtype=float), (measured_points.shape[0], 1)))

    overlay_path = None
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        overlay_path = output_path / "axis_nodes_measured_overlay.ply"

        if overlay_points:
            merged_points = np.vstack(overlay_points)
            merged_colors = np.vstack(overlay_colors)
        else:
            merged_points = np.zeros((0, 3), dtype=float)
            merged_colors = np.zeros((0, 3), dtype=float)

        overlay = o3d.geometry.PointCloud()
        overlay.points = o3d.utility.Vector3dVector(merged_points)
        overlay.colors = o3d.utility.Vector3dVector(merged_colors)
        o3d.io.write_point_cloud(str(overlay_path), overlay)

    summary = {
        "axis_point_count": int(axis_points.shape[0]),
        "expected_node_count": int(expected_nodes.shape[0]),
        "measured_point_count": int(measured_points.shape[0]),
        "overlay_ply": str(overlay_path) if overlay_path is not None else None,
    }

    print(
        f"比较数据准备完成: axis_points={summary['axis_point_count']}, "
        f"expected_nodes={summary['expected_node_count']}, measured_points={summary['measured_point_count']}"
    )
    if overlay_path is not None:
        print(f"可视化叠加文件已保存: {overlay_path}")

    if show_open3d and geometries:
        o3d.visualization.draw_geometries(
            geometries,
            window_name="DXF Axis / Expected Nodes / Measured Cloud",
            width=1600,
            height=1000,
        )

    return summary


def extract_design_records_from_dxf(
    dxf_path: str,
    min_axis_length: float = 1000.0,
    node_distance_threshold: float = 5.0,
    design_scale: float = 1.0,
    junction_min_axis_count: int = 2,
    segment_margin: Optional[float] = None,
    node_grouping_radius: Optional[float] = None,
    multi_min_members: int = 3,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    axis_records, node_records = mod.extract_axis_data(
        dxf_path=dxf_path,
        min_axis_length=min_axis_length,
        node_distance_threshold=node_distance_threshold,
        segment_margin=segment_margin,
        node_grouping_radius=node_grouping_radius,
        multi_min_members=multi_min_members,
    )
    axis_records, node_records = mod.scale_axis_node_records(axis_records, node_records, design_scale)
    node_records = mod.filter_junction_nodes(node_records, min_axis_count=junction_min_axis_count)
    return axis_records, node_records


def compute_node_deviations_from_records(
    node_records: List[Dict[str, Any]],
    measured_ply: str,
    output_csv: Optional[str] = None,
    output_crosswalk_csv: Optional[str] = None,
    output_json: Optional[str] = None,
    max_match_distance: float = 999999.0,
    summary_extra: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    measured_points = _read_ply_points(measured_ply)
    if not measured_points:
        raise RuntimeError("未读取到任何实测点")

    deviation_records = compare_design_nodes_to_measured_points(
        node_records=node_records,
        measured_points=measured_points,
        max_match_distance=max_match_distance,
    )

    if output_csv:
        export_records_to_csv(
            csv_path=output_csv,
            fieldnames=[
                "measured_index",
                "measured_x",
                "measured_y",
                "measured_z",
                "node_id",
                "node_role",
                "axis_count",
                "axis_ids",
                "axis_pairs",
                "entity_type",
                "layer",
                "design_index",
                "design_x",
                "design_y",
                "design_z",
                "design_is_intersection_expected",
                "design_solve_method",
                "design_member_count",
                "design_on_segment_ratio",
                "dx",
                "dy",
                "dz",
                "deviation_3d",
                "matched",
            ],
            records=deviation_records,
        )

    crosswalk_records = build_design_node_crosswalk_records(node_records, deviation_records)
    if output_crosswalk_csv:
        export_records_to_csv(
            csv_path=output_crosswalk_csv,
            fieldnames=[
                "measured_index",
                "measured_x",
                "measured_y",
                "measured_z",
                "node_id",
                "node_role",
                "axis_count",
                "axis_ids",
                "axis_pairs",
                "entity_type",
                "layer",
                "design_index",
                "design_x",
                "design_y",
                "design_z",
                "design_is_intersection_expected",
                "design_solve_method",
                "design_member_count",
                "design_on_segment_ratio",
                "matched",
                "dx",
                "dy",
                "dz",
                "deviation_3d",
            ],
            records=crosswalk_records,
        )

    if output_json:
        summary = summarize_node_deviation_records(deviation_records)
        solve_methods = [str(node.get("solve_method", "")) for node in node_records if str(node.get("solve_method", ""))]
        solve_method_counts: Dict[str, int] = {}
        for method in solve_methods:
            solve_method_counts[method] = solve_method_counts.get(method, 0) + 1
        summary.update(
            {
                "measured_ply": str(measured_ply),
                "design_node_count": int(len(node_records)),
                "measured_point_count": int(len(measured_points)),
                "output_csv": output_csv,
                "output_crosswalk_csv": output_crosswalk_csv,
                "max_match_distance": float(max_match_distance),
                "design_node_fields": [
                    "node_id",
                    "node_role",
                    "axis_ids",
                    "axis_count",
                    "axis_pairs",
                    "entity_type",
                    "layer",
                    "x",
                    "y",
                    "z",
                    "distance",
                    "max_distance",
                    "is_intersection_expected",
                    "solve_method",
                    "member_count",
                    "on_segment_ratio",
                ],
                "design_solve_method_counts": solve_method_counts,
            }
        )
        if summary_extra:
            summary.update(summary_extra)
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"偏差报告 JSON 已保存: {output_path}")

    return deviation_records


def compute_node_deviations(
    design_nodes_csv: str,
    measured_ply: str,
    output_csv: Optional[str] = None,
    output_crosswalk_csv: Optional[str] = None,
    output_json: Optional[str] = None,
    max_match_distance: float = 999999.0,
) -> List[Dict[str, Any]]:
    design_rows = _read_design_nodes_csv(design_nodes_csv)
    return compute_node_deviations_from_records(
        node_records=design_rows,
        measured_ply=measured_ply,
        output_csv=output_csv,
        output_crosswalk_csv=output_crosswalk_csv,
        output_json=output_json,
        max_match_distance=max_match_distance,
        summary_extra={"design_nodes_csv": str(design_nodes_csv)},
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DXF 采样转 PLY，并支持设计/实测节点偏差比较")
    parser.add_argument("dxf_path", nargs="?", default=None, help="输入 DXF 路径")
    parser.add_argument("--output-ply", default=None, help="输出 PLY 路径")
    parser.add_argument("--output-node-ply", default=None, help="输出 DXF 汇交节点 PLY 路径")
    parser.add_argument("--spacing", type=float, default=100.0, help="采样间距，单位与 DXF 相同")
    parser.add_argument("--min-axis-length", type=float, default=1000.0, help="忽略短于该长度的轴段")
    parser.add_argument("--node-distance-threshold", type=float, default=5.0, help="两条长轴视为交点的最大距离阈值")
    parser.add_argument("--segment-margin", type=float, default=None, help="轴段内判断的投影余量；为空则按轴长的5%")
    parser.add_argument("--node-grouping-radius", type=float, default=None, help="候选交点聚类半径；为空则沿用节点合并容差")
    parser.add_argument("--multi-min-members", type=int, default=3, help="触发全局最小二乘交点的最小轴线数量")
    parser.add_argument("--design-scale", type=float, default=1.0, help="在比较前对 DXF 设计坐标统一缩放")
    parser.add_argument("--junction-min-axis-count", type=int, default=2, help="保留至少连接该数量轴线的节点")
    parser.add_argument("--design-nodes", default=None, help="设计节点 CSV，通常来自 line2point.py")
    parser.add_argument("--measured-ply", default=None, help="实测节点 PLY 文件")
    parser.add_argument("--output-deviation", default=None, help="输出偏差 CSV 路径")
    parser.add_argument("--output-crosswalk", default=None, help="输出设计节点交叉对照 CSV 路径")
    parser.add_argument("--output-json", default=None, help="输出偏差统计 JSON 路径")
    parser.add_argument("--node-match-distance", type=float, default=999999.0, help="节点匹配最大距离")
    parser.add_argument("--output-dir", default=None, help="输出目录，未显式指定文件时使用")
    parser.add_argument("--report-name", default="node_deviation_report", help="输出文件基础名")
    parser.add_argument("--axis-ply", default=None, help="已有的轴点 PLY，例如 axis_points_blue.ply")
    parser.add_argument("--expected-nodes-ply", default=None, help="已有的期望节点 PLY，例如 nodes_expected_red.ply")
    parser.add_argument("--measured-cloud", default=None, help="实测点云 PCD/PLY")
    parser.add_argument("--compare-show-open3d", action="store_true", help="显示轴点/节点/实测点云的 Open3D 可视化窗口")
    parser.add_argument("--measured-downsample-step", type=int, default=1, help="实测点云可视化下采样步长")
    parser.add_argument("--show-open3d", action="store_true", help="显示 DXF 采样点云的 Open3D 可视化窗口")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    has_any_operation = bool(
        args.dxf_path
        or (args.design_nodes and args.measured_ply)
        or (args.axis_ply and args.expected_nodes_ply and args.measured_cloud)
    )
    if not has_any_operation:
        print("未执行任何操作。请参考 --help 用法。")
        return

    run_output_dir = create_timestamped_output_dir(args.output_dir)
    run_stamp = run_output_dir.name
    print(f"[RUN] timestamp={run_stamp}")
    print(f"[RUN] output_dir={run_output_dir}")

    sampled_ply_path: Optional[Path] = None
    node_ply_path: Optional[Path] = None
    axis_records_from_dxf: Optional[List[Dict[str, Any]]] = None
    node_records_from_dxf: Optional[List[Dict[str, Any]]] = None

    if args.dxf_path:
        axis_records_from_dxf, node_records_from_dxf = extract_design_records_from_dxf(
            dxf_path=args.dxf_path,
            min_axis_length=args.min_axis_length,
            node_distance_threshold=args.node_distance_threshold,
            design_scale=args.design_scale,
            junction_min_axis_count=args.junction_min_axis_count,
            segment_margin=args.segment_margin,
            node_grouping_radius=args.node_grouping_radius,
            multi_min_members=args.multi_min_members,
        )

        sampled_points = sample_dxf_to_points(
            dxf_path=args.dxf_path,
            spacing=args.spacing,
            min_axis_length=args.min_axis_length,
        )
        if abs(float(args.design_scale) - 1.0) >= 1e-12:
            sampled_points = [
                (float(point[0]) * float(args.design_scale), float(point[1]) * float(args.design_scale), float(point[2]) * float(args.design_scale))
                for point in sampled_points
            ]
        sampled_ply_path = Path(args.output_ply) if args.output_ply else run_output_dir / _default_run_name("dxf_line_samples", run_stamp, ".ply")
        write_ply(sampled_points, str(sampled_ply_path))

        design_axes_csv = run_output_dir / _default_run_name("dxf_axes", run_stamp, ".csv")
        design_nodes_csv = run_output_dir / _default_run_name("dxf_nodes", run_stamp, ".csv")
        node_ply_path = Path(args.output_node_ply) if args.output_node_ply else run_output_dir / _default_run_name("dxf_nodes", run_stamp, ".ply")
        export_records_to_csv(
            csv_path=str(design_axes_csv),
            fieldnames=[
                "axis_id",
                "entity_type",
                "layer",
                "length",
                "start_x",
                "start_y",
                "start_z",
                "end_x",
                "end_y",
                "end_z",
            ],
            records=axis_records_from_dxf,
        )
        export_records_to_csv(
            csv_path=str(design_nodes_csv),
            fieldnames=[
                "node_id",
                "node_role",
                "axis_ids",
                "axis_count",
                "axis_pairs",
                "entity_type",
                "layer",
                "x",
                "y",
                "z",
                "distance",
                "max_distance",
                "is_intersection_expected",
                "solve_method",
                "member_count",
                "on_segment_ratio",
            ],
            records=node_records_from_dxf,
        )
        write_node_ply(node_records_from_dxf, str(node_ply_path))

        print(
            f"DXF 直采样完成: axes={len(axis_records_from_dxf)}, nodes={len(node_records_from_dxf)}, sampled_points={len(sampled_points)}"
        )
        print(f"DXF 采样 PLY: {sampled_ply_path}")
        print(f"DXF 节点 PLY: {node_ply_path}")
        print(f"DXF 轴线 CSV: {design_axes_csv}")
        print(f"DXF 节点 CSV: {design_nodes_csv}")

        if args.show_open3d:
            print("正在打开 DXF 采样点云可视化窗口...")
            if node_records_from_dxf:
                visualize_sampled_points_and_nodes_open3d(
                    sampled_points=sampled_points,
                    node_records=node_records_from_dxf,
                    window_name="DXF Sampled Points and Junction Nodes",
                )
            else:
                visualize_points_open3d(sampled_points, window_name="DXF Sampled Line Cloud")

    if args.axis_ply and args.expected_nodes_ply and args.measured_cloud:
        compare_output_dir = str(run_output_dir)
        compare_axis_node_plys_with_measured_cloud(
            axis_ply=args.axis_ply,
            expected_nodes_ply=args.expected_nodes_ply,
            measured_cloud=args.measured_cloud,
            output_dir=compare_output_dir,
            show_open3d=args.compare_show_open3d,
            measured_downsample_step=max(1, int(args.measured_downsample_step)),
        )

    if args.dxf_path and args.measured_ply:
        if node_records_from_dxf is None:
            raise RuntimeError("DXF 节点提取失败，无法进行偏差比较")

        deviation_csv = args.output_deviation or str(run_output_dir / _default_run_name(args.report_name, run_stamp, ".csv"))
        crosswalk_csv = args.output_crosswalk or str(run_output_dir / _default_run_name(f"{args.report_name}_crosswalk", run_stamp, ".csv"))
        output_json = args.output_json or str(run_output_dir / _default_run_name(args.report_name, run_stamp, ".json"))

        deviation_records = compute_node_deviations_from_records(
            node_records=node_records_from_dxf,
            measured_ply=args.measured_ply,
            output_csv=deviation_csv,
            output_crosswalk_csv=crosswalk_csv,
            output_json=output_json,
            max_match_distance=args.node_match_distance,
            summary_extra={
                "run_stamp": run_stamp,
                "run_output_dir": str(run_output_dir),
                "design_source": "dxf_direct",
                "dxf_path": str(args.dxf_path),
                "sampled_dxf_ply": str(sampled_ply_path) if sampled_ply_path is not None else None,
            },
        )

        summary = summarize_node_deviation_records(deviation_records)
        print(
            f"节点偏差比较完成: matched={summary['matched_count']}/{summary['total_count']}, "
            f"unmatched={summary['unmatched_count']}"
        )
        if summary.get("matched_count", 0):
            print(
                f"偏差统计: mean={summary.get('mean_deviation_3d', 0.0):.6f}, "
                f"max={summary.get('max_deviation_3d', 0.0):.6f}, "
                f"min={summary.get('min_deviation_3d', 0.0):.6f}, "
                f"rmse={summary.get('rmse_deviation_3d', 0.0):.6f}"
            )
        print(f"偏差 CSV: {deviation_csv}")
        print(f"交叉对照 CSV: {crosswalk_csv}")
        print(f"偏差 JSON: {output_json}")

    elif args.design_nodes and args.measured_ply:
        deviation_csv = args.output_deviation or str(run_output_dir / _default_run_name(args.report_name, run_stamp, ".csv"))
        crosswalk_csv = args.output_crosswalk or str(run_output_dir / _default_run_name(f"{args.report_name}_crosswalk", run_stamp, ".csv"))
        output_json = args.output_json or str(run_output_dir / _default_run_name(args.report_name, run_stamp, ".json"))

        deviation_records = compute_node_deviations(
            design_nodes_csv=args.design_nodes,
            measured_ply=args.measured_ply,
            output_csv=deviation_csv,
            output_crosswalk_csv=crosswalk_csv,
            output_json=output_json,
            max_match_distance=args.node_match_distance,
        )

        summary = summarize_node_deviation_records(deviation_records)
        print(
            f"节点偏差比较完成: matched={summary['matched_count']}/{summary['total_count']}, "
            f"unmatched={summary['unmatched_count']}"
        )
        if summary.get("matched_count", 0):
            print(
                f"偏差统计: mean={summary.get('mean_deviation_3d', 0.0):.6f}, "
                f"max={summary.get('max_deviation_3d', 0.0):.6f}, "
                f"min={summary.get('min_deviation_3d', 0.0):.6f}, "
                f"rmse={summary.get('rmse_deviation_3d', 0.0):.6f}"
            )
        print(f"偏差 CSV: {deviation_csv}")
        print(f"交叉对照 CSV: {crosswalk_csv}")
        print(f"偏差 JSON: {output_json}")


if __name__ == "__main__":
    main()
