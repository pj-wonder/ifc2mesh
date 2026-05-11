from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import open3d as o3d
import numpy as np

from design_nodes import line2point as mod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive Open3D viewer for design-vs-measured node deviation")
    parser.add_argument(
        "--report-json",
        default=r"d:/南京北站点云/南京北数据处理相关代码/ifc2mesh/axis_node_exaction/node_fit_results/run_20260428_example_output_scaled/node_deviation_report_scaled.json",
        help="Path to the deviation report JSON",
    )
    parser.add_argument(
        "--design-scale",
        type=float,
        default=1.0,
        help="Extra scale to apply to design geometry before display",
    )
    parser.add_argument(
        "--show-axes",
        action="store_true",
        help="Also display the design axis lines",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=1600,
        help="Open3D window width",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=1000,
        help="Open3D window height",
    )
    return parser.parse_args()


def build_scene(report: Dict[str, Any], design_scale: float, show_axes: bool) -> List[Any]:
    dxf_path = report["design_dxf"]
    measured_nodes_ply = report["measured_nodes_ply"]
    auto_register = report.get("auto_register", {})
    junction_min_axis_count = int(report.get("junction_min_axis_count", 2))

    axis_records, node_records = mod.extract_axis_data(dxf_path)
    axis_records, node_records = mod.scale_axis_node_records(axis_records, node_records, design_scale)
    node_records = mod.filter_junction_nodes(node_records, min_axis_count=junction_min_axis_count)

    transform_4x4 = auto_register.get("transform_4x4")
    if auto_register.get("enabled") and transform_4x4:
        axis_records, node_records = mod._apply_transform_to_axis_node_records(
            axis_records,
            node_records,
            np.asarray(transform_4x4, dtype=float),
        )

    measured_points = mod.load_measured_node_points(measured_nodes_ply)
    deviation_records = mod.compare_design_nodes_to_measured_points(node_records, measured_points)

    geometries: List[Any] = []

    if show_axes:
        axis_points: List[mod.Point3D] = []
        axis_lines: List[List[int]] = []
        axis_colors: List[List[float]] = []
        for axis in axis_records:
            start = (float(axis["start_x"]), float(axis["start_y"]), float(axis["start_z"]))
            end = (float(axis["end_x"]), float(axis["end_y"]), float(axis["end_z"]))
            idx = len(axis_points)
            axis_points.extend([start, end])
            axis_lines.append([idx, idx + 1])
            axis_colors.append([0.72, 0.72, 0.72])

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(axis_points)
        line_set.lines = o3d.utility.Vector2iVector(axis_lines)
        line_set.colors = o3d.utility.Vector3dVector(axis_colors)
        geometries.append(line_set)

    matched_design: List[mod.Point3D] = []
    matched_design_colors: List[List[float]] = []
    matched_measured: List[mod.Point3D] = []
    unmatched: List[mod.Point3D] = []
    vector_points: List[mod.Point3D] = []
    vector_indices: List[List[int]] = []
    vector_colors: List[List[float]] = []

    matched_devs = [float(r["deviation_3d"]) for r in deviation_records if bool(r.get("matched", False))]
    max_dev = max(matched_devs) if matched_devs else 1.0
    if max_dev <= 0:
        max_dev = 1.0

    def _axis_count_color(axis_count: int) -> List[float]:
        capped = max(1, min(int(axis_count), 6))
        t = (capped - 1) / 5.0
        return [t, 0.25 + 0.55 * (1.0 - t), 1.0 - 0.75 * t]

    def _axis_count_from_node(node: Dict[str, Any]) -> int:
        axis_ids = node.get("axis_ids", [])
        if isinstance(axis_ids, str):
            return len([item for item in axis_ids.split(";") if item])
        return len(axis_ids)

    design_point_cache: Dict[str, mod.Point3D] = {}
    design_color_cache: Dict[str, List[float]] = {}
    for node in node_records:
        node_id = str(node["node_id"])
        design_point = (float(node["x"]), float(node["y"]), float(node["z"]))
        design_point_cache[node_id] = design_point
        design_color_cache[node_id] = _axis_count_color(_axis_count_from_node(node))
        matched_design.append(design_point)
        matched_design_colors.append(design_color_cache[node_id])

    for record in deviation_records:
        if not bool(record.get("matched", False)):
            unmatched.append((float(record["measured_x"]), float(record["measured_y"]), float(record["measured_z"])))
            continue

        measured_point = (
            float(record["measured_x"]),
            float(record["measured_y"]),
            float(record["measured_z"]),
        )
        design_node_id = str(record.get("node_id", ""))
        design_point = design_point_cache.get(design_node_id)
        if design_point is None:
            continue
        start_index = len(vector_points)
        vector_points.extend([design_point, measured_point])
        vector_indices.append([start_index, start_index + 1])
        ratio = max(0.0, min(1.0, float(record["deviation_3d"]) / max_dev))
        vector_colors.append([ratio, 1.0 - ratio, 0.15])
        matched_measured.append(measured_point)

    if vector_points:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(vector_points)
        line_set.lines = o3d.utility.Vector2iVector(vector_indices)
        line_set.colors = o3d.utility.Vector3dVector(vector_colors)
        geometries.append(line_set)

    if matched_design:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(matched_design)
        pcd.colors = o3d.utility.Vector3dVector(matched_design_colors)
        geometries.append(pcd)

    if matched_measured:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(matched_measured)
        pcd.paint_uniform_color([0.18, 0.45, 0.95])
        geometries.append(pcd)

    if unmatched:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(unmatched)
        pcd.paint_uniform_color([0.95, 0.72, 0.18])
        geometries.append(pcd)

    return geometries


def main() -> None:
    args = parse_args()
    report_path = Path(args.report_json)
    if not report_path.exists():
        raise FileNotFoundError(f"Report JSON not found: {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    geometries = build_scene(report, design_scale=args.design_scale, show_axes=args.show_axes)

    if not geometries:
        raise RuntimeError("No geometry to display")

    print("Open3D interactive viewer ready. Mouse: rotate/pan/zoom. Close the window to exit.")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Open3D Node Deviation Viewer",
        width=args.window_width,
        height=args.window_height,
    )


if __name__ == "__main__":
    main()
