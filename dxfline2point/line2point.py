from __future__ import annotations

import math
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import csv

import ezdxf
import numpy as np


Point3D = Tuple[float, float, float]


def _segment_length(start_point: Point3D, end_point: Point3D) -> float:
    return math.dist(start_point, end_point)


def _normalize_point(values: Sequence[Any]) -> Point3D:
    coords = [float(value) for value in values]
    if len(coords) == 2:
        coords.append(0.0)
    return coords[0], coords[1], coords[2]


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
        vertices = []
        for point in entity.get_points("xy"):
            x, y = point[0], point[1]
            vertices.append((float(x), float(y), 0.0))
        return [(vertices[index], vertices[index + 1]) for index in range(len(vertices) - 1)]

    if entity_type == "POLYLINE":
        vertices = []
        for vertex in entity.vertices:
            location = vertex.dxf.location
            vertices.append(_normalize_point((location.x, location.y, location.z)))
        return [(vertices[index], vertices[index + 1]) for index in range(len(vertices) - 1)]

    return []


def _closest_points_between_segments(
    p1: Point3D,
    q1: Point3D,
    p2: Point3D,
    q2: Point3D,
) -> Tuple[Point3D, Point3D, Point3D, float, float, float]:
    d1 = (q1[0] - p1[0], q1[1] - p1[1], q1[2] - p1[2])
    d2 = (q2[0] - p2[0], q2[1] - p2[1], q2[2] - p2[2])
    w0 = (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])

    a = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2]
    b = d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]
    c = d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2]
    d = d1[0] * w0[0] + d1[1] * w0[1] + d1[2] * w0[2]
    e = d2[0] * w0[0] + d2[1] * w0[1] + d2[2] * w0[2]
    den = a * c - b * b

    if abs(den) < 1e-12:
        t1 = 0.0
        t2 = 0.0
    else:
        t1 = (b * e - c * d) / den
        t2 = (a * e - b * d) / den

    q1p = (p1[0] + t1 * d1[0], p1[1] + t1 * d1[1], p1[2] + t1 * d1[2])
    q2p = (p2[0] + t2 * d2[0], p2[1] + t2 * d2[1], p2[2] + t2 * d2[2])
    node = ((q1p[0] + q2p[0]) * 0.5, (q1p[1] + q2p[1]) * 0.5, (q1p[2] + q2p[2]) * 0.5)
    dist = math.dist(q1p, q2p)
    return q1p, q2p, node, dist, t1, t2


def _node_key(point: Point3D, tolerance: float) -> Tuple[int, int, int]:
    scale = 1.0 / tolerance if tolerance > 0 else 1000.0
    return (round(point[0] * scale), round(point[1] * scale), round(point[2] * scale))


def _normalize_vector(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return vector.copy()
    return vector / norm


def _point_on_segment(point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray, margin: float) -> bool:
    seg = seg_end - seg_start
    length = float(np.linalg.norm(seg))
    if length < 1e-10:
        return False
    direction = seg / length
    s = float(np.dot(point - seg_start, direction))
    return (-margin <= s <= length + margin)


def _cluster_points(points: np.ndarray, radius: float) -> List[List[int]]:
    if len(points) == 0:
        return []
    if radius <= 0:
        return [[i] for i in range(len(points))]

    visited = np.zeros(len(points), dtype=bool)
    clusters: List[List[int]] = []
    for i in range(len(points)):
        if visited[i]:
            continue
        queue = [i]
        visited[i] = True
        current = [i]
        while queue:
            cur = queue.pop()
            dist = np.linalg.norm(points - points[cur], axis=1)
            nbr = np.where((dist <= radius) & (~visited))[0]
            for j in nbr.tolist():
                visited[j] = True
                queue.append(j)
                current.append(j)
        clusters.append(current)
    return clusters


def _solve_global_pseudointersection(anchors: np.ndarray, directions: np.ndarray) -> Tuple[np.ndarray, float, float]:
    identity = np.eye(3, dtype=float)
    system_matrix = np.zeros((3, 3), dtype=float)
    rhs = np.zeros(3, dtype=float)

    for anchor, direction in zip(anchors, directions):
        direction = _normalize_vector(np.asarray(direction, dtype=float))
        projector = identity - np.outer(direction, direction)
        system_matrix += projector
        rhs += projector @ np.asarray(anchor, dtype=float)

    center = np.linalg.pinv(system_matrix) @ rhs
    distances = [
        float(np.linalg.norm(np.cross(center - anchor, _normalize_vector(np.asarray(direction, dtype=float)))))
        for anchor, direction in zip(anchors, directions)
    ]
    return center, float(np.mean(distances)) if distances else 0.0, float(np.max(distances)) if distances else 0.0


def _point_to_homogeneous(point: Point3D) -> np.ndarray:
    return np.array([float(point[0]), float(point[1]), float(point[2]), 1.0], dtype=float)


def _apply_transform_to_point(point: Point3D, transform: np.ndarray) -> Point3D:
    transformed = transform @ _point_to_homogeneous(point)
    return float(transformed[0]), float(transformed[1]), float(transformed[2])


def _estimate_rigid_transform(source_points: np.ndarray, target_points: np.ndarray) -> Tuple[np.ndarray, float]:
    if source_points.shape != target_points.shape:
        raise ValueError("source_points and target_points must have the same shape")
    if source_points.shape[0] < 3:
        identity = np.eye(4, dtype=float)
        if source_points.shape[0] == 0:
            return identity, 0.0
        residual = float(np.sqrt(np.mean(np.sum((source_points - target_points) ** 2, axis=1))))
        return identity, residual

    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid

    h_matrix = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(h_matrix)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T

    translation = target_centroid - rotation @ source_centroid
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation

    aligned = (rotation @ source_points.T).T + translation
    residual = float(np.sqrt(np.mean(np.sum((aligned - target_points) ** 2, axis=1))))
    return transform, residual


def _apply_transform_to_axis_node_records(
    axis_records: List[Dict[str, Any]],
    node_records: List[Dict[str, Any]],
    transform: np.ndarray,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if np.allclose(transform, np.eye(4, dtype=float)):
        return axis_records, node_records

    transformed_axes: List[Dict[str, Any]] = []
    for axis in axis_records:
        start_point = _apply_transform_to_point(
            (float(axis["start_x"]), float(axis["start_y"]), float(axis["start_z"])),
            transform,
        )
        end_point = _apply_transform_to_point(
            (float(axis["end_x"]), float(axis["end_y"]), float(axis["end_z"])),
            transform,
        )
        transformed_axes.append(
            {
                **axis,
                "start_x": start_point[0],
                "start_y": start_point[1],
                "start_z": start_point[2],
                "end_x": end_point[0],
                "end_y": end_point[1],
                "end_z": end_point[2],
                "length": _segment_length(start_point, end_point),
            }
        )

    transformed_nodes: List[Dict[str, Any]] = []
    for node in node_records:
        transformed_point = _apply_transform_to_point(
            (float(node["x"]), float(node["y"]), float(node["z"])),
            transform,
        )
        transformed_nodes.append(
            {
                **node,
                "x": transformed_point[0],
                "y": transformed_point[1],
                "z": transformed_point[2],
            }
        )

    return transformed_axes, transformed_nodes


def extract_axis_data(
    dxf_path: str,
    min_axis_length: float = 1000.0,
    node_distance_threshold: float = 5.0,
    node_merge_tolerance: float = 1.0,
    segment_margin: Optional[float] = None,
    node_grouping_radius: Optional[float] = None,
    multi_min_members: int = 3,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    axis_records: List[Dict[str, Any]] = []
    node_records: List[Dict[str, Any]] = []

    segment_candidates: List[Dict[str, Any]] = []
    for entity in msp:
        segments = _extract_axis_segments(entity)
        if not segments:
            continue

        for start_point, end_point in segments:
            length = _segment_length(start_point, end_point)
            if length < min_axis_length:
                continue

            segment_candidates.append(
                {
                    "entity_type": entity.dxftype(),
                    "layer": getattr(entity.dxf, "layer", ""),
                    "start": start_point,
                    "end": end_point,
                    "length": length,
                }
            )

    for index, segment in enumerate(segment_candidates, start=1):
        axis_id = f"axis_{index:04d}"
        start_point = segment["start"]
        end_point = segment["end"]

        axis_records.append(
            {
                "axis_id": axis_id,
                "entity_type": segment["entity_type"],
                "layer": segment["layer"],
                "length": segment["length"],
                "start_x": start_point[0],
                "start_y": start_point[1],
                "start_z": start_point[2],
                "end_x": end_point[0],
                "end_y": end_point[1],
                "end_z": end_point[2],
            }
        )

    axis_map = {record["axis_id"]: record for record in axis_records}
    pair_candidates: List[Dict[str, Any]] = []
    for i, left in enumerate(axis_records):
        left_start = np.array([left["start_x"], left["start_y"], left["start_z"]], dtype=float)
        left_end = np.array([left["end_x"], left["end_y"], left["end_z"]], dtype=float)
        left_len = float(np.linalg.norm(left_end - left_start))
        left_margin = float(segment_margin) if segment_margin is not None else max(0.0, left_len * 0.05)

        for j in range(i + 1, len(axis_records)):
            right = axis_records[j]
            right_start = np.array([right["start_x"], right["start_y"], right["start_z"]], dtype=float)
            right_end = np.array([right["end_x"], right["end_y"], right["end_z"]], dtype=float)
            right_len = float(np.linalg.norm(right_end - right_start))
            right_margin = float(segment_margin) if segment_margin is not None else max(0.0, right_len * 0.05)

            _, _, node, distance, _, _ = _closest_points_between_segments(
                tuple(left_start), tuple(left_end), tuple(right_start), tuple(right_end)
            )
            if distance > node_distance_threshold:
                continue

            node_np = np.array([node[0], node[1], node[2]], dtype=float)
            on_left = _point_on_segment(node_np, left_start, left_end, margin=left_margin)
            on_right = _point_on_segment(node_np, right_start, right_end, margin=right_margin)
            expected = bool(on_left and on_right)

            pair_candidates.append(
                {
                    "left": left,
                    "right": right,
                    "node": node_np,
                    "distance": float(distance),
                    "on_left": bool(on_left),
                    "on_right": bool(on_right),
                    "is_expected": expected,
                }
            )

    grouping_radius = float(node_grouping_radius) if node_grouping_radius is not None else float(node_merge_tolerance)
    expected_idx = [idx for idx, r in enumerate(pair_candidates) if bool(r["is_expected"])]
    if expected_idx:
        exp_pts = np.array([pair_candidates[i]["node"] for i in expected_idx], dtype=float)
        clusters = _cluster_points(exp_pts, radius=grouping_radius)
    else:
        clusters = []

    suppressed_pairs = set()
    final_counter = 0

    for cl in clusters:
        cand_indices = [expected_idx[k] for k in cl]
        members = sorted(
            {
                str(pair_candidates[idx]["left"]["axis_id"]) for idx in cand_indices
            }
            |
            {
                str(pair_candidates[idx]["right"]["axis_id"]) for idx in cand_indices
            }
        )
        if len(members) < max(3, int(multi_min_members)):
            continue

        anchors = []
        directions = []
        for axis_id in members:
            axis = axis_map.get(axis_id)
            if axis is None:
                continue
            start = np.array([float(axis["start_x"]), float(axis["start_y"]), float(axis["start_z"])], dtype=float)
            end = np.array([float(axis["end_x"]), float(axis["end_y"]), float(axis["end_z"])], dtype=float)
            direction = end - start
            if float(np.linalg.norm(direction)) < 1e-12:
                continue
            anchors.append(start)
            directions.append(direction)

        if len(anchors) < 3:
            continue

        center, mean_distance, max_distance = _solve_global_pseudointersection(
            np.asarray(anchors), np.asarray(directions)
        )

        on_flags = []
        for axis_id in members:
            axis = axis_map.get(axis_id)
            if axis is None:
                on_flags.append(False)
                continue
            start = np.array([float(axis["start_x"]), float(axis["start_y"]), float(axis["start_z"])], dtype=float)
            end = np.array([float(axis["end_x"]), float(axis["end_y"]), float(axis["end_z"])], dtype=float)
            axis_len = float(np.linalg.norm(end - start))
            margin = float(segment_margin) if segment_margin is not None else max(0.0, axis_len * 0.05)
            on_flags.append(_point_on_segment(center, start, end, margin=margin))

        on_ratio = float(np.mean(on_flags)) if on_flags else 0.0
        expected_flag = bool((mean_distance <= float(node_distance_threshold)) and (on_ratio >= 0.6))

        axis_pairs = sorted({
            f"{pair_candidates[idx]['left']['axis_id']}|{pair_candidates[idx]['right']['axis_id']}"
            for idx in cand_indices
        })
        entity_types = sorted({
            f"{pair_candidates[idx]['left']['entity_type']}|{pair_candidates[idx]['right']['entity_type']}"
            for idx in cand_indices
        })
        layers = sorted({
            f"{pair_candidates[idx]['left']['layer']}|{pair_candidates[idx]['right']['layer']}"
            for idx in cand_indices
        })

        final_counter += 1
        node_records.append(
            {
                "node_id": f"node_mls_{final_counter:04d}",
                "node_role": "intersection",
                "axis_ids": ";".join(members),
                "axis_count": int(len(members)),
                "axis_pairs": ";".join(axis_pairs),
                "entity_type": ";".join(entity_types),
                "layer": ";".join(layers),
                "x": float(center[0]),
                "y": float(center[1]),
                "z": float(center[2]),
                "distance": float(mean_distance),
                "max_distance": float(max_distance),
                "is_intersection_expected": expected_flag,
                "solve_method": "global_least_squares_pinv",
                "member_count": int(len(members)),
                "on_segment_ratio": float(on_ratio),
            }
        )
        for idx in cand_indices:
            suppressed_pairs.add(idx)

    for idx, cand in enumerate(pair_candidates):
        if idx in suppressed_pairs:
            continue
        left = cand["left"]
        right = cand["right"]
        axis_pair = f"{left['axis_id']}|{right['axis_id']}"
        final_counter += 1
        node_records.append(
            {
                "node_id": f"node_pair_{final_counter:04d}",
                "node_role": "intersection",
                "axis_ids": f"{left['axis_id']};{right['axis_id']}",
                "axis_count": 2,
                "axis_pairs": axis_pair,
                "entity_type": f"{left['entity_type']}|{right['entity_type']}",
                "layer": f"{left['layer']}|{right['layer']}",
                "x": float(cand["node"][0]),
                "y": float(cand["node"][1]),
                "z": float(cand["node"][2]),
                "distance": float(cand["distance"]),
                "max_distance": float(cand["distance"]),
                "is_intersection_expected": bool(cand["is_expected"]),
                "solve_method": "pairwise_midpoint",
                "member_count": 2,
                "on_segment_ratio": 1.0 if (cand["on_left"] and cand["on_right"]) else 0.5 if (cand["on_left"] or cand["on_right"]) else 0.0,
            }
        )

    return axis_records, node_records


def scale_axis_node_records(
    axis_records: List[Dict[str, Any]],
    node_records: List[Dict[str, Any]],
    scale: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if abs(float(scale) - 1.0) < 1e-12:
        return axis_records, node_records

    scaled_axes: List[Dict[str, Any]] = []
    for axis in axis_records:
        scaled_axes.append(
            {
                **axis,
                "length": float(axis["length"]) * float(scale),
                "start_x": float(axis["start_x"]) * float(scale),
                "start_y": float(axis["start_y"]) * float(scale),
                "start_z": float(axis["start_z"]) * float(scale),
                "end_x": float(axis["end_x"]) * float(scale),
                "end_y": float(axis["end_y"]) * float(scale),
                "end_z": float(axis["end_z"]) * float(scale),
            }
        )

    scaled_nodes: List[Dict[str, Any]] = []
    for node in node_records:
        scaled_nodes.append(
            {
                **node,
                "distance": float(node["distance"]) * float(scale),
                "x": float(node["x"]) * float(scale),
                "y": float(node["y"]) * float(scale),
                "z": float(node["z"]) * float(scale),
            }
        )

    return scaled_axes, scaled_nodes


def filter_junction_nodes(
    node_records: List[Dict[str, Any]],
    min_axis_count: int = 2,
) -> List[Dict[str, Any]]:
    if min_axis_count <= 1:
        return node_records

    filtered: List[Dict[str, Any]] = []
    for node in node_records:
        axis_count = node.get("axis_count", 0)
        try:
            axis_count_value = int(axis_count)
        except (TypeError, ValueError):
            axis_count_value = 0
        if axis_count_value >= int(min_axis_count):
            filtered.append(node)
    return filtered


def export_records_to_csv(csv_path: str, fieldnames: List[str], records: List[Dict[str, Any]]) -> None:
    output_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"CSV 已保存: {output_path} (records={len(records)})")


def _read_point_cloud_safe(ply_path: Path) -> "o3d.geometry.PointCloud":
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("open3d 未安装，无法读取 PLY。请先执行: pip install open3d") from exc

    try:
        return o3d.io.read_point_cloud(str(ply_path))
    except UnicodeDecodeError:
        ascii_name = f"ply_input_{ply_path.stem}.ply"
        with tempfile.TemporaryDirectory(prefix="ply_ascii_") as tmp_dir:
            ascii_path = Path(tmp_dir) / ascii_name
            shutil.copyfile(ply_path, ascii_path)
            return o3d.io.read_point_cloud(str(ascii_path))


def load_measured_node_points(ply_path: str) -> List[Point3D]:
    path = Path(ply_path)
    if not path.exists():
        raise FileNotFoundError(f"测量节点 PLY 不存在: {path}")

    pcd = _read_point_cloud_safe(path)
    points = np.asarray(pcd.points, dtype=float)
    return [(float(p[0]), float(p[1]), float(p[2])) for p in points]


def load_measured_cloud_stats(ply_path: str) -> Dict[str, Any]:
    path = Path(ply_path)
    if not path.exists():
        raise FileNotFoundError(f"PLY 不存在: {path}")

    pcd = _read_point_cloud_safe(path)
    points = np.asarray(pcd.points, dtype=float)
    stats: Dict[str, Any] = {
        "path": str(path),
        "point_count": int(points.shape[0]),
    }
    if points.size:
        stats.update(
            {
                "bbox_min_x": float(np.min(points[:, 0])),
                "bbox_min_y": float(np.min(points[:, 1])),
                "bbox_min_z": float(np.min(points[:, 2])),
                "bbox_max_x": float(np.max(points[:, 0])),
                "bbox_max_y": float(np.max(points[:, 1])),
                "bbox_max_z": float(np.max(points[:, 2])),
            }
        )
    return stats


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
                "node_id": str(design_node["node_id"]),
                "node_role": str(design_node.get("node_role", "")),
                "axis_count": int(len(axis_ids)),
                "axis_ids": ";".join(str(item) for item in axis_ids),
                "design_index": int(design_idx + 1),
                "design_x": float(design_point[0]),
                "design_y": float(design_point[1]),
                "design_z": float(design_point[2]),
                "dx": float(delta[0]),
                "dy": float(delta[1]),
                "dz": float(delta[2]),
                "deviation_3d": float(np.linalg.norm(delta)),
                "matched": True,
            }
        )

    return rows


def summarize_node_deviation_records(deviation_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    matched_records = [r for r in deviation_records if bool(r.get("matched", False))]
    deviations = [float(r["deviation_3d"]) for r in matched_records]
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
                "design_index": deviation.get("design_index", ""),
                "design_x": deviation.get("design_x", ""),
                "design_y": deviation.get("design_y", ""),
                "design_z": deviation.get("design_z", ""),
                "matched": bool(deviation.get("matched", False)),
                "dx": deviation.get("dx", ""),
                "dy": deviation.get("dy", ""),
                "dz": deviation.get("dz", ""),
                "deviation_3d": deviation.get("deviation_3d", ""),
            }
        )

    return rows


def visualize_axis_and_nodes(
    axis_records: List[Dict[str, Any]],
    node_records: List[Dict[str, Any]],
    save_fig_path: Optional[str] = None,
    show_labels: bool = False,
    show_figure: bool = True,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib 未安装，无法可视化。请先执行: pip install matplotlib") from exc

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    all_points: List[Point3D] = []

    for axis in axis_records:
        start = (axis["start_x"], axis["start_y"], axis["start_z"])
        end = (axis["end_x"], axis["end_y"], axis["end_z"])
        all_points.extend([start, end])

        xs = [start[0], end[0]]
        ys = [start[1], end[1]]
        zs = [start[2], end[2]]
        ax.plot(xs, ys, zs, color="crimson", linewidth=2.0, alpha=0.9)
        ax.scatter(xs, ys, zs, color="gold", s=24)

        if show_labels:
            ax.text(start[0], start[1], start[2], f"{axis['axis_id']}-S", fontsize=7)
            ax.text(end[0], end[1], end[2], f"{axis['axis_id']}-E", fontsize=7)

    if node_records:
        xs = [node["x"] for node in node_records]
        ys = [node["y"] for node in node_records]
        zs = [node["z"] for node in node_records]
        all_points.extend([(node["x"], node["y"], node["z"]) for node in node_records])
        ax.scatter(xs, ys, zs, color="royalblue", s=28, marker="o", alpha=0.85)

        if show_labels:
            for node in node_records:
                ax.text(node["x"], node["y"], node["z"], node["node_id"], fontsize=7)

    if all_points:
        min_x = min(point[0] for point in all_points)
        max_x = max(point[0] for point in all_points)
        min_y = min(point[1] for point in all_points)
        max_y = max(point[1] for point in all_points)
        min_z = min(point[2] for point in all_points)
        max_z = max(point[2] for point in all_points)

        pad = 0.05
        dx = (max_x - min_x) or 1.0
        dy = (max_y - min_y) or 1.0
        dz = (max_z - min_z) or 1.0
        ax.set_xlim(min_x - dx * pad, max_x + dx * pad)
        ax.set_ylim(min_y - dy * pad, max_y + dy * pad)
        ax.set_zlim(min_z - dz * pad, max_z + dz * pad)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("DXF Long-Axis Intersection Nodes")

    axis_handle, = ax.plot([], [], [], color="crimson", linewidth=2.0, label="Axis")
    node_handle = ax.scatter([], [], [], color="royalblue", s=28, marker="o", label="Node")
    ax.legend(handles=[axis_handle, node_handle], loc="best")

    if save_fig_path:
        output_path = Path(save_fig_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"图像已保存: {output_path}")

    if show_figure:
        plt.show()


def visualize_axis_and_nodes_open3d(
    axis_records: List[Dict[str, Any]],
    node_records: List[Dict[str, Any]],
    node_radius: Optional[float] = None,
) -> None:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("open3d 未安装，无法可视化。请先执行: pip install open3d") from exc

    geometries = []
    all_points: List[Point3D] = []

    if axis_records:
        line_points: List[Point3D] = []
        line_indices: List[List[int]] = []
        line_colors: List[List[float]] = []

        for axis in axis_records:
            start = (axis["start_x"], axis["start_y"], axis["start_z"])
            end = (axis["end_x"], axis["end_y"], axis["end_z"])
            idx = len(line_points)
            line_points.extend([start, end])
            line_indices.append([idx, idx + 1])
            line_colors.append([0.85, 0.2, 0.3])
            all_points.extend([start, end])

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_points)
        line_set.lines = o3d.utility.Vector2iVector(line_indices)
        line_set.colors = o3d.utility.Vector3dVector(line_colors)
        geometries.append(line_set)

    if node_records:
        node_points: List[Point3D] = []
        for node in node_records:
            p = (float(node["x"]), float(node["y"]), float(node["z"]))
            node_points.append(p)
        all_points.extend(node_points)

        if node_radius is None:
            if all_points:
                min_x = min(p[0] for p in all_points)
                max_x = max(p[0] for p in all_points)
                min_y = min(p[1] for p in all_points)
                max_y = max(p[1] for p in all_points)
                min_z = min(p[2] for p in all_points)
                max_z = max(p[2] for p in all_points)
                diag = math.dist((min_x, min_y, min_z), (max_x, max_y, max_z))
                node_radius = max(0.2, min(200.0, diag * 0.0005))
            else:
                node_radius = 1.0

        for p in node_points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=float(node_radius))
            sphere.translate(p)
            sphere.paint_uniform_color([0.2, 0.4, 0.9])
            geometries.append(sphere)

    if not geometries:
        print("无可视化对象：axis_records 和 node_records 都为空。")
        return

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Open3D Axis and Nodes",
        width=1280,
        height=820,
    )


def visualize_node_deviation_open3d(
    axis_records: List[Dict[str, Any]],
    node_records: List[Dict[str, Any]],
    deviation_records: List[Dict[str, Any]],
    show_axes: bool = True,
    node_radius: Optional[float] = None,
    line_radius_hint: Optional[float] = None,
) -> None:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("open3d 未安装，无法可视化。请先执行: pip install open3d") from exc

    geometries = []
    all_points: List[Point3D] = []

    if show_axes and axis_records:
        line_points: List[Point3D] = []
        line_indices: List[List[int]] = []
        line_colors: List[List[float]] = []

        for axis in axis_records:
            start = (axis["start_x"], axis["start_y"], axis["start_z"])
            end = (axis["end_x"], axis["end_y"], axis["end_z"])
            idx = len(line_points)
            line_points.extend([start, end])
            line_indices.append([idx, idx + 1])
            line_colors.append([0.72, 0.72, 0.72])
            all_points.extend([start, end])

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_points)
        line_set.lines = o3d.utility.Vector2iVector(line_indices)
        line_set.colors = o3d.utility.Vector3dVector(line_colors)
        geometries.append(line_set)

    if node_records:
        design_points: List[Point3D] = []
        for node in node_records:
            p = (float(node["x"]), float(node["y"]), float(node["z"]))
            design_points.append(p)
        all_points.extend(design_points)

        matched_lookup = {
            str(record.get("node_id", "")): record
            for record in deviation_records
            if bool(record.get("matched", False))
        }

        matched_design_points: List[Point3D] = []
        matched_measured_points: List[Point3D] = []
        matched_line_points: List[Point3D] = []
        matched_line_indices: List[List[int]] = []
        matched_line_colors: List[List[float]] = []
        unmatched_points: List[Point3D] = []

        matched_deviations = [float(r["deviation_3d"]) for r in deviation_records if bool(r.get("matched", False))]
        max_dev = max(matched_deviations) if matched_deviations else 1.0
        if max_dev <= 0:
            max_dev = 1.0

        def _deviation_color(distance: float) -> List[float]:
            t = max(0.0, min(1.0, float(distance) / max_dev))
            return [t, 1.0 - t, 0.15]

        for node in node_records:
            node_id = str(node["node_id"])
            record = matched_lookup.get(node_id)
            design_point = (float(node["x"]), float(node["y"]), float(node["z"]))
            if record is None:
                unmatched_points.append(design_point)
                continue

            measured_point = (
                float(record["measured_x"]),
                float(record["measured_y"]),
                float(record["measured_z"]),
            )

            design_idx = len(matched_line_points)
            matched_line_points.extend([design_point, measured_point])
            matched_line_indices.append([design_idx, design_idx + 1])
            matched_line_colors.append(_deviation_color(float(record["deviation_3d"])))

            matched_design_points.append(design_point)
            matched_measured_points.append(measured_point)
            all_points.extend([design_point, measured_point])

        if matched_line_points:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(matched_line_points)
            line_set.lines = o3d.utility.Vector2iVector(matched_line_indices)
            line_set.colors = o3d.utility.Vector3dVector(matched_line_colors)
            geometries.append(line_set)

        if matched_design_points:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(matched_design_points)
            pcd.paint_uniform_color([0.95, 0.35, 0.25])
            geometries.append(pcd)

        if matched_measured_points:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(matched_measured_points)
            pcd.paint_uniform_color([0.18, 0.45, 0.95])
            geometries.append(pcd)

        if unmatched_points:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(unmatched_points)
            pcd.paint_uniform_color([0.95, 0.72, 0.18])
            geometries.append(pcd)

    if not geometries:
        print("无可视化对象：axis_records、node_records 和 deviation_records 都为空。")
        return

    if all_points:
        min_x = min(p[0] for p in all_points)
        max_x = max(p[0] for p in all_points)
        min_y = min(p[1] for p in all_points)
        max_y = max(p[1] for p in all_points)
        min_z = min(p[2] for p in all_points)
        max_z = max(p[2] for p in all_points)
        diag = math.dist((min_x, min_y, min_z), (max_x, max_y, max_z))
        if node_radius is None:
            node_radius = max(0.02, min(2.0, diag * 0.004))
        if line_radius_hint is None:
            line_radius_hint = max(0.02, min(1.0, diag * 0.002))

        for _ in []:
            pass

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Open3D Node Deviation",
        width=1400,
        height=900,
    )


if __name__ == "__main__":
    default_dxf_path = Path(__file__).with_name("center_line.dxf")

    import argparse

    parser = argparse.ArgumentParser(description="Extract 3D axis segments and node coordinates from DXF.")
    parser.add_argument("dxf_path", nargs="?", default=str(default_dxf_path), help="Path to the DXF file")
    parser.add_argument("--min-axis-length", type=float, default=1000.0, help="Drop axis segments shorter than this length")
    parser.add_argument("--node-distance-threshold", type=float, default=5.0, help="Max distance between two long axes to treat as an intersection")
    parser.add_argument("--open3d", action="store_true", help="Use Open3D to visualize axis and nodes")
    parser.add_argument("--open3d-node-radius", type=float, default=None, help="Node sphere radius in Open3D")
    parser.add_argument("--save-fig", default=None, help="Path to save visualization image")
    parser.add_argument("--show-labels", action="store_true", help="Show axis/node labels")
    parser.add_argument("--no-show", action="store_true", help="Do not open figure window; useful for headless runs")
    parser.add_argument("--export-axes-csv", default=None, help="Export axis segments to CSV")
    parser.add_argument("--export-nodes-csv", default=None, help="Export node coordinates to CSV")
    parser.add_argument("--output-dir", default=None, help="Output directory for the generated deviation report")
    parser.add_argument("--report-name", default="node_deviation_report", help="Base name for generated report files")
    parser.add_argument("--measured-axes-ply", default=None, help="Measured axis PLY file (recorded in summary for traceability)")
    parser.add_argument("--measured-nodes-ply", default=None, help="Measured node PLY file for coordinate deviation comparison")
    parser.add_argument("--export-node-deviation-csv", default=None, help="Export design-vs-measured node deviation CSV")
    parser.add_argument("--export-node-crosswalk-csv", default=None, help="Export design-node crosswalk CSV with axis counts and matched measured nodes")
    parser.add_argument("--node-match-distance", type=float, default=999999.0, help="Max distance allowed for node matching")
    parser.add_argument("--open3d-deviation", action="store_true", help="Show deviation vectors in Open3D after node matching")
    parser.add_argument("--design-scale", type=float, default=1.0, help="Scale factor applied to design DXF coordinates before comparison")
    parser.add_argument("--no-auto-register", action="store_true", help="Disable automatic rigid registration between design and measured nodes")
    parser.add_argument("--junction-min-axis-count", type=int, default=2, help="Keep only design junction nodes with at least this many incident axes")
    args = parser.parse_args()

    axis_records, node_records = extract_axis_data(
        args.dxf_path,
        min_axis_length=args.min_axis_length,
        node_distance_threshold=args.node_distance_threshold,
    )

    axis_records, node_records = scale_axis_node_records(axis_records, node_records, args.design_scale)
    node_records_before_filter = node_records
    node_records = filter_junction_nodes(node_records, min_axis_count=args.junction_min_axis_count)

    print(
        f"成功提取了 {len(axis_records)} 条长轴线，{len(node_records)} 个交点节点！"
        f" 过滤阈值=min_axis_length={args.min_axis_length}, node_distance_threshold={args.node_distance_threshold}, design_scale={args.design_scale}, junction_min_axis_count={args.junction_min_axis_count}"
    )

    if args.export_axes_csv:
        export_records_to_csv(
            csv_path=args.export_axes_csv,
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
            records=axis_records,
        )

    if args.export_nodes_csv:
        export_records_to_csv(
            csv_path=args.export_nodes_csv,
            fieldnames=["node_id", "node_role", "axis_ids", "axis_pairs", "entity_type", "layer", "x", "y", "z", "distance"],
            records=node_records,
        )

    measured_axes_stats: Optional[Dict[str, Any]] = None
    measured_nodes_stats: Optional[Dict[str, Any]] = None
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is None and args.measured_nodes_ply:
        ifc2mesh_root = Path(__file__).resolve().parent.parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ifc2mesh_root / "result" / "node_deviation" / timestamp
        print(f"未指定 output_dir，自动使用时间戳目录: {output_dir}")
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.measured_axes_ply:
        measured_axes_stats = load_measured_cloud_stats(args.measured_axes_ply)
        print(
            f"实测轴线PLY: points={measured_axes_stats['point_count']}, "
            f"bbox_min=({measured_axes_stats.get('bbox_min_x', '')}, {measured_axes_stats.get('bbox_min_y', '')}, {measured_axes_stats.get('bbox_min_z', '')}), "
            f"bbox_max=({measured_axes_stats.get('bbox_max_x', '')}, {measured_axes_stats.get('bbox_max_y', '')}, {measured_axes_stats.get('bbox_max_z', '')})"
        )

    registration_summary: Dict[str, Any] = {
        "enabled": bool(args.measured_nodes_ply and not args.no_auto_register),
        "pair_count": 0,
        "rms_before": None,
        "rms_after": None,
        "transform_4x4": np.eye(4, dtype=float).tolist(),
    }

    if args.measured_nodes_ply:
        measured_points = load_measured_node_points(args.measured_nodes_ply)
        measured_nodes_stats = load_measured_cloud_stats(args.measured_nodes_ply)

        if not args.no_auto_register:
            pre_registration_records = compare_design_nodes_to_measured_points(
                node_records=node_records,
                measured_points=measured_points,
                max_match_distance=args.node_match_distance,
            )
            matched_design_points = np.asarray(
                [
                    [float(row["design_x"]), float(row["design_y"]), float(row["design_z"])]
                    for row in pre_registration_records
                    if bool(row.get("matched", False))
                ],
                dtype=float,
            )
            matched_measured_points = np.asarray(
                [
                    [float(row["measured_x"]), float(row["measured_y"]), float(row["measured_z"])]
                    for row in pre_registration_records
                    if bool(row.get("matched", False))
                ],
                dtype=float,
            )
            registration_transform, rms_before = _estimate_rigid_transform(matched_design_points, matched_measured_points)
            axis_records, node_records = _apply_transform_to_axis_node_records(axis_records, node_records, registration_transform)
            registration_summary.update(
                {
                    "enabled": True,
                    "pair_count": int(matched_design_points.shape[0]),
                    "rms_before": float(rms_before),
                    "transform_4x4": registration_transform.tolist(),
                }
            )

        deviation_records = compare_design_nodes_to_measured_points(
            node_records=node_records,
            measured_points=measured_points,
            max_match_distance=args.node_match_distance,
        )
        if registration_summary.get("enabled"):
            matched_design_points = np.asarray(
                [
                    [float(row["design_x"]), float(row["design_y"]), float(row["design_z"])]
                    for row in deviation_records
                    if bool(row.get("matched", False))
                ],
                dtype=float,
            )
            matched_measured_points = np.asarray(
                [
                    [float(row["measured_x"]), float(row["measured_y"]), float(row["measured_z"])]
                    for row in deviation_records
                    if bool(row.get("matched", False))
                ],
                dtype=float,
            )
            _, rms_after = _estimate_rigid_transform(matched_design_points, matched_measured_points)
            registration_summary["rms_after"] = float(rms_after)

        report_csv = args.export_node_deviation_csv
        if report_csv is None and output_dir is not None:
            report_csv = str(output_dir / f"{args.report_name}.csv")

        if report_csv:
            export_records_to_csv(
                csv_path=report_csv,
                fieldnames=[
                    "measured_index",
                    "measured_x",
                    "measured_y",
                    "measured_z",
                    "node_id",
                    "node_role",
                    "axis_count",
                    "axis_ids",
                    "design_index",
                    "design_x",
                    "design_y",
                    "design_z",
                    "dx",
                    "dy",
                    "dz",
                    "deviation_3d",
                    "matched",
                ],
                records=deviation_records,
            )

        crosswalk_records = build_design_node_crosswalk_records(node_records=node_records, deviation_records=deviation_records)
        crosswalk_csv = args.export_node_crosswalk_csv
        if crosswalk_csv is None and output_dir is not None:
            crosswalk_csv = str(output_dir / f"{args.report_name}_crosswalk.csv")
        if crosswalk_csv:
            export_records_to_csv(
                csv_path=crosswalk_csv,
                fieldnames=[
                    "measured_index",
                    "measured_x",
                    "measured_y",
                    "measured_z",
                    "node_id",
                    "node_role",
                    "axis_count",
                    "axis_ids",
                    "design_index",
                    "design_x",
                    "design_y",
                    "design_z",
                    "matched",
                    "dx",
                    "dy",
                    "dz",
                    "deviation_3d",
                ],
                records=crosswalk_records,
            )

        report_json_path: Optional[Path] = None
        if output_dir is not None:
            report_json_path = output_dir / f"{args.report_name}.json"

        report_summary = summarize_node_deviation_records(deviation_records)
        report_summary.update(
            {
                "design_dxf": str(args.dxf_path),
                "measured_axes_ply": str(args.measured_axes_ply) if args.measured_axes_ply else None,
                "measured_nodes_ply": str(args.measured_nodes_ply),
                "design_axis_count": int(len(axis_records)),
                "design_node_count": int(len(node_records)),
                "design_node_count_before_junction_filter": int(len(node_records_before_filter)),
                "design_node_count_after_junction_filter": int(len(node_records)),
                "design_scale": float(args.design_scale),
                "junction_min_axis_count": int(args.junction_min_axis_count),
                "auto_register": registration_summary,
                "node_match_distance": float(args.node_match_distance),
                "measured_axes_stats": measured_axes_stats,
                "measured_nodes_stats": measured_nodes_stats,
                "report_csv": report_csv,
                "crosswalk_csv": crosswalk_csv if "crosswalk_csv" in locals() else None,
            }
        )
        if report_json_path is not None:
            with report_json_path.open("w", encoding="utf-8") as f:
                import json

                json.dump(report_summary, f, ensure_ascii=False, indent=2)

        print(
            f"节点偏差比较完成: matched={report_summary['matched_count']}/{report_summary['total_count']}, "
            f"measured_points={measured_nodes_stats['point_count'] if measured_nodes_stats else len(measured_points)}"
        )
        if report_summary.get("matched_count", 0):
            print(
                f"节点偏差统计: mean={report_summary.get('mean_deviation_3d', 0.0):.6f}, "
                f"max={report_summary.get('max_deviation_3d', 0.0):.6f}, "
                f"min={report_summary.get('min_deviation_3d', 0.0):.6f}, "
                f"rmse={report_summary.get('rmse_deviation_3d', 0.0):.6f}"
            )
        if report_json_path is not None:
            print(f"节点偏差报告JSON已保存: {report_json_path}")

    if args.open3d_deviation and args.measured_nodes_ply:
        visualize_node_deviation_open3d(
            axis_records=axis_records,
            node_records=node_records,
            deviation_records=deviation_records if "deviation_records" in locals() else [],
        )
    elif args.open3d:
        visualize_axis_and_nodes_open3d(
            axis_records=axis_records,
            node_records=node_records,
            node_radius=args.open3d_node_radius,
        )
    else:
        visualize_axis_and_nodes(
            axis_records=axis_records,
            node_records=node_records,
            save_fig_path=args.save_fig,
            show_labels=args.show_labels,
            show_figure=not args.no_show,
        )
