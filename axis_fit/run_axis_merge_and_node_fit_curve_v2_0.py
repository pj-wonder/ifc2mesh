from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
try:
    import open3d as o3d
except ImportError:  # keep --help usable on machines without Open3D
    o3d = None


@dataclass
class AxisCurveItem:
    """Tube axis represented by a 3D curve instead of one global straight line."""

    name: str
    points: np.ndarray
    centerline: np.ndarray          # M x 3 centerline points
    tangents: np.ndarray            # M x 3 local tangents
    start: np.ndarray
    end: np.ndarray
    start_tangent: np.ndarray
    end_tangent: np.ndarray
    length: float
    radius_median: float
    circle_rmse_median: float
    anchor: np.ndarray              # compatibility: mean centerline point
    direction: np.ndarray           # compatibility: chord direction / average direction


# Backward-compatible alias used by several utility functions.
AxisItem = AxisCurveItem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract curved tube centerlines, merge continuous components, and fit junction nodes"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "ifc2mesh/result_run_ifc311_world/label_transfer/subresult_bbox_split_20260426_152254/components_pcd"
        ),
        help="Input directory of component pcd files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ifc2mesh/node_fit_results"),
        help="Output directory",
    )

    # Curved-axis extraction parameters.
    parser.add_argument("--curve-step", type=float, default=0.05, help="Centerline sampling step along initial axis (m)")
    parser.add_argument("--curve-window", type=float, default=0.25, help="Local PCA window length for initial curve (m)")
    parser.add_argument("--slice-thickness", type=float, default=0.06, help="Adaptive local slice thickness (m)")
    parser.add_argument("--min-slice-points", type=int, default=80, help="Minimum points in a local slice")
    parser.add_argument("--smooth-k", type=int, default=5, help="Odd moving-average window for centerline smoothing")
    parser.add_argument("--refine-iterations", type=int, default=2, help="Circle-slice refinement iterations")
    parser.add_argument("--max-circle-rmse", type=float, default=0.04, help="Reject a fitted section if circle RMSE exceeds this value (m)")
    parser.add_argument(
        "--fallback-to-straight",
        action="store_true",
        help="If curved extraction fails, use straight PCA axis as fallback instead of skipping the component",
    )

    # Curve-aware merge and node parameters.
    parser.add_argument("--merge-endpoint-dist", type=float, default=0.15, help="Max endpoint distance for curve merging (m)")
    parser.add_argument("--merge-tangent-angle-deg", type=float, default=20.0, help="Max tangent angle for curve merging")
    parser.add_argument("--node-grouping-radius", type=float, default=0.25, help="Endpoint clustering radius for nodes (m)")
    parser.add_argument("--multi-min-members", type=int, default=2, help="Minimum clustered endpoints to output a node")
    parser.add_argument("--min-axis-length", type=float, default=0.8, help="Drop axes shorter than this length (m)")
    parser.add_argument("--min-axis-points", type=int, default=1000, help="Drop axes with fewer points than this threshold")

    # Legacy options retained so old command lines do not break. They are not used by curve-aware merging.
    parser.add_argument("--merge-angle-deg", type=float, default=6.0, help=argparse.SUPPRESS)
    parser.add_argument("--merge-line-dist", type=float, default=0.08, help=argparse.SUPPRESS)
    parser.add_argument("--merge-gap", type=float, default=0.35, help=argparse.SUPPRESS)
    parser.add_argument("--node-distance-threshold", type=float, default=0.25, help=argparse.SUPPRESS)
    parser.add_argument("--segment-margin", type=float, default=0.5, help=argparse.SUPPRESS)
    return parser.parse_args()


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v.copy()
    return v / n


def _fit_straight_axis(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    center = points.mean(axis=0)
    x = points - center
    cov = x.T @ x
    eigvals, eigvecs = np.linalg.eigh(cov)
    d = _normalize(eigvecs[:, int(np.argmax(eigvals))])
    t = x @ d
    start = center + float(np.min(t)) * d
    end = center + float(np.max(t)) * d
    return center, d, start, end, float(np.linalg.norm(end - start))


def _make_plane_basis(tangent: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    e3 = _normalize(tangent)
    tmp = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(tmp, e3))) > 0.90:
        tmp = np.array([0.0, 1.0, 0.0], dtype=float)
    e1 = _normalize(np.cross(e3, tmp))
    e2 = _normalize(np.cross(e3, e1))
    return e1, e2, e3


def _fit_circle_2d(xy: np.ndarray) -> Tuple[np.ndarray, float, float]:
    x = xy[:, 0]
    y = xy[:, 1]
    a = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy, c0 = sol
    radius2 = max(float(c0 + cx * cx + cy * cy), 0.0)
    radius = float(np.sqrt(radius2))
    residual = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - radius
    rmse = float(np.sqrt(np.mean(residual ** 2))) if len(residual) else float("inf")
    return np.array([float(cx), float(cy)], dtype=float), radius, rmse


def _compute_tangents(centerline: np.ndarray) -> np.ndarray:
    n = len(centerline)
    tangents = np.zeros_like(centerline)
    if n == 1:
        tangents[0] = np.array([1.0, 0.0, 0.0])
        return tangents
    for i in range(n):
        if i == 0:
            t = centerline[1] - centerline[0]
        elif i == n - 1:
            t = centerline[-1] - centerline[-2]
        else:
            t = centerline[i + 1] - centerline[i - 1]
        tangents[i] = _normalize(t)
    # Keep tangent directions continuous.
    for i in range(1, n):
        if float(np.dot(tangents[i - 1], tangents[i])) < 0:
            tangents[i] *= -1.0
    return tangents


def _smooth_centerline(centerline: np.ndarray, k: int) -> np.ndarray:
    if len(centerline) == 0:
        return centerline
    k = int(k)
    if k < 3 or len(centerline) < 3:
        return centerline.copy()
    if k % 2 == 0:
        k += 1
    k = min(k, len(centerline) if len(centerline) % 2 == 1 else len(centerline) - 1)
    if k < 3:
        return centerline.copy()
    pad = k // 2
    padded = np.pad(centerline, ((pad, pad), (0, 0)), mode="edge")
    return np.asarray([padded[i : i + k].mean(axis=0) for i in range(len(centerline))])


def _initial_centerline_by_sliding_pca(
    points: np.ndarray,
    step: float,
    window: float,
    min_pts: int,
) -> Tuple[np.ndarray, np.ndarray]:
    center, d0, _, _, _ = _fit_straight_axis(points)
    t = (points - center) @ d0
    t_min, t_max = float(np.min(t)), float(np.max(t))
    if t_max <= t_min:
        return np.empty((0, 3)), np.empty((0, 3))

    step = max(float(step), 1e-4)
    window = max(float(window), step * 2.0)
    samples = np.arange(t_min, t_max + 0.5 * step, step)

    centers: List[np.ndarray] = []
    tangents: List[np.ndarray] = []
    for s in samples:
        mask = (t >= s - window / 2.0) & (t <= s + window / 2.0)
        local = points[mask]
        if local.shape[0] < min_pts:
            continue
        c = local.mean(axis=0)
        x = local - c
        cov = x.T @ x
        eigvals, eigvecs = np.linalg.eigh(cov)
        d = _normalize(eigvecs[:, int(np.argmax(eigvals))])
        if tangents and float(np.dot(d, tangents[-1])) < 0:
            d *= -1.0
        centers.append(c)
        tangents.append(d)

    centers_arr = np.asarray(centers, dtype=float)
    tangents_arr = np.asarray(tangents, dtype=float)
    if len(centers_arr) >= 2 and float(np.dot(centers_arr[-1] - centers_arr[0], d0)) < 0:
        centers_arr = centers_arr[::-1]
        tangents_arr = -tangents_arr[::-1]
    return centers_arr, tangents_arr


def _refine_centerline_by_circle_slices(
    points: np.ndarray,
    centers: np.ndarray,
    tangents: np.ndarray,
    thickness: float,
    min_pts: int,
    max_circle_rmse: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    refined: List[np.ndarray] = []
    radii: List[float] = []
    rmses: List[float] = []

    for c0, t0 in zip(centers, tangents):
        t0 = _normalize(t0)
        signed_dist = (points - c0) @ t0
        local = points[np.abs(signed_dist) <= thickness / 2.0]
        if local.shape[0] < min_pts:
            continue

        e1, e2, _ = _make_plane_basis(t0)
        uv = np.column_stack([(local - c0) @ e1, (local - c0) @ e2])
        center_2d, radius, rmse = _fit_circle_2d(uv)
        if np.isfinite(rmse) and rmse <= max_circle_rmse:
            refined.append(c0 + center_2d[0] * e1 + center_2d[1] * e2)
            radii.append(radius)
            rmses.append(rmse)

    return np.asarray(refined, dtype=float), np.asarray(radii, dtype=float), np.asarray(rmses, dtype=float)


def _curve_length(centerline: np.ndarray) -> float:
    if len(centerline) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(centerline, axis=0), axis=1)))


def _axis_curve_from_centerline(name: str, points: np.ndarray, centerline: np.ndarray, radii: np.ndarray, rmses: np.ndarray) -> AxisCurveItem:
    centerline = np.asarray(centerline, dtype=float)
    tangents = _compute_tangents(centerline)
    start = centerline[0]
    end = centerline[-1]
    chord = end - start
    direction = _normalize(chord) if np.linalg.norm(chord) > 1e-12 else _normalize(np.mean(tangents, axis=0))
    radius_med = float(np.median(radii)) if len(radii) else 0.0
    rmse_med = float(np.median(rmses)) if len(rmses) else 0.0
    return AxisCurveItem(
        name=name,
        points=points,
        centerline=centerline,
        tangents=tangents,
        start=start,
        end=end,
        start_tangent=tangents[0],
        end_tangent=tangents[-1],
        length=_curve_length(centerline),
        radius_median=radius_med,
        circle_rmse_median=rmse_med,
        anchor=centerline.mean(axis=0),
        direction=direction,
    )


def _fit_curve_axis(
    name: str,
    points: np.ndarray,
    step: float,
    window: float,
    slice_thickness: float,
    min_slice_pts: int,
    smooth_k: int,
    refine_iterations: int,
    max_circle_rmse: float,
    fallback_to_straight: bool,
) -> Optional[AxisCurveItem]:
    centers, tangents = _initial_centerline_by_sliding_pca(points, step, window, min_slice_pts)
    if len(centers) < 3:
        if not fallback_to_straight:
            return None
        _, d, start, end, length = _fit_straight_axis(points)
        cl = np.vstack([start, end])
        return _axis_curve_from_centerline(name, points, cl, np.asarray([]), np.asarray([]))

    radii = np.asarray([])
    rmses = np.asarray([])
    centers = _smooth_centerline(centers, smooth_k)
    tangents = _compute_tangents(centers)

    for _ in range(max(1, int(refine_iterations))):
        refined, radii, rmses = _refine_centerline_by_circle_slices(
            points,
            centers,
            tangents,
            thickness=max(slice_thickness, 1e-4),
            min_pts=max(5, int(min_slice_pts)),
            max_circle_rmse=max_circle_rmse,
        )
        if len(refined) < 3:
            break
        centers = _smooth_centerline(refined, smooth_k)
        tangents = _compute_tangents(centers)

    if len(centers) < 2:
        return None
    return _axis_curve_from_centerline(name, points, centers, radii, rmses)


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


def _solve_global_pseudointersection(anchors: np.ndarray, dirs: np.ndarray) -> Tuple[np.ndarray, float, float]:
    i3 = np.eye(3, dtype=float)
    s = np.zeros((3, 3), dtype=float)
    v = np.zeros(3, dtype=float)
    for a, d in zip(anchors, dirs):
        d = _normalize(d)
        p = i3 - np.outer(d, d)
        s += p
        v += p @ a
    c = np.linalg.pinv(s) @ v
    dists = [float(np.linalg.norm(np.cross(c - a, _normalize(d)))) for a, d in zip(anchors, dirs)]
    return c, float(np.mean(dists)) if dists else 0.0, float(np.max(dists)) if dists else 0.0


def _load_axes(input_dir: Path, args: argparse.Namespace) -> Tuple[List[AxisCurveItem], List[Dict[str, object]]]:
    axes: List[AxisCurveItem] = []
    failed: List[Dict[str, object]] = []
    ascii_dir = input_dir.parent / "_ascii_alias_cache"
    ascii_dir.mkdir(parents=True, exist_ok=True)

    for idx, p in enumerate(sorted(input_dir.glob("*.pcd")), start=1):
        alias = ascii_dir / f"comp_{idx:04d}.pcd"
        if (not alias.exists()) or (alias.stat().st_size != p.stat().st_size):
            shutil.copyfile(p, alias)
        pcd = o3d.io.read_point_cloud(str(alias))
        pts = np.asarray(pcd.points)
        if pts.size == 0:
            failed.append({"name": p.stem, "reason": "empty_pcd"})
            continue
        axis = _fit_curve_axis(
            name=p.stem,
            points=pts,
            step=args.curve_step,
            window=args.curve_window,
            slice_thickness=args.slice_thickness,
            min_slice_pts=args.min_slice_points,
            smooth_k=args.smooth_k,
            refine_iterations=args.refine_iterations,
            max_circle_rmse=args.max_circle_rmse,
            fallback_to_straight=args.fallback_to_straight,
        )
        if axis is None:
            failed.append({"name": p.stem, "reason": "curve_axis_fit_failed"})
            continue
        axes.append(axis)
    return axes, failed


def _union_find(n: int) -> Tuple[List[int], Callable[[int], int], Callable[[int, int], None]]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    return parent, find, union


def _best_endpoint_match(a: AxisCurveItem, b: AxisCurveItem) -> Dict[str, object]:
    endpoints = [
        ("a_start", a.start, a.start_tangent, "b_start", b.start, b.start_tangent),
        ("a_start", a.start, a.start_tangent, "b_end", b.end, -b.end_tangent),
        ("a_end", a.end, -a.end_tangent, "b_start", b.start, b.start_tangent),
        ("a_end", a.end, -a.end_tangent, "b_end", b.end, -b.end_tangent),
    ]
    best: Optional[Dict[str, object]] = None
    for wa, pa, ta, wb, pb, tb in endpoints:
        dist = float(np.linalg.norm(pa - pb))
        cosv = float(np.clip(abs(np.dot(_normalize(ta), _normalize(tb))), -1.0, 1.0))
        angle = float(np.rad2deg(np.arccos(cosv)))
        score = dist + 0.01 * angle
        row = {"a_end": wa, "b_end": wb, "dist": dist, "angle_deg": angle, "score": score}
        if best is None or score < float(best["score"]):
            best = row
    return best or {"dist": float("inf"), "angle_deg": 180.0, "score": float("inf")}


def _concatenate_centerlines(group_axes: List[AxisCurveItem]) -> np.ndarray:
    """Order and concatenate centerlines inside a merge group by nearest endpoints."""
    unused = group_axes[:]
    current = unused.pop(0).centerline.copy()
    while unused:
        head, tail = current[0], current[-1]
        best_i, best_mode, best_dist = -1, "", float("inf")
        for i, a in enumerate(unused):
            candidates = [
                ("tail_start", np.linalg.norm(tail - a.start)),
                ("tail_end", np.linalg.norm(tail - a.end)),
                ("head_start", np.linalg.norm(head - a.start)),
                ("head_end", np.linalg.norm(head - a.end)),
            ]
            mode, dist = min(candidates, key=lambda x: x[1])
            if float(dist) < best_dist:
                best_i, best_mode, best_dist = i, mode, float(dist)
        nxt = unused.pop(best_i).centerline.copy()
        if best_mode == "tail_start":
            current = np.vstack([current, nxt])
        elif best_mode == "tail_end":
            current = np.vstack([current, nxt[::-1]])
        elif best_mode == "head_start":
            current = np.vstack([nxt[::-1], current])
        else:  # head_end
            current = np.vstack([nxt, current])
    return current


def merge_axis_near_components_curve(
    axes: List[AxisCurveItem],
    merge_endpoint_dist: float,
    merge_tangent_angle_deg: float,
) -> Tuple[List[AxisCurveItem], List[Dict[str, str]], List[Dict[str, object]]]:
    if not axes:
        return [], [], []
    parent, find, union = _union_find(len(axes))
    merge_checks: List[Dict[str, object]] = []

    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            best = _best_endpoint_match(axes[i], axes[j])
            do_merge = bool(best["dist"] <= merge_endpoint_dist and best["angle_deg"] <= merge_tangent_angle_deg)
            merge_checks.append(
                {
                    "component_1": axes[i].name,
                    "component_2": axes[j].name,
                    "endpoint_distance": float(best["dist"]),
                    "tangent_angle_deg": float(best["angle_deg"]),
                    "merged": do_merge,
                }
            )
            if do_merge:
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(len(axes)):
        groups.setdefault(find(i), []).append(i)

    merged: List[AxisCurveItem] = []
    mapping_rows: List[Dict[str, str]] = []
    for gid, idxs in enumerate(groups.values(), start=1):
        group = [axes[i] for i in idxs]
        all_pts = np.vstack([a.points for a in group])
        merged_centerline = _concatenate_centerlines(group) if len(group) > 1 else group[0].centerline.copy()
        merged_centerline = _smooth_centerline(merged_centerline, 5)
        radii = np.asarray([a.radius_median for a in group if a.radius_median > 0], dtype=float)
        rmses = np.asarray([a.circle_rmse_median for a in group if a.circle_rmse_median > 0], dtype=float)
        merged_name = f"merged_{gid:03d}"
        merged_axis = _axis_curve_from_centerline(merged_name, all_pts, merged_centerline, radii, rmses)
        merged.append(merged_axis)
        for a in group:
            mapping_rows.append({"merged_name": merged_name, "source_name": a.name})
    return merged, mapping_rows, merge_checks


def filter_noise_axes(
    axes: List[AxisCurveItem],
    mapping_rows: List[Dict[str, str]],
    min_axis_length: float,
    min_axis_points: int,
) -> Tuple[List[AxisCurveItem], List[Dict[str, str]], List[Dict[str, object]]]:
    kept: List[AxisCurveItem] = []
    removed: List[Dict[str, object]] = []
    for a in axes:
        reasons: List[str] = []
        if float(a.length) < float(min_axis_length):
            reasons.append("short_length")
        if int(a.points.shape[0]) < int(min_axis_points):
            reasons.append("few_points")
        if len(a.centerline) < 2:
            reasons.append("too_few_centerline_points")
        if reasons:
            removed.append(
                {
                    "name": a.name,
                    "point_count": int(a.points.shape[0]),
                    "axis_length_m": float(a.length),
                    "centerline_point_count": int(len(a.centerline)),
                    "remove_reason": "|".join(reasons),
                }
            )
        else:
            kept.append(a)
    keep_names = {a.name for a in kept}
    kept_mapping = [r for r in mapping_rows if str(r.get("merged_name", "")) in keep_names]
    return kept, kept_mapping, removed


def solve_nodes_from_curve_endpoints(
    axes: List[AxisCurveItem],
    grouping_radius: float,
    multi_min_members: int,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for a in axes:
        records.append({"component": a.name, "which": "start", "point": a.start, "tangent": a.start_tangent})
        records.append({"component": a.name, "which": "end", "point": a.end, "tangent": -a.end_tangent})

    if not records:
        return []
    pts = np.asarray([r["point"] for r in records], dtype=float)
    clusters = _cluster_points(pts, radius=grouping_radius)

    rows: List[Dict[str, object]] = []
    counter = 0
    for cl in clusters:
        members = [records[i] for i in cl]
        unique_components = sorted({str(m["component"]) for m in members})
        if len(members) < int(multi_min_members) or len(unique_components) < 2:
            continue
        anchors = np.asarray([m["point"] for m in members], dtype=float)
        dirs = np.asarray([m["tangent"] for m in members], dtype=float)
        node_ls, mean_dist, max_dist = _solve_global_pseudointersection(anchors, dirs)
        node_mean = anchors.mean(axis=0)
        # Least-squares can be unstable for almost parallel lines; fall back to endpoint mean if it drifts too far.
        if float(np.linalg.norm(node_ls - node_mean)) > 2.0 * grouping_radius:
            node = node_mean
            method = "curve_endpoint_cluster_mean"
            mean_dist = float(np.mean(np.linalg.norm(anchors - node, axis=1)))
            max_dist = float(np.max(np.linalg.norm(anchors - node, axis=1)))
        else:
            node = node_ls
            method = "curve_endpoint_tangent_least_squares"
        counter += 1
        rows.append(
            {
                "node_id": f"node_curve_{counter:04d}",
                "x": float(node[0]),
                "y": float(node[1]),
                "z": float(node[2]),
                "distance": float(mean_dist),
                "max_line_distance": float(max_dist),
                "is_intersection_expected": True,
                "solve_method": method,
                "involved_components": "|".join(unique_components),
                "member_count": int(len(unique_components)),
                "involved_endpoints": "|".join(f"{m['component']}:{m['which']}" for m in members),
            }
        )
    return rows


def _write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in columns})


def _write_centerline_csv(path: Path, axes: List[AxisCurveItem]) -> None:
    rows: List[Dict[str, object]] = []
    for a in axes:
        for idx, (p, t) in enumerate(zip(a.centerline, a.tangents)):
            rows.append(
                {
                    "name": a.name,
                    "point_index": idx,
                    "x": float(p[0]),
                    "y": float(p[1]),
                    "z": float(p[2]),
                    "tx": float(t[0]),
                    "ty": float(t[1]),
                    "tz": float(t[2]),
                    "radius_median": float(a.radius_median),
                    "circle_rmse_median": float(a.circle_rmse_median),
                }
            )
    _write_csv(path, rows, ["name", "point_index", "x", "y", "z", "tx", "ty", "tz", "radius_median", "circle_rmse_median"])


def _write_centerline_pcds(out_dir: Path, axes: List[AxisCurveItem]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for a in axes:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(a.centerline)
        o3d.io.write_point_cloud(str(out_dir / f"{a.name}_centerline.pcd"), pcd)


def _resolve_ifc2mesh_root(path_like: Path) -> Path:
    resolved = path_like.resolve()
    parts = list(resolved.parts)
    if "ifc2mesh" in parts:
        idx = parts.index("ifc2mesh")
        return Path(*parts[: idx + 1])
    return resolved.parent


def main() -> None:
    args = parse_args()
    if o3d is None:
        raise ImportError("open3d is required to read/write PCD files. Install it with: pip install open3d")
    if not args.input_dir.exists():
        raise FileNotFoundError(f"input-dir not found: {args.input_dir}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / f"run_curve_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    axes, failed_axes = _load_axes(args.input_dir, args)
    if not axes:
        raise RuntimeError("No valid pcd components produced a curved axis; try --fallback-to-straight or relax slice parameters")

    merged_axes, mapping_rows, merge_checks = merge_axis_near_components_curve(
        axes,
        merge_endpoint_dist=args.merge_endpoint_dist,
        merge_tangent_angle_deg=args.merge_tangent_angle_deg,
    )

    merged_axes, mapping_rows, removed_axes = filter_noise_axes(
        merged_axes,
        mapping_rows,
        min_axis_length=args.min_axis_length,
        min_axis_points=args.min_axis_points,
    )
    if not merged_axes:
        raise RuntimeError("All merged axes were removed by noise filters; relax --min-axis-length/--min-axis-points")

    merged_pcd_dir = out_dir / "merged_components_pcd"
    merged_pcd_dir.mkdir(parents=True, exist_ok=True)
    centerline_pcd_dir = out_dir / "centerline_pcd"
    _write_centerline_pcds(centerline_pcd_dir, merged_axes)

    axis_rows: List[Dict[str, object]] = []
    for a in merged_axes:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(a.points)
        pcd_path = merged_pcd_dir / f"{a.name}.pcd"
        o3d.io.write_point_cloud(str(pcd_path), pcd)
        axis_rows.append(
            {
                "name": a.name,
                "point_count": int(a.points.shape[0]),
                "axis_length_m": float(a.length),
                "centerline_point_count": int(len(a.centerline)),
                "radius_median_m": float(a.radius_median),
                "circle_rmse_median_m": float(a.circle_rmse_median),
                "center_x": float(a.anchor[0]),
                "center_y": float(a.anchor[1]),
                "center_z": float(a.anchor[2]),
                "start_x": float(a.start[0]),
                "start_y": float(a.start[1]),
                "start_z": float(a.start[2]),
                "end_x": float(a.end[0]),
                "end_y": float(a.end[1]),
                "end_z": float(a.end[2]),
                "start_tx": float(a.start_tangent[0]),
                "start_ty": float(a.start_tangent[1]),
                "start_tz": float(a.start_tangent[2]),
                "end_tx": float(a.end_tangent[0]),
                "end_ty": float(a.end_tangent[1]),
                "end_tz": float(a.end_tangent[2]),
                "chord_dir_x": float(a.direction[0]),
                "chord_dir_y": float(a.direction[1]),
                "chord_dir_z": float(a.direction[2]),
                "pcd_path": str(pcd_path),
                "centerline_pcd_path": str(centerline_pcd_dir / f"{a.name}_centerline.pcd"),
            }
        )

    axis_csv = out_dir / "merged_member_curve_axes.csv"
    axis_columns = [
        "name", "point_count", "axis_length_m", "centerline_point_count", "radius_median_m", "circle_rmse_median_m",
        "center_x", "center_y", "center_z", "start_x", "start_y", "start_z", "end_x", "end_y", "end_z",
        "start_tx", "start_ty", "start_tz", "end_tx", "end_ty", "end_tz",
        "chord_dir_x", "chord_dir_y", "chord_dir_z", "pcd_path", "centerline_pcd_path",
    ]
    _write_csv(axis_csv, axis_rows, axis_columns)

    centerline_csv = out_dir / "curve_centerline_points.csv"
    _write_centerline_csv(centerline_csv, merged_axes)

    mapping_csv = out_dir / "merge_mapping.csv"
    _write_csv(mapping_csv, mapping_rows, ["merged_name", "source_name"])

    merge_checks_csv = out_dir / "curve_merge_checks.csv"
    _write_csv(merge_checks_csv, merge_checks, ["component_1", "component_2", "endpoint_distance", "tangent_angle_deg", "merged"])

    removed_axes_csv = out_dir / "removed_noise_axes.csv"
    _write_csv(removed_axes_csv, removed_axes, ["name", "point_count", "axis_length_m", "centerline_point_count", "remove_reason"])

    failed_axes_csv = out_dir / "failed_axis_fits.csv"
    _write_csv(failed_axes_csv, failed_axes, ["name", "reason"])

    nodes_rows = solve_nodes_from_curve_endpoints(
        merged_axes,
        grouping_radius=args.node_grouping_radius,
        multi_min_members=args.multi_min_members,
    )
    nodes_csv = out_dir / "nodes_from_curve_axes.csv"
    node_columns = [
        "node_id", "x", "y", "z", "distance", "max_line_distance", "is_intersection_expected",
        "solve_method", "involved_components", "member_count", "involved_endpoints",
    ]
    _write_csv(nodes_csv, nodes_rows, node_columns)

    summary = {
        "input_dir": str(args.input_dir),
        "output_dir": str(out_dir),
        "source_component_count": int(len(axes) + len(failed_axes)),
        "fit_success_component_count": int(len(axes)),
        "fit_failed_component_count": int(len(failed_axes)),
        "merged_component_count": int(len(merged_axes)),
        "curve_step_m": float(args.curve_step),
        "curve_window_m": float(args.curve_window),
        "slice_thickness_m": float(args.slice_thickness),
        "min_slice_points": int(args.min_slice_points),
        "smooth_k": int(args.smooth_k),
        "refine_iterations": int(args.refine_iterations),
        "max_circle_rmse_m": float(args.max_circle_rmse),
        "merge_endpoint_dist_m": float(args.merge_endpoint_dist),
        "merge_tangent_angle_deg": float(args.merge_tangent_angle_deg),
        "node_grouping_radius_m": float(args.node_grouping_radius),
        "multi_min_members": int(args.multi_min_members),
        "min_axis_length_m": float(args.min_axis_length),
        "min_axis_points": int(args.min_axis_points),
        "node_count": int(len(nodes_rows)),
        "removed_noise_axis_count": int(len(removed_axes)),
        "files": {
            "merged_curve_axes_csv": str(axis_csv),
            "centerline_points_csv": str(centerline_csv),
            "merge_mapping_csv": str(mapping_csv),
            "curve_merge_checks_csv": str(merge_checks_csv),
            "removed_noise_axes_csv": str(removed_axes_csv),
            "failed_axis_fits_csv": str(failed_axes_csv),
            "nodes_csv": str(nodes_csv),
            "merged_components_pcd_dir": str(merged_pcd_dir),
            "centerline_pcd_dir": str(centerline_pcd_dir),
        },
    }

    summary_json = out_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] source components: {summary['source_component_count']}")
    print(f"[DONE] curve axis fit success: {len(axes)}, failed: {len(failed_axes)}")
    print(f"[DONE] merged components: {len(merged_axes)}")
    print(f"[DONE] removed noise axes: {len(removed_axes)}")
    print(f"[DONE] nodes: {len(nodes_rows)}")
    print(f"[OUT] {axis_csv}")
    print(f"[OUT] {centerline_csv}")
    print(f"[OUT] {nodes_csv}")
    print(f"[OUT] {summary_json}")
    print(f"[NOTE] Result root (big folder under workspace): {_resolve_ifc2mesh_root(out_dir)}")


if __name__ == "__main__":
    main()
