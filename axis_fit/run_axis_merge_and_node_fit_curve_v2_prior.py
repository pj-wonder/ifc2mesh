from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

try:
    import open3d as o3d
except ImportError:  # keep --help usable on machines without Open3D
    o3d = None

try:
    from axis_fit import run_axis_merge_and_node_fit_curve_v2_0 as v2_0
except ImportError:
    v2_0 = None


@dataclass
class AxisCurveItem:
    name: str
    points: np.ndarray
    centerline: np.ndarray
    tangents: np.ndarray
    start: np.ndarray
    end: np.ndarray
    start_tangent: np.ndarray
    end_tangent: np.ndarray
    length: float
    radius_median: float
    circle_rmse_median: float
    anchor: np.ndarray
    direction: np.ndarray


AxisItem = AxisCurveItem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract curved tube centerlines with optional IFC prior corrections"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--component-ifc-map", type=Path, default=None, help="component_ifc_prior_map.csv or json from v3 run")

    # curved params (copied from v2)
    parser.add_argument("--curve-step", type=float, default=0.05)
    parser.add_argument("--curve-window", type=float, default=0.25)
    parser.add_argument("--slice-thickness", type=float, default=0.06)
    parser.add_argument("--min-slice-points", type=int, default=80)
    parser.add_argument("--smooth-k", type=int, default=5)
    parser.add_argument("--refine-iterations", type=int, default=2)
    parser.add_argument("--max-circle-rmse", type=float, default=0.04)
    parser.add_argument("--fallback-to-straight", action="store_true")

    parser.add_argument("--merge-endpoint-dist", type=float, default=0.20)
    parser.add_argument("--merge-tangent-angle-deg", type=float, default=25.0)
    parser.add_argument("--node-grouping-radius", type=float, default=0.25)
    parser.add_argument("--multi-min-members", type=int, default=2)
    parser.add_argument("--min-axis-length", type=float, default=0.1)
    parser.add_argument("--min-axis-points", type=int, default=3)
    return parser.parse_args()


def _print_merge_diagnostics(axes: List[AxisCurveItem], merge_endpoint_dist: float, merge_tangent_angle_deg: float) -> None:
    """Print diagnostic info on which axis pairs fail to merge."""
    print(f"\n[DIAG] Merge thresholds: dist={merge_endpoint_dist:.3f}m, angle={merge_tangent_angle_deg:.1f}°")
    merge_count = 0
    fail_count = 0
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            matched = (
                _endpoint_match(axes[i], "start", axes[j], "start", merge_endpoint_dist, merge_tangent_angle_deg)
                or _endpoint_match(axes[i], "start", axes[j], "end", merge_endpoint_dist, merge_tangent_angle_deg)
                or _endpoint_match(axes[i], "end", axes[j], "start", merge_endpoint_dist, merge_tangent_angle_deg)
                or _endpoint_match(axes[i], "end", axes[j], "end", merge_endpoint_dist, merge_tangent_angle_deg)
            )
            if matched:
                merge_count += 1
            else:
                min_dist = float("inf")
                best_pair = None
                for endpoint_a in ["start", "end"]:
                    for endpoint_b in ["start", "end"]:
                        pa = axes[i].start if endpoint_a == "start" else axes[i].end
                        pb = axes[j].start if endpoint_b == "start" else axes[j].end
                        d = float(np.linalg.norm(pa - pb))
                        if d < min_dist:
                            min_dist = d
                            best_pair = (endpoint_a, endpoint_b)
                if min_dist < merge_endpoint_dist * 1.5:
                    # Compute angle for best pair
                    angle_deg = 0.0
                    if best_pair:
                        va = _normalize(axes[i].end - axes[i].start)
                        vb = _normalize(axes[j].end - axes[j].start)
                        angle_rad = np.arccos(np.clip(np.dot(va, vb), -1, 1))
                        angle_deg = float(np.degrees(angle_rad))
                        angle_deg = min(angle_deg, 180 - angle_deg)
                    fail_count += 1
                    print(f"  [NEAR] {axes[i].name} ↔ {axes[j].name}:")
                    print(f"         dist={min_dist:.4f}m (threshold={merge_endpoint_dist:.3f}m), "
                          f"angle={angle_deg:.1f}° (threshold={merge_tangent_angle_deg:.1f}°)")
    print(f"[DIAG] Potential merges: {merge_count}, Close misses (dist < 1.5x threshold): {fail_count}\n")


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


def _bridge_centerline_gaps(centerline: np.ndarray, gap_factor: float = 2.5) -> np.ndarray:
    if len(centerline) < 2:
        return centerline
    diffs = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    valid = diffs[diffs > 1e-9]
    if len(valid) == 0:
        return centerline
    step = float(np.median(valid))
    if step <= 1e-9:
        return centerline

    max_gap = step * max(float(gap_factor), 1.0)
    bridged: List[np.ndarray] = [centerline[0]]
    for prev, curr in zip(centerline[:-1], centerline[1:]):
        dist = float(np.linalg.norm(curr - prev))
        if dist > max_gap:
            fill_count = int(np.ceil(dist / step)) - 1
            if fill_count > 0:
                extra = np.linspace(prev, curr, fill_count + 2, axis=0)[1:-1]
                bridged.extend(extra)
        bridged.append(curr)
    return np.asarray(bridged, dtype=float)


def _initial_centerline_by_sliding_pca(points: np.ndarray, step: float, window: float, min_pts: int) -> Tuple[np.ndarray, np.ndarray]:
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


def _refine_centerline_by_circle_slices(points: np.ndarray, centers: np.ndarray, tangents: np.ndarray, thickness: float, min_pts: int, max_circle_rmse: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _densify_centerline(centerline: np.ndarray, max_gap_factor: float = 1.8, max_insert_points: int = 8) -> np.ndarray:
    if len(centerline) < 2:
        return centerline
    diffs = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    valid = diffs[diffs > 1e-9]
    if len(valid) == 0:
        return centerline
    step = float(np.median(valid))
    if step <= 1e-9:
        return centerline

    densified: List[np.ndarray] = [centerline[0]]
    for i in range(len(centerline) - 1):
        p0 = centerline[i]
        p1 = centerline[i + 1]
        seg = p1 - p0
        dist = float(np.linalg.norm(seg))
        if dist <= 1e-9:
            continue
        if dist > max_gap_factor * step:
            insert_n = int(np.clip(round(dist / step) - 1, 1, max_insert_points))
            for k in range(1, insert_n + 1):
                densified.append(p0 + seg * (k / (insert_n + 1)))
        densified.append(p1)
    return np.asarray(densified, dtype=float)


def _axis_curve_from_centerline(name: str, points: np.ndarray, centerline: np.ndarray, radii: np.ndarray, rmses: np.ndarray) -> AxisCurveItem:
    centerline = np.asarray(centerline, dtype=float)
    centerline = _densify_centerline(centerline)
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
        if len(refined) < 2:
            break
        centers = _bridge_centerline_gaps(_smooth_centerline(refined, smooth_k))
        tangents = _compute_tangents(centers)

    centers = _bridge_centerline_gaps(centers)
    if len(centers) < 2:
        return None
    return _axis_curve_from_centerline(name, points, centers, radii, rmses)


def _load_component_map(map_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if map_path is None or not map_path.exists():
        return {}
    data: Dict[str, Dict[str, Any]] = {}
    if map_path.suffix.lower() == ".json":
        js = json.load(map_path.open("r", encoding="utf-8"))
        # json is list of dicts
        if isinstance(js, list):
            for r in js:
                name = Path(r.get("segmented_pcd", "")).stem if r.get("segmented_pcd") else r.get("mesh_id") or r.get("name")
                if not name:
                    continue
                data[name] = r
    else:
        with map_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                name = Path(r.get("segmented_pcd", "")).stem if r.get("segmented_pcd") else r.get("mesh_id") or r.get("name")
                if not name:
                    continue
                data[name] = r
    return data


def _load_axes(input_dir: Path, args: argparse.Namespace, component_map: Dict[str, Dict[str, Any]]) -> Tuple[List[AxisCurveItem], List[Dict[str, object]]]:
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
            meta = component_map.get(p.stem)
            if meta is not None and str(meta.get("design_shape", "")).lower() == "straight":
                _, _, s, e, _ = _fit_straight_axis(pts)
                cl = np.vstack([s, e])
                axis = _axis_curve_from_centerline(p.stem, pts, cl, np.asarray([]), np.asarray([]))
            elif args.fallback_to_straight:
                _, _, s, e, _ = _fit_straight_axis(pts)
                cl = np.vstack([s, e])
                axis = _axis_curve_from_centerline(p.stem, pts, cl, np.asarray([]), np.asarray([]))
            else:
                failed.append({"name": p.stem, "reason": "curve_axis_fit_failed"})
                continue

        # Apply IFC prior correction: if component map marks design_shape == 'straight', replace centerline by straight fit
        meta = component_map.get(p.stem)
        if meta is not None and str(meta.get("design_shape", "")).lower() == "straight":
            # compute straight axis and convert to short centerline
            _, _, s, e, _ = _fit_straight_axis(pts)
            cl = np.vstack([s, e])
            axis = _axis_curve_from_centerline(p.stem, pts, cl, np.asarray([]), np.asarray([]))

        axes.append(axis)
    return axes, failed


def _endpoint_match(axis_a: AxisCurveItem, endpoint_a: str, axis_b: AxisCurveItem, endpoint_b: str, merge_endpoint_dist: float, merge_tangent_angle_deg: float) -> bool:
    pa = axis_a.start if endpoint_a == "start" else axis_a.end
    pb = axis_b.start if endpoint_b == "start" else axis_b.end
    if float(np.linalg.norm(pa - pb)) > merge_endpoint_dist:
        return False

    ta = axis_a.start_tangent if endpoint_a == "start" else axis_a.end_tangent
    tb = axis_b.start_tangent if endpoint_b == "start" else axis_b.end_tangent
    na = float(np.linalg.norm(ta))
    nb = float(np.linalg.norm(tb))
    if na <= 1e-9 or nb <= 1e-9:
        return True

    cos_theta = float(np.clip(abs(float(np.dot(ta, tb))) / (na * nb), 0.0, 1.0))
    angle = float(np.degrees(np.arccos(cos_theta)))
    return angle <= merge_tangent_angle_deg


def _stitch_group_centerlines(group: List[AxisCurveItem]) -> np.ndarray:
    if len(group) == 1:
        return group[0].centerline

    endpoints = np.vstack([np.vstack([axis.start, axis.end]) for axis in group])
    center = endpoints.mean(axis=0)
    x = endpoints - center
    cov = x.T @ x
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = _normalize(eigvecs[:, int(np.argmax(eigvals))])

    ordered: List[Tuple[float, np.ndarray]] = []
    for axis in group:
        cl = axis.centerline
        if len(cl) >= 2 and float(np.dot(cl[-1] - cl[0], direction)) < 0:
            cl = cl[::-1]
        ordered.append((float(np.dot(axis.anchor - center, direction)), cl))

    ordered.sort(key=lambda item: item[0])

    stitched: List[np.ndarray] = [ordered[0][1]]
    for _, cl in ordered[1:]:
        prev_end = stitched[-1][-1]
        if float(np.linalg.norm(prev_end - cl[0])) > float(np.linalg.norm(prev_end - cl[-1])):
            cl = cl[::-1]
        stitched.append(cl)

    merged_cl = np.vstack(stitched)
    merged_cl = _smooth_centerline(merged_cl, 5)
    merged_cl = _densify_centerline(merged_cl)
    return merged_cl


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


def merge_axis_near_components_curve(axes: List[AxisCurveItem], merge_endpoint_dist: float, merge_tangent_angle_deg: float):
    parent, find, union = _union_find(len(axes))
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            if (
                _endpoint_match(axes[i], "start", axes[j], "start", merge_endpoint_dist, merge_tangent_angle_deg)
                or _endpoint_match(axes[i], "start", axes[j], "end", merge_endpoint_dist, merge_tangent_angle_deg)
                or _endpoint_match(axes[i], "end", axes[j], "start", merge_endpoint_dist, merge_tangent_angle_deg)
                or _endpoint_match(axes[i], "end", axes[j], "end", merge_endpoint_dist, merge_tangent_angle_deg)
            ):
                union(i, j)
    groups: Dict[int, List[int]] = {}
    for i in range(len(axes)):
        groups.setdefault(find(i), []).append(i)
    merged, mapping = [], []
    for gid, idxs in enumerate(groups.values(), start=1):
        group = [axes[i] for i in idxs]
        all_pts = np.vstack([a.points for a in group])
        if len(group) == 1:
            merged_axis = group[0]
        else:
            merged_cl = _stitch_group_centerlines(group)
            merged_axis = _axis_curve_from_centerline(f"merged_{gid:03d}", all_pts, merged_cl, np.asarray([]), np.asarray([]))
        merged.append(merged_axis)
        for a in group:
            mapping.append({"merged_name": merged_axis.name, "source_name": a.name})
    return merged, mapping, []


def _write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in columns})


def main() -> None:
    args = parse_args()
    if o3d is None:
        raise ImportError("open3d is required to read/write PCD files. Install it with: pip install open3d")
    if not args.input_dir.exists():
        raise FileNotFoundError(f"input-dir not found: {args.input_dir}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / f"run_curve_prior_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    component_map = _load_component_map(args.component_ifc_map)

    axes, failed_axes = _load_axes(args.input_dir, args, component_map)
    if not axes:
        raise RuntimeError("No valid pcd components produced a curved axis; try --fallback-to-straight or relax slice parameters")

    merged_axes, mapping_rows, _ = merge_axis_near_components_curve(
        axes, merge_endpoint_dist=args.merge_endpoint_dist, merge_tangent_angle_deg=args.merge_tangent_angle_deg
    )

    if v2_0 is not None:
        merged_axes, mapping_rows, removed_axes = v2_0.filter_noise_axes(merged_axes, mapping_rows, min_axis_length=args.min_axis_length, min_axis_points=args.min_axis_points)
    else:
        merged_axes, mapping_rows, removed_axes = merged_axes, mapping_rows, []

    merged_pcd_dir = out_dir / "merged_components_pcd"
    merged_pcd_dir.mkdir(parents=True, exist_ok=True)

    axis_rows: List[Dict[str, object]] = []
    for a in merged_axes:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(a.centerline)
        pcd_path = merged_pcd_dir / f"{a.name}_centerline.pcd"
        o3d.io.write_point_cloud(str(pcd_path), pcd)
        axis_rows.append({
            "name": a.name,
            "point_count": int(a.points.shape[0]),
            "axis_length_m": float(a.length),
            "centerline_point_count": int(len(a.centerline)),
            "center_x": float(a.anchor[0]),
            "center_y": float(a.anchor[1]),
            "center_z": float(a.anchor[2]),
            "pcd_path": str(pcd_path),
        })

    axis_csv = out_dir / "merged_member_curve_axes.csv"
    _write_csv(axis_csv, axis_rows, ["name", "point_count", "axis_length_m", "centerline_point_count", "center_x", "center_y", "center_z", "pcd_path"])

    mapping_csv = out_dir / "merge_mapping.csv"
    _write_csv(mapping_csv, mapping_rows, ["merged_name", "source_name"])

    summary = {
        "input_dir": str(args.input_dir),
        "output_dir": str(out_dir),
        "source_component_count": int(len(axes) + len(failed_axes)),
        "fit_success_component_count": int(len(axes)),
        "fit_failed_component_count": int(len(failed_axes)),
        "merged_component_count": int(len(merged_axes)),
        "files": {"merged_curve_axes_csv": str(axis_csv), "merge_mapping_csv": str(mapping_csv)},
    }

    summary_json = out_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] source components: {summary['source_component_count']}")
    print(f"[OUT] {axis_csv}")
    print(f"[OUT] {mapping_csv}")
    print(f"[OUT] {summary_json}")


if __name__ == "__main__":
    main()
    


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


def _initial_centerline_by_sliding_pca(points: np.ndarray, step: float, window: float, min_pts: int):
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


def _refine_centerline_by_circle_slices(points: np.ndarray, centers: np.ndarray, tangents: np.ndarray, thickness: float, min_pts: int, max_circle_rmse: float):
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


def _fit_curve_axis(name: str, points: np.ndarray, step: float, window: float, slice_thickness: float, min_slice_pts: int, smooth_k: int, refine_iterations: int, max_circle_rmse: float, fallback_to_straight: bool) -> Optional[AxisCurveItem]:
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
        refined, radii, rmses = _refine_centerline_by_circle_slices(points, centers, tangents, thickness=max(slice_thickness, 1e-4), min_pts=max(5, int(min_slice_pts)), max_circle_rmse=max_circle_rmse)
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


def _load_component_map(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if path is None or not path.exists():
        return {}
    data: Dict[str, Dict[str, str]] = {}
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            arr = json.load(f)
        if isinstance(arr, list):
            for r in arr:
                name = r.get("component_name") or r.get("component") or r.get("name")
                if name:
                    data[str(name)] = {k: str(v) for k, v in r.items()}
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                name = r.get("component_name") or r.get("name") or r.get("component")
                if name:
                    data[str(name)] = {k: str(v) for k, v in r.items()}
    return data


def _load_axes(input_dir: Path, args: argparse.Namespace, comp_map: Dict[str, Dict[str, str]]) -> Tuple[List[AxisCurveItem], List[Dict[str, object]]]:
    axes: List[AxisCurveItem] = []
    failed: List[Dict[str, object]] = []
    ascii_dir = input_dir.parent / "_ascii_alias_cache"
    ascii_dir.mkdir(parents=True, exist_ok=True)

    # First pass: load all components and try initial fits
    component_data: List[Dict[str, Any]] = []
    for idx, p in enumerate(sorted(input_dir.glob("*.pcd")), start=1):
        alias = ascii_dir / f"comp_{idx:04d}.pcd"
        if (not alias.exists()) or (alias.stat().st_size != p.stat().st_size):
            shutil.copyfile(p, alias)
        pcd = o3d.io.read_point_cloud(str(alias))
        pts = np.asarray(pcd.points)
        
        component_data.append({
            "name": p.stem,
            "points": pts,
            "pcd_path": p,
            "alias_path": alias,
            "fit_result": None,  # Will be set after fitting
            "failed": False,
            "fail_reason": None,
        })

    # Second pass: try to fit each component
    for data in component_data:
        pts = data["points"]
        if pts.size == 0:
            data["failed"] = True
            data["fail_reason"] = "empty_pcd"
            continue

        axis = _fit_curve_axis(
            name=data["name"],
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
            data["failed"] = True
            data["fail_reason"] = "curve_axis_fit_failed"
            continue

        # Apply IFC prior correction
        meta = comp_map.get(data["name"])
        prior_shape = (meta.get("design_shape") if meta else None) if meta is not None else None
        if prior_shape is None and meta is not None:
            prior_shape = meta.get("design_shape")
        if prior_shape is not None and prior_shape.lower() == "straight":
            _, _, start, end, _ = _fit_straight_axis(pts)
            cl = np.vstack([start, end])
            axis = _axis_curve_from_centerline(data["name"], pts, cl, np.asarray([]), np.asarray([]))

        data["fit_result"] = axis

    # Third pass: merge failed small components into nearest successful ones
    successful_data = [d for d in component_data if d["fit_result"] is not None]
    failed_data = [d for d in component_data if d["fit_result"] is None]

    for fail_data in failed_data:
        if fail_data["points"].size == 0:
            failed.append({"name": fail_data["name"], "reason": fail_data["fail_reason"]})
            continue

        # Find the nearest successful component
        fail_pts = fail_data["points"]
        fail_centroid = fail_pts.mean(axis=0)
        
        nearest_success = None
        min_dist = float("inf")
        for succ_data in successful_data:
            succ_pts = succ_data["fit_result"].points
            succ_centroid = succ_pts.mean(axis=0)
            dist = float(np.linalg.norm(fail_centroid - succ_centroid))
            if dist < min_dist:
                min_dist = dist
                nearest_success = succ_data

        if nearest_success is not None:
            # Merge failed component's points into the nearest successful one
            merged_pts = np.vstack([nearest_success["fit_result"].points, fail_pts])
            
            # Re-fit curve on merged points
            merged_axis = _fit_curve_axis(
                name=nearest_success["fit_result"].name,  # Keep the name of the successful component
                points=merged_pts,
                step=args.curve_step,
                window=args.curve_window,
                slice_thickness=args.slice_thickness,
                min_slice_pts=args.min_slice_points,
                smooth_k=args.smooth_k,
                refine_iterations=args.refine_iterations,
                max_circle_rmse=args.max_circle_rmse,
                fallback_to_straight=args.fallback_to_straight,
            )
            
            if merged_axis is not None:
                nearest_success["fit_result"] = merged_axis
                print(f"[MERGE] Attached {fail_data['name']} ({len(fail_pts)} pts) to {nearest_success['fit_result'].name}")
            else:
                # If re-fit fails, just keep the original
                print(f"[WARN] Could not re-fit {nearest_success['fit_result'].name} after adding {fail_data['name']}, keeping original")
                failed.append({"name": fail_data["name"], "reason": "absorbed_but_refit_failed"})
        else:
            failed.append({"name": fail_data["name"], "reason": fail_data["fail_reason"]})

    # Collect all successful axes
    for data in component_data:
        if data["fit_result"] is not None:
            axes.append(data["fit_result"])

    return axes, failed


def _write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in columns})


def _to_bool(v: str) -> bool:
    """Convert string to boolean."""
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y", "t"}


def _load_centerline_points_from_csv(csv_path: Path) -> np.ndarray:
    """Load centerline points from merged_member_curve_axes.csv using centerline_points stored inline."""
    pts: List[np.ndarray] = []
    # Try to load from curve_centerline_points.csv if it exists
    if csv_path.exists():
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pt = np.array(
                        [float(row["x"]), float(row["y"]), float(row["z"])],
                        dtype=np.float64,
                    )
                    pts.append(pt)
        except Exception:
            pass
    if not pts:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)


def _load_axes_from_centerline_pcds(centerline_pcd_dir: Path, sample_count: int = 500) -> np.ndarray:
    """Load and sample points from all centerline PCD files."""
    all_pts: List[np.ndarray] = []
    if not centerline_pcd_dir.exists():
        return np.zeros((0, 3), dtype=np.float64)
    
    for pcd_file in sorted(centerline_pcd_dir.glob("*.pcd")):
        try:
            pcd = o3d.io.read_point_cloud(str(pcd_file))
            pts = np.asarray(pcd.points)
            if len(pts) == 0:
                continue
            # Sample evenly from centerline
            if len(pts) > sample_count:
                indices = np.linspace(0, len(pts) - 1, sample_count, dtype=int)
                pts = pts[indices]
            all_pts.append(pts)
        except Exception:
            pass
    
    if not all_pts:
        return np.zeros((0, 3), dtype=np.float64)
    return np.vstack(all_pts)


def _load_nodes(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load nodes from nodes_from_curve_axes.csv."""
    xyz: List[List[float]] = []
    expected: List[bool] = []
    if not csv_path.exists():
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=bool)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xyz.append([float(row["x"]), float(row["y"]), float(row["z"])])
            expected.append(_to_bool(row.get("is_intersection_expected", "")))
    if not xyz:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=bool)
    return np.asarray(xyz, dtype=np.float64), np.asarray(expected, dtype=bool)


def _save_colored_pcd(path: Path, points: np.ndarray, color: np.ndarray) -> None:
    """Save points as colored PLY."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.tile(color[None, :], (points.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd)


def _generate_visualization(out_dir: Path, merged_axes: List[AxisCurveItem], nodes_rows: List[Dict[str, object]]) -> None:
    """Generate PLY and PNG visualizations from axis data and nodes."""
    if o3d is None:
        print("[WARN] open3d not available, skipping visualization")
        return
    
    # Prepare visualization subdirectory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_dir = out_dir / f"visualization_{ts}"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract centerline points from merged axes
    centerline_pts_list: List[np.ndarray] = []
    for axis in merged_axes:
        if axis.centerline is not None and len(axis.centerline) > 0:
            centerline_pts_list.append(axis.centerline)
    
    if centerline_pts_list:
        centerline_pts = np.vstack(centerline_pts_list)
    else:
        centerline_pts = np.zeros((0, 3), dtype=np.float64)
    
    # Extract node points
    node_pts: List[List[float]] = []
    node_expected: List[bool] = []
    for row in nodes_rows:
        try:
            node_pts.append([float(row["x"]), float(row["y"]), float(row["z"])])
            expected = _to_bool(str(row.get("is_intersection_expected", "")))
            node_expected.append(expected)
        except (ValueError, KeyError):
            pass
    
    if node_pts:
        node_pts_arr = np.asarray(node_pts, dtype=np.float64)
        node_expected_arr = np.asarray(node_expected, dtype=bool)
    else:
        node_pts_arr = np.zeros((0, 3), dtype=np.float64)
        node_expected_arr = np.zeros((0,), dtype=bool)
    
    # 1) Centerline-only cloud (blue)
    centerline_only_ply = vis_dir / "centerline_points_blue.ply"
    _save_colored_pcd(centerline_only_ply, centerline_pts, np.array([0.15, 0.45, 0.95], dtype=np.float64))
    
    # 2) Expected nodes only (red)
    expected_only_ply = vis_dir / "nodes_expected_red.ply"
    if node_pts_arr.shape[0] > 0:
        expected_pts = node_pts_arr[node_expected_arr]
    else:
        expected_pts = np.zeros((0, 3), dtype=np.float64)
    _save_colored_pcd(expected_only_ply, expected_pts, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    
    # 3) Combined overlay: centerline blue + expected nodes red
    combined_ply = vis_dir / "centerline_and_nodes_overlay.ply"
    overlay_pts = []
    overlay_cols = []
    if centerline_pts.shape[0] > 0:
        overlay_pts.append(centerline_pts)
        overlay_cols.append(np.tile(np.array([[0.15, 0.45, 0.95]], dtype=np.float64), (centerline_pts.shape[0], 1)))
    if expected_pts.shape[0] > 0:
        overlay_pts.append(expected_pts)
        overlay_cols.append(np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float64), (expected_pts.shape[0], 1)))
    
    if overlay_pts:
        pts = np.vstack(overlay_pts)
        c = np.vstack(overlay_cols)
    else:
        pts = np.zeros((0, 3), dtype=np.float64)
        c = np.zeros((0, 3), dtype=np.float64)
    
    overlay = o3d.geometry.PointCloud()
    overlay.points = o3d.utility.Vector3dVector(pts)
    overlay.colors = o3d.utility.Vector3dVector(c)
    o3d.io.write_point_cloud(str(combined_ply), overlay)
    
    # 4) Matplotlib 3D PNG
    matplot_png = ""
    if plt is not None:
        try:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            
            if centerline_pts.shape[0] > 0:
                ax.scatter(centerline_pts[:, 0], centerline_pts[:, 1], centerline_pts[:, 2], 
                          s=1, c="#2673f2", alpha=0.55, label="centerline")
            
            if expected_pts.shape[0] > 0:
                ax.scatter(expected_pts[:, 0], expected_pts[:, 1], expected_pts[:, 2], 
                          s=22, c="#e31a1c", label="nodes_expected")
            
            ax.set_title("Curved Axis Centerline and Fitted Nodes")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.legend(loc="best")
            fig.tight_layout()
            
            matplot_png_path = vis_dir / "centerline_and_nodes_matplot.png"
            fig.savefig(matplot_png_path, dpi=220)
            matplot_png = str(matplot_png_path)
            plt.close(fig)
        except Exception as e:
            print(f"[WARN] matplotlib visualization failed: {e}")
    
    # Print outputs
    print(f"[OUT] {centerline_only_ply}")
    print(f"[OUT] {expected_only_ply}")
    print(f"[OUT] {combined_ply}")
    if matplot_png:
        print(f"[OUT] {matplot_png}")
    print(f"[VIS] visualization saved to: {vis_dir}")


def main() -> None:
    args = parse_args()
    if o3d is None:
        raise ImportError("open3d is required to read/write PCD files. Install it with: pip install open3d")
    if not args.input_dir.exists():
        raise FileNotFoundError(f"input-dir not found: {args.input_dir}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / f"run_curve_prior_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    comp_map = _load_component_map(args.component_ifc_map)
    axes, failed_axes = _load_axes(args.input_dir, args, comp_map)
    if not axes:
        raise RuntimeError("No valid pcd components produced a curved axis; try --fallback-to-straight or relax slice parameters")

    # Print merge diagnostics before merging
    _print_merge_diagnostics(axes, args.merge_endpoint_dist, args.merge_tangent_angle_deg)

    merged_axes, mapping_rows, merge_checks = merge_axis_near_components_curve(axes, merge_endpoint_dist=args.merge_endpoint_dist, merge_tangent_angle_deg=args.merge_tangent_angle_deg)

    if v2_0 is not None:
        merged_axes, mapping_rows, removed_axes = v2_0.filter_noise_axes(merged_axes, mapping_rows, min_axis_length=args.min_axis_length, min_axis_points=args.min_axis_points)
    else:
        merged_axes, mapping_rows, removed_axes = merged_axes, mapping_rows, []
    
    # Attempt to recover small removed axes by merging them into nearest neighbor
    if removed_axes and merged_axes:
        recovered_count = 0
        for removed in removed_axes:
            removed_name = removed.get("name", "")
            if not removed_name:
                continue
            
            # Find the original axis object by searching the pre-filter list
            # Since we don't have direct access, we'll need to find by name from mapping
            # For now, we skip this as it's complex; users should adjust thresholds if needed
            print(f"[NOTE] Removed axis {removed_name} (reason: {removed.get('remove_reason', 'unknown')}); "
                  f"consider adjusting --min-axis-length or --min-axis-points to retain")
    
    if not merged_axes:
        raise RuntimeError("All merged axes were removed by noise filters; relax --min-axis-length/--min-axis-points")

    merged_pcd_dir = out_dir / "merged_components_pcd"
    merged_pcd_dir.mkdir(parents=True, exist_ok=True)
    centerline_pcd_dir = out_dir / "centerline_pcd"
    centerline_pcd_dir.mkdir(parents=True, exist_ok=True)
    for a in merged_axes:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(a.centerline)
        o3d.io.write_point_cloud(str(centerline_pcd_dir / f"{a.name}_centerline.pcd"), pcd)

    axis_rows: List[Dict[str, object]] = []
    for a in merged_axes:
        axis_rows.append({
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
            "chord_dir_x": float(a.direction[0]),
            "chord_dir_y": float(a.direction[1]),
            "chord_dir_z": float(a.direction[2]),
            "pcd_path": "",
            "centerline_pcd_path": str(centerline_pcd_dir / f"{a.name}_centerline.pcd"),
        })

    axis_csv = out_dir / "merged_member_curve_axes.csv"
    axis_columns = [
        "name", "point_count", "axis_length_m", "centerline_point_count", "radius_median_m", "circle_rmse_median_m",
        "center_x", "center_y", "center_z", "start_x", "start_y", "start_z", "end_x", "end_y", "end_z",
        "chord_dir_x", "chord_dir_y", "chord_dir_z", "pcd_path", "centerline_pcd_path",
    ]
    _write_csv(axis_csv, axis_rows, axis_columns)

    mapping_csv = out_dir / "merge_mapping.csv"
    _write_csv(mapping_csv, mapping_rows, ["merged_name", "source_name"])

    removed_axes_csv = out_dir / "removed_noise_axes.csv"
    _write_csv(removed_axes_csv, removed_axes, ["name", "point_count", "axis_length_m", "centerline_point_count", "remove_reason"])

    failed_axes_csv = out_dir / "failed_axis_fits.csv"
    _write_csv(failed_axes_csv, failed_axes, ["name", "reason"])

    if v2_0 is not None:
        nodes_rows = v2_0.solve_nodes_from_curve_endpoints(merged_axes, grouping_radius=args.node_grouping_radius, multi_min_members=args.multi_min_members)
    else:
        nodes_rows = []
    nodes_csv = out_dir / "nodes_from_curve_axes.csv"
    node_columns = ["node_id", "x", "y", "z", "distance", "max_line_distance", "is_intersection_expected", "solve_method", "involved_components", "member_count", "involved_endpoints"]
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
            "merge_mapping_csv": str(mapping_csv),
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

    # Generate visualization outputs (PLY + PNG)
    _generate_visualization(out_dir, merged_axes, nodes_rows)

    print(f"[DONE] source components: {summary['source_component_count']}")
    print(f"[DONE] curve axis fit success: {len(axes)}, failed: {len(failed_axes)}")
    print(f"[DONE] merged components: {len(merged_axes)}")
    print(f"[DONE] removed noise axes: {len(removed_axes)}")
    print(f"[DONE] nodes: {len(nodes_rows)}")
    print(f"[OUT] {axis_csv}")
    print(f"[OUT] {mapping_csv}")
    print(f"[OUT] {nodes_csv}")
    print(f"[OUT] {summary_json}")
    if v2_0 is not None:
        print(f"[NOTE] Result root (big folder under workspace): {v2_0._resolve_ifc2mesh_root(out_dir)}")
    else:
        print(f"[NOTE] Result root (big folder under workspace): {out_dir.parent}")


if __name__ == "__main__":
    main()
