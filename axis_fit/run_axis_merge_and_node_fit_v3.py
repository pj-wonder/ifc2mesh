from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d

try:
    import ifcopenshell
except ImportError:  # pragma: no cover
    ifcopenshell = None

try:
    from axis_fit import run_axis_merge_and_node_fit_curve_v2 as curve_v2
except ImportError:  # pragma: no cover
    curve_v2 = None


@dataclass
class AxisItem:
    name: str
    points: np.ndarray
    anchor: np.ndarray
    direction: np.ndarray
    start: np.ndarray
    end: np.ndarray
    length: float
    design_shape: str = "unknown"
    fit_mode: str = "auto_mode"
    ifc_global_id: str = ""
    ifc_name: str = ""
    ifc_type: str = ""
    component_pcd: str = ""


@dataclass
class ComponentIfcMeta:
    component_name: str
    ifc_index: int
    ifc_global_id: str
    ifc_name: str
    ifc_type: str
    design_shape: str
    source_type: str
    mesh_file: str = ""
    segmented_pcd: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge axis-near components and fit junction nodes")
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
    parser.add_argument("--ifc-path", type=Path, default=None, help="Optional IFC file used as design prior")
    parser.add_argument(
        "--use-ifc-prior",
        action="store_true",
        help="Use IFC metadata to choose straight/curved/polyline/auto processing mode",
    )
    parser.add_argument(
        "--pcd-ifc-map",
        type=Path,
        default=None,
        help="Optional component_label_map.json or .csv that links segmented PCD files to IFC metadata",
    )
    parser.add_argument("--curve-step", type=float, default=0.05, help="Centerline sampling step for curved mode (m)")
    parser.add_argument("--curve-window", type=float, default=0.25, help="Local PCA window length for curved mode (m)")
    parser.add_argument("--slice-thickness", type=float, default=0.06, help="Adaptive local slice thickness (m)")
    parser.add_argument("--min-slice-points", type=int, default=80, help="Minimum points in a local slice")
    parser.add_argument("--smooth-k", type=int, default=5, help="Moving-average window for centerline smoothing")
    parser.add_argument("--refine-iterations", type=int, default=2, help="Circle-slice refinement iterations")
    parser.add_argument("--max-circle-rmse", type=float, default=0.04, help="Reject a fitted section if circle RMSE exceeds this value (m)")
    parser.add_argument(
        "--straight-endpoint-trim-ratio",
        type=float,
        default=0.03,
        help="Trim ratio on both ends when fitting straight axes to suppress endpoint outliers",
    )
    parser.add_argument(
        "--auto-linearity-threshold",
        type=float,
        default=0.02,
        help="Normalized RMS threshold for deciding whether an unknown component stays straight or switches to curve mode",
    )
    parser.add_argument(
        "--fallback-to-straight",
        action="store_true",
        help="If curved extraction fails, use straight PCA axis as fallback instead of skipping the component",
    )
    parser.add_argument("--merge-angle-deg", type=float, default=6.0, help="Max axis angle for merging")
    parser.add_argument("--merge-line-dist", type=float, default=0.08, help="Max line distance for merging (m)")
    parser.add_argument("--merge-gap", type=float, default=0.35, help="Max segment gap for merging (m)")
    parser.add_argument("--node-distance-threshold", type=float, default=0.25, help="Node expected threshold (m)")
    parser.add_argument("--segment-margin", type=float, default=0.5, help="Projection margin for on-segment check")
    parser.add_argument(
        "--node-grouping-radius",
        type=float,
        default=0.25,
        help="Radius for clustering expected pairwise nodes into one multi-member junction (m)",
    )
    parser.add_argument(
        "--multi-min-members",
        type=int,
        default=3,
        help="Minimum unique members in a cluster to trigger global least-squares node solve",
    )
    parser.add_argument(
        "--min-axis-length",
        type=float,
        default=0.8,
        help="Drop merged axes shorter than this length (m)",
    )
    parser.add_argument(
        "--min-axis-points",
        type=int,
        default=1000,
        help="Drop merged axes with fewer points than this threshold",
    )
    return parser.parse_args()


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v.copy()
    return v / n


def _linearity_ratio(points: np.ndarray, trim_ratio: float = 0.03) -> float:
    if len(points) < 3:
        return 0.0
    center = points.mean(axis=0)
    x = points - center
    cov = x.T @ x
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = _normalize(eigvecs[:, int(np.argmax(eigvals))])
    t = x @ direction
    trim_ratio = float(np.clip(trim_ratio, 0.0, 0.25))
    if trim_ratio > 0.0 and len(t) >= 10:
        lo = float(np.quantile(t, trim_ratio))
        hi = float(np.quantile(t, 1.0 - trim_ratio))
    else:
        lo = float(np.min(t))
        hi = float(np.max(t))
    length = max(hi - lo, 1e-12)
    closest = center + np.outer(t, direction)
    residual = np.linalg.norm(points - closest, axis=1)
    return float(np.sqrt(np.mean(residual ** 2)) / length)


def _safe_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _parse_component_stem(component_name: str) -> Dict[str, str]:
    parts = component_name.split("_")
    if len(parts) < 3:
        return {"ifc_index": "", "ifc_type": "", "ifc_name": "", "ifc_global_id": ""}
    ifc_index = parts[0]
    ifc_type = parts[1]
    ifc_global_id = parts[-1]
    ifc_name = "_".join(parts[2:-1]) if len(parts) > 3 else ""
    return {
        "ifc_index": ifc_index,
        "ifc_type": ifc_type,
        "ifc_name": ifc_name,
        "ifc_global_id": ifc_global_id,
    }


def _resolve_ifc_guid(ifc_guid: str, ifc_lookup: Dict[str, Dict[str, Any]]) -> str:
    if not ifc_guid:
        return ""
    if ifc_guid in ifc_lookup:
        return ifc_guid
    matches = [guid for guid in ifc_lookup.keys() if guid.startswith(ifc_guid)]
    if len(matches) == 1:
        return matches[0]
    return ifc_guid


def _resolve_component_map_path(input_dir: Path, explicit_map: Optional[Path]) -> Optional[Path]:
    if explicit_map is not None and explicit_map.exists():
        return explicit_map

    result_root = input_dir.parent.parent if input_dir.parent.name == "label_transfer_50mm" else input_dir.parent
    candidates = sorted(result_root.glob("subresult_bbox_split_v2_*/component_label_map.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]

    csv_candidates = sorted(result_root.glob("subresult_bbox_split_v2_*/component_label_map.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if csv_candidates:
        return csv_candidates[0]

    return None


def _load_component_ifc_map(map_path: Optional[Path]) -> Dict[str, ComponentIfcMeta]:
    if map_path is None or not map_path.exists():
        return {}

    rows: List[Dict[str, Any]] = []
    if map_path.suffix.lower() == ".json":
        with map_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            rows = [r for r in data if isinstance(r, dict)]
    else:
        with map_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = [dict(r) for r in reader]

    component_map: Dict[str, ComponentIfcMeta] = {}
    for row in rows:
        segmented_pcd = _safe_text(row.get("segmented_pcd") or row.get("pcd_path") or row.get("component_pcd"))
        component_name = Path(segmented_pcd).stem if segmented_pcd else _safe_text(row.get("component_name") or row.get("name"))
        if not component_name:
            continue

        meta = ComponentIfcMeta(
            component_name=component_name,
            ifc_index=int(row.get("ifc_index") or row.get("index") or 0),
            ifc_global_id=_safe_text(row.get("guid") or row.get("ifc_global_id") or row.get("GlobalId")),
            ifc_name=_safe_text(row.get("name") or row.get("ifc_name")),
            ifc_type=_safe_text(row.get("ifc_type") or row.get("type")),
            design_shape="unknown",
            source_type=_safe_text(row.get("source_type") or row.get("segment_source_type") or "unknown"),
            mesh_file=_safe_text(row.get("mesh_file")),
            segmented_pcd=segmented_pcd,
        )
        component_map[component_name] = meta
        if meta.ifc_global_id:
            component_map[meta.ifc_global_id] = meta

    return component_map


def _normalize_ifc_shape_type(raw_type: str) -> str:
    text = raw_type.lower()
    if text in {"straight", "curved", "polyline", "unknown"}:
        return text
    return "unknown"


def _classify_ifc_axis_item(item: Any) -> str:
    if item is None:
        return "unknown"

    if item.is_a("IfcPolyline"):
        points = getattr(item, "Points", None) or []
        if len(points) <= 2:
            return "straight"
        return "polyline"

    if item.is_a("IfcIndexedPolyCurve"):
        points = getattr(getattr(item, "Points", None), "CoordList", None) or []
        if len(points) <= 2:
            return "straight"
        return "polyline"

    if item.is_a("IfcCompositeCurve"):
        segments = getattr(item, "Segments", None) or []
        if not segments:
            return "unknown"
        parent_types = []
        for seg in segments:
            parent = getattr(seg, "ParentCurve", None)
            parent_types.append(getattr(parent, "is_a", lambda: "")())
        if any(pt not in {"IfcLine", "IfcPolyline", "IfcIndexedPolyCurve"} for pt in parent_types):
            return "curved"
        return "polyline" if len(segments) > 1 else "straight"

    return "unknown"


def _extract_ifc_design_shape(element: Any) -> Tuple[str, str]:
    representation = getattr(element, "Representation", None)
    if representation:
        for rep in getattr(representation, "Representations", []) or []:
            if getattr(rep, "RepresentationIdentifier", None) != "Axis":
                continue
            items = getattr(rep, "Items", []) or []
            for item in items:
                shape = _classify_ifc_axis_item(item)
                if shape != "unknown":
                    return shape, "Axis"

    if getattr(element, "ObjectPlacement", None) is not None:
        return "straight", "ObjectPlacement"

    return "unknown", "unknown"


def _build_ifc_lookup(ifc_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if ifc_path is None or not ifc_path.exists():
        return {}
    if ifcopenshell is None:
        raise ImportError("ifcopenshell is required when --use-ifc-prior is enabled")

    model = ifcopenshell.open(str(ifc_path))
    lookup: Dict[str, Dict[str, Any]] = {}
    for element in (model.by_type("IfcBeam") + model.by_type("IfcColumn") + model.by_type("IfcMember")):
        guid = _safe_text(getattr(element, "GlobalId", ""))
        if not guid:
            continue
        design_shape, source_type = _extract_ifc_design_shape(element)
        lookup[guid] = {
            "guid": guid,
            "ifc_name": _safe_text(getattr(element, "Name", "")),
            "ifc_type": _safe_text(getattr(element, "is_a", lambda: "")()),
            "design_shape": _normalize_ifc_shape_type(design_shape),
            "source_type": source_type,
        }
    return lookup


def _fit_axis_from_curve_mode(
    points: np.ndarray,
    curve_step: float,
    curve_window: float,
    slice_thickness: float,
    min_slice_points: int,
    smooth_k: int,
    refine_iterations: int,
    max_circle_rmse: float,
    fallback_to_straight: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if curve_v2 is None:
        return _fit_axis(points)

    curve_axis = curve_v2._fit_curve_axis(
        name="tmp",
        points=points,
        step=curve_step,
        window=curve_window,
        slice_thickness=slice_thickness,
        min_slice_pts=min_slice_points,
        smooth_k=smooth_k,
        refine_iterations=refine_iterations,
        max_circle_rmse=max_circle_rmse,
        fallback_to_straight=fallback_to_straight,
    )
    if curve_axis is None or len(curve_axis.centerline) < 2:
        return _fit_axis(points)

    start = np.asarray(curve_axis.start, dtype=float)
    end = np.asarray(curve_axis.end, dtype=float)
    anchor = np.asarray(curve_axis.anchor, dtype=float)
    direction = np.asarray(curve_axis.direction, dtype=float)
    length = float(curve_axis.length)
    return anchor, direction, start, end, length


def _choose_fit_mode(design_shape: str, use_ifc_prior: bool, points: np.ndarray, auto_linearity_threshold: float) -> str:
    if not use_ifc_prior:
        return "straight_mode" if _linearity_ratio(points) <= auto_linearity_threshold else "curved_mode"
    if design_shape == "straight":
        return "straight_mode"
    if design_shape == "curved":
        return "curved_mode"
    if design_shape == "polyline":
        return "polyline_mode"
    return "straight_mode" if _linearity_ratio(points) <= auto_linearity_threshold else "curved_mode"


def _combine_design_shapes(axes: List[AxisItem]) -> str:
    if not axes:
        return "unknown"
    priority = {"unknown": 0, "straight": 1, "polyline": 2, "curved": 3}
    return max((a.design_shape for a in axes), key=lambda shape: priority.get(shape, 0))


def _combine_fit_modes(axes: List[AxisItem]) -> str:
    if not axes:
        return "auto_mode"
    priority = {"auto_mode": 0, "straight_mode": 1, "polyline_mode": 2, "curved_mode": 3}
    return max((a.fit_mode for a in axes), key=lambda mode: priority.get(mode, 0))


def _fit_axis_with_mode(
    points: np.ndarray,
    design_shape: str,
    use_ifc_prior: bool,
    curve_step: float,
    curve_window: float,
    slice_thickness: float,
    min_slice_points: int,
    smooth_k: int,
    refine_iterations: int,
    max_circle_rmse: float,
    fallback_to_straight: bool,
    straight_endpoint_trim_ratio: float,
    auto_linearity_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, str]:
    fit_mode = _choose_fit_mode(design_shape, use_ifc_prior, points, auto_linearity_threshold)
    if fit_mode == "straight_mode":
        anchor, direction, start, end, length = _fit_axis(points, trim_ratio=straight_endpoint_trim_ratio)
        return anchor, direction, start, end, length, fit_mode
    if fit_mode in {"curved_mode", "polyline_mode"}:
        anchor, direction, start, end, length = _fit_axis_from_curve_mode(
            points,
            curve_step=curve_step,
            curve_window=curve_window,
            slice_thickness=slice_thickness,
            min_slice_points=min_slice_points,
            smooth_k=smooth_k,
            refine_iterations=refine_iterations,
            max_circle_rmse=max_circle_rmse,
            fallback_to_straight=fallback_to_straight,
        )
        return anchor, direction, start, end, length, fit_mode
    anchor, direction, start, end, length = _fit_axis(points, trim_ratio=straight_endpoint_trim_ratio)
    return anchor, direction, start, end, length, fit_mode


def _fit_axis(points: np.ndarray, trim_ratio: float = 0.03) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    center = points.mean(axis=0)
    x = points - center
    cov = x.T @ x
    eigvals, eigvecs = np.linalg.eigh(cov)
    d = eigvecs[:, int(np.argmax(eigvals))]
    d = _normalize(d)

    t = x @ d
    trim_ratio = float(np.clip(trim_ratio, 0.0, 0.25))
    if trim_ratio > 0.0 and len(t) >= 10:
        t_min = float(np.quantile(t, trim_ratio))
        t_max = float(np.quantile(t, 1.0 - trim_ratio))
    else:
        t_min = float(np.min(t))
        t_max = float(np.max(t))
    start = center + t_min * d
    end = center + t_max * d
    length = float(np.linalg.norm(end - start))
    return center, d, start, end, length


def _line_line_closest_points(p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray):
    d1 = _normalize(d1)
    d2 = _normalize(d2)
    w0 = p1 - p2
    a = float(np.dot(d1, d1))
    b = float(np.dot(d1, d2))
    c = float(np.dot(d2, d2))
    d = float(np.dot(d1, w0))
    e = float(np.dot(d2, w0))
    den = a * c - b * b

    if abs(den) < 1e-12:
        t1 = 0.0
        t2 = 0.0
        q1 = p1
        q2 = p2
    else:
        t1 = float((b * e - c * d) / den)
        t2 = float((a * e - b * d) / den)
        q1 = p1 + t1 * d1
        q2 = p2 + t2 * d2

    dist = float(np.linalg.norm(q1 - q2))
    return q1, q2, dist


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


def _point_on_segment(point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray, margin: float) -> bool:
    seg = seg_end - seg_start
    length = float(np.linalg.norm(seg))
    if length < 1e-10:
        return False
    direction = seg / length
    s = float(np.dot(point - seg_start, direction))
    return (-margin <= s <= length + margin)


def _segment_gap_on_dir(a: AxisItem, b: AxisItem) -> float:
    d = a.direction
    if float(np.dot(d, b.direction)) < 0:
        d = -d

    a0 = float(np.dot(a.start - a.anchor, d))
    a1 = float(np.dot(a.end - a.anchor, d))
    b0 = float(np.dot(b.start - a.anchor, d))
    b1 = float(np.dot(b.end - a.anchor, d))

    lo1, hi1 = (a0, a1) if a0 <= a1 else (a1, a0)
    lo2, hi2 = (b0, b1) if b0 <= b1 else (b1, b0)

    if hi1 < lo2:
        return float(lo2 - hi1)
    if hi2 < lo1:
        return float(lo1 - hi2)
    return 0.0


def _load_axes(
    input_dir: Path,
    component_map: Dict[str, ComponentIfcMeta],
    ifc_lookup: Dict[str, Dict[str, Any]],
    use_ifc_prior: bool,
    args: argparse.Namespace,
) -> Tuple[List[AxisItem], List[Dict[str, object]]]:
    axes: List[AxisItem] = []
    component_rows: List[Dict[str, object]] = []
    ascii_dir = input_dir.parent / "_ascii_alias_cache"
    ascii_dir.mkdir(parents=True, exist_ok=True)

    for idx, p in enumerate(sorted(input_dir.glob("*.pcd")), start=1):
        alias = ascii_dir / f"comp_{idx:04d}.pcd"
        if (not alias.exists()) or (alias.stat().st_size != p.stat().st_size):
            shutil.copyfile(p, alias)

        pcd = o3d.io.read_point_cloud(str(alias))
        pts = np.asarray(pcd.points)
        if pts.size == 0:
            continue
        meta = component_map.get(p.stem)
        stem_info = _parse_component_stem(p.stem)
        ifc_guid = meta.ifc_global_id if meta and meta.ifc_global_id else stem_info["ifc_global_id"]
        ifc_guid = _resolve_ifc_guid(ifc_guid, ifc_lookup)
        ifc_meta = ifc_lookup.get(ifc_guid) if ifc_guid else None
        design_shape = _normalize_ifc_shape_type(ifc_meta["design_shape"]) if ifc_meta else (meta.design_shape if meta else "unknown")
        if use_ifc_prior and design_shape == "unknown" and meta is not None:
            design_shape = _normalize_ifc_shape_type(meta.design_shape)

        if ifc_meta is not None:
            ifc_global_id = _safe_text(ifc_meta.get("guid", ifc_guid))
            ifc_name = _safe_text(ifc_meta.get("ifc_name", meta.ifc_name if meta else stem_info["ifc_name"]))
            ifc_type = _safe_text(ifc_meta.get("ifc_type", meta.ifc_type if meta else stem_info["ifc_type"]))
            source_type = _safe_text(ifc_meta.get("source_type", meta.source_type if meta else "unknown"))
        elif meta is not None:
            ifc_global_id = meta.ifc_global_id or ifc_guid
            ifc_name = meta.ifc_name or stem_info["ifc_name"]
            ifc_type = meta.ifc_type or stem_info["ifc_type"]
            source_type = meta.source_type
        else:
            ifc_global_id = ifc_guid
            ifc_name = stem_info["ifc_name"]
            ifc_type = stem_info["ifc_type"]
            source_type = "unknown"

        anchor, direction, start, end, length, fit_mode = _fit_axis_with_mode(
            pts,
            design_shape,
            use_ifc_prior,
            curve_step=float(args.curve_step),
            curve_window=float(args.curve_window),
            slice_thickness=float(args.slice_thickness),
            min_slice_points=int(args.min_slice_points),
            smooth_k=int(args.smooth_k),
            refine_iterations=int(args.refine_iterations),
            max_circle_rmse=float(args.max_circle_rmse),
            fallback_to_straight=bool(args.fallback_to_straight),
            straight_endpoint_trim_ratio=float(args.straight_endpoint_trim_ratio),
            auto_linearity_threshold=float(args.auto_linearity_threshold),
        )
        axes.append(
            AxisItem(
                name=p.stem,
                points=pts,
                anchor=anchor,
                direction=direction,
                start=start,
                end=end,
                length=length,
                design_shape=design_shape,
                fit_mode=fit_mode,
                ifc_global_id=ifc_global_id,
                ifc_name=ifc_name,
                ifc_type=ifc_type,
                component_pcd=str(p),
            )
        )
        component_rows.append(
            {
                "component_name": p.stem,
                "ifc_global_id": ifc_global_id,
                "ifc_name": ifc_name,
                "ifc_type": ifc_type,
                "design_shape": design_shape,
                "fit_mode": fit_mode,
                "source_type": source_type,
                "component_pcd": str(p),
                "ifc_prior_enabled": bool(use_ifc_prior),
            }
        )
    return axes, component_rows


def _union_find(n: int) -> Tuple[List[int], callable, callable]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    return parent, find, union


def merge_axis_near_components(
    axes: List[AxisItem],
    merge_angle_deg: float,
    merge_line_dist: float,
    merge_gap: float,
) -> Tuple[List[AxisItem], List[Dict[str, str]]]:
    if not axes:
        return [], []

    parent, find, union = _union_find(len(axes))
    cos_thr = float(np.cos(np.deg2rad(max(merge_angle_deg, 0.0))))

    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            ai = axes[i]
            aj = axes[j]

            axis_sim = abs(float(np.dot(ai.direction, aj.direction)))
            if axis_sim < cos_thr:
                continue

            _, _, line_dist = _line_line_closest_points(ai.anchor, ai.direction, aj.anchor, aj.direction)
            if line_dist > merge_line_dist:
                continue

            seg_gap = _segment_gap_on_dir(ai, aj)
            if seg_gap > merge_gap:
                continue

            union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(len(axes)):
        r = find(i)
        groups.setdefault(r, []).append(i)

    merged_axes: List[AxisItem] = []
    mapping_rows: List[Dict[str, str]] = []

    for gid, idxs in enumerate(groups.values(), start=1):
        names = [axes[i].name for i in idxs]
        all_pts = np.vstack([axes[i].points for i in idxs])
        anchor, direction, start, end, length = _fit_axis(all_pts)
        merged_name = f"merged_{gid:03d}"
        group_axes = [axes[i] for i in idxs]
        merged_design_shape = _combine_design_shapes(group_axes)
        merged_fit_mode = _combine_fit_modes(group_axes)
        merged_ifc_ids = "|".join([a.ifc_global_id for a in group_axes if a.ifc_global_id])
        merged_ifc_names = "|".join([a.ifc_name for a in group_axes if a.ifc_name])
        merged_ifc_types = "|".join([a.ifc_type for a in group_axes if a.ifc_type])
        merged_axes.append(
            AxisItem(
                name=merged_name,
                points=all_pts,
                anchor=anchor,
                direction=direction,
                start=start,
                end=end,
                length=length,
                design_shape=merged_design_shape,
                fit_mode=merged_fit_mode,
                ifc_global_id=merged_ifc_ids,
                ifc_name=merged_ifc_names,
                ifc_type=merged_ifc_types,
            )
        )
        for n in names:
            source_axis = axes[[axis.name for axis in axes].index(n)]
            mapping_rows.append(
                {
                    "merged_name": merged_name,
                    "source_name": n,
                    "source_ifc_global_id": source_axis.ifc_global_id,
                    "source_ifc_name": source_axis.ifc_name,
                    "source_ifc_type": source_axis.ifc_type,
                    "source_design_shape": source_axis.design_shape,
                    "source_fit_mode": source_axis.fit_mode,
                }
            )

    return merged_axes, mapping_rows


def filter_noise_axes(
    axes: List[AxisItem],
    mapping_rows: List[Dict[str, str]],
    min_axis_length: float,
    min_axis_points: int,
) -> Tuple[List[AxisItem], List[Dict[str, str]], List[Dict[str, object]]]:
    kept_axes: List[AxisItem] = []
    removed: List[Dict[str, object]] = []

    for a in axes:
        reasons: List[str] = []
        if float(a.length) < float(min_axis_length):
            reasons.append("short_length")
        if int(a.points.shape[0]) < int(min_axis_points):
            reasons.append("few_points")
        if reasons:
            removed.append(
                {
                    "name": a.name,
                    "point_count": int(a.points.shape[0]),
                    "axis_length_m": float(a.length),
                    "remove_reason": "|".join(reasons),
                }
            )
        else:
            kept_axes.append(a)

    keep_names = {a.name for a in kept_axes}
    kept_mapping = [r for r in mapping_rows if str(r.get("merged_name", "")) in keep_names]
    return kept_axes, kept_mapping, removed


def solve_nodes_pairwise(
    axes: List[AxisItem],
    distance_threshold: float,
    segment_margin: float,
    grouping_radius: float,
    multi_min_members: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    pair_candidates: List[Dict[str, object]] = []
    axis_map = {a.name: a for a in axes}

    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            a = axes[i]
            b = axes[j]
            q1, q2, dist = _line_line_closest_points(a.anchor, a.direction, b.anchor, b.direction)
            node = 0.5 * (q1 + q2)
            on_a = _point_on_segment(node, a.start, a.end, margin=segment_margin)
            on_b = _point_on_segment(node, b.start, b.end, margin=segment_margin)
            expected = bool((dist <= distance_threshold) and on_a and on_b)

            pair_candidates.append(
                {
                    "component_1": a.name,
                    "component_2": b.name,
                    "x": float(node[0]),
                    "y": float(node[1]),
                    "z": float(node[2]),
                    "distance": float(dist),
                    "on_component_1_segment": bool(on_a),
                    "on_component_2_segment": bool(on_b),
                    "is_intersection_expected": expected,
                    "solve_method": "pairwise_midpoint",
                    "involved_components": f"{a.name}|{b.name}",
                    "member_count": 2,
                    "max_line_distance": float(dist),
                    "on_segment_ratio": 1.0 if (on_a and on_b) else 0.5 if (on_a or on_b) else 0.0,
                }
            )

    expected_idx = [idx for idx, r in enumerate(pair_candidates) if bool(r["is_intersection_expected"])]
    if expected_idx:
        exp_pts = np.array(
            [[pair_candidates[i]["x"], pair_candidates[i]["y"], pair_candidates[i]["z"]] for i in expected_idx],
            dtype=float,
        )
        clusters = _cluster_points(exp_pts, radius=grouping_radius)
    else:
        clusters = []

    suppressed_pairs = set()
    final_counter = 0

    for cl in clusters:
        cand_indices = [expected_idx[k] for k in cl]
        members = sorted(
            {
                str(pair_candidates[idx]["component_1"])
                for idx in cand_indices
            }
            |
            {
                str(pair_candidates[idx]["component_2"])
                for idx in cand_indices
            }
        )
        if len(members) < max(3, int(multi_min_members)):
            continue

        anchors = [axis_map[m].anchor for m in members]
        dirs = [axis_map[m].direction for m in members]
        c, mean_dist, max_dist = _solve_global_pseudointersection(np.asarray(anchors), np.asarray(dirs))

        on_flags = [
            _point_on_segment(c, axis_map[m].start, axis_map[m].end, margin=segment_margin)
            for m in members
        ]
        on_ratio = float(np.mean(on_flags)) if on_flags else 0.0
        expected_flag = bool((mean_dist <= distance_threshold) and (on_ratio >= 0.6))

        final_counter += 1
        comp1 = members[0]
        comp2 = members[1] if len(members) > 1 else members[0]
        rows.append(
            {
                "node_id": f"node_mls_{final_counter:04d}",
                "component_1": comp1,
                "component_2": comp2,
                "x": float(c[0]),
                "y": float(c[1]),
                "z": float(c[2]),
                "distance": float(mean_dist),
                "on_component_1_segment": bool(on_flags[0]) if on_flags else False,
                "on_component_2_segment": bool(on_flags[1]) if len(on_flags) > 1 else False,
                "is_intersection_expected": expected_flag,
                "solve_method": "global_least_squares_pinv",
                "involved_components": "|".join(members),
                "member_count": int(len(members)),
                "max_line_distance": float(max_dist),
                "on_segment_ratio": float(on_ratio),
            }
        )
        for idx in cand_indices:
            suppressed_pairs.add(idx)

    for idx, r in enumerate(pair_candidates):
        if idx in suppressed_pairs:
            continue
        final_counter += 1
        row = dict(r)
        row["node_id"] = f"node_pair_{final_counter:04d}"
        rows.append(row)

    return rows


def _write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in columns})


def _resolve_ifc2mesh_root(path_like: Path) -> Path:
    resolved = path_like.resolve()
    parts = list(resolved.parts)
    if "ifc2mesh" in parts:
        idx = parts.index("ifc2mesh")
        return Path(*parts[: idx + 1])
    return resolved.parent


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"input-dir not found: {args.input_dir}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / f"run_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    component_map_path = _resolve_component_map_path(args.input_dir, args.pcd_ifc_map)
    component_map = _load_component_ifc_map(component_map_path)
    ifc_lookup = _build_ifc_lookup(args.ifc_path) if args.use_ifc_prior else {}

    axes, component_rows = _load_axes(args.input_dir, component_map, ifc_lookup, args.use_ifc_prior, args)
    if not axes:
        raise RuntimeError("No valid pcd components found")

    merged_axes, mapping_rows = merge_axis_near_components(
        axes,
        merge_angle_deg=args.merge_angle_deg,
        merge_line_dist=args.merge_line_dist,
        merge_gap=args.merge_gap,
    )

    merged_axes, mapping_rows, removed_axes = filter_noise_axes(
        merged_axes,
        mapping_rows,
        min_axis_length=args.min_axis_length,
        min_axis_points=args.min_axis_points,
    )
    if not merged_axes:
        raise RuntimeError("All merged axes were removed by noise filters; relax --min-axis-length/--min-axis-points")

    component_map_csv = out_dir / "component_ifc_prior_map.csv"
    component_columns = [
        "component_name",
        "ifc_global_id",
        "ifc_name",
        "ifc_type",
        "design_shape",
        "fit_mode",
        "source_type",
        "component_pcd",
        "ifc_prior_enabled",
    ]
    _write_csv(component_map_csv, component_rows, component_columns)

    merged_pcd_dir = out_dir / "merged_components_pcd"
    merged_pcd_dir.mkdir(parents=True, exist_ok=True)

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
                "center_x": float(a.anchor[0]),
                "center_y": float(a.anchor[1]),
                "center_z": float(a.anchor[2]),
                "start_x": float(a.start[0]),
                "start_y": float(a.start[1]),
                "start_z": float(a.start[2]),
                "end_x": float(a.end[0]),
                "end_y": float(a.end[1]),
                "end_z": float(a.end[2]),
                "dir_x": float(a.direction[0]),
                "dir_y": float(a.direction[1]),
                "dir_z": float(a.direction[2]),
                "design_shape": a.design_shape,
                "fit_mode": a.fit_mode,
                "ifc_global_id": a.ifc_global_id,
                "ifc_name": a.ifc_name,
                "ifc_type": a.ifc_type,
                "pcd_path": str(pcd_path),
            }
        )

    axis_csv = out_dir / "merged_member_axes.csv"
    axis_columns = [
        "name",
        "point_count",
        "axis_length_m",
        "center_x",
        "center_y",
        "center_z",
        "start_x",
        "start_y",
        "start_z",
        "end_x",
        "end_y",
        "end_z",
        "dir_x",
        "dir_y",
        "dir_z",
        "design_shape",
        "fit_mode",
        "ifc_global_id",
        "ifc_name",
        "ifc_type",
        "pcd_path",
    ]
    _write_csv(axis_csv, axis_rows, axis_columns)

    mapping_csv = out_dir / "merge_mapping.csv"
    _write_csv(
        mapping_csv,
        mapping_rows,
        [
            "merged_name",
            "source_name",
            "source_ifc_global_id",
            "source_ifc_name",
            "source_ifc_type",
            "source_design_shape",
            "source_fit_mode",
        ],
    )

    removed_axes_csv = out_dir / "removed_noise_axes.csv"
    _write_csv(removed_axes_csv, removed_axes, ["name", "point_count", "axis_length_m", "remove_reason"])

    nodes_df = solve_nodes_pairwise(
        merged_axes,
        distance_threshold=args.node_distance_threshold,
        segment_margin=args.segment_margin,
        grouping_radius=args.node_grouping_radius,
        multi_min_members=args.multi_min_members,
    )
    nodes_csv = out_dir / "nodes_from_merged_axes.csv"
    node_columns = [
        "node_id",
        "component_1",
        "component_2",
        "x",
        "y",
        "z",
        "distance",
        "on_component_1_segment",
        "on_component_2_segment",
        "is_intersection_expected",
        "solve_method",
        "involved_components",
        "member_count",
        "max_line_distance",
        "on_segment_ratio",
    ]
    _write_csv(nodes_csv, nodes_df, node_columns)

    expected_count = int(sum(1 for r in nodes_df if bool(r.get("is_intersection_expected", False))))
    ls_count = int(sum(1 for r in nodes_df if str(r.get("solve_method", "")) == "global_least_squares_pinv"))
    summary = {
        "input_dir": str(args.input_dir),
        "output_dir": str(out_dir),
        "source_component_count": int(len(axes)),
        "merged_component_count": int(len(merged_axes)),
        "merge_angle_deg": float(args.merge_angle_deg),
        "merge_line_dist_m": float(args.merge_line_dist),
        "merge_gap_m": float(args.merge_gap),
        "node_distance_threshold_m": float(args.node_distance_threshold),
        "node_grouping_radius_m": float(args.node_grouping_radius),
        "multi_min_members": int(args.multi_min_members),
        "min_axis_length_m": float(args.min_axis_length),
        "min_axis_points": int(args.min_axis_points),
        "node_count": int(len(nodes_df)),
        "expected_node_count": expected_count,
        "least_squares_node_count": ls_count,
        "removed_noise_axis_count": int(len(removed_axes)),
        "files": {
            "merged_axes_csv": str(axis_csv),
            "merge_mapping_csv": str(mapping_csv),
            "removed_noise_axes_csv": str(removed_axes_csv),
            "nodes_csv": str(nodes_csv),
            "merged_components_pcd_dir": str(merged_pcd_dir),
            "component_ifc_prior_map_csv": str(component_map_csv),
            "input_ifc": str(args.ifc_path) if args.ifc_path else "",
            "component_map_path": str(component_map_path) if component_map_path else "",
        },
        "design_shape_counts": {
            "straight": int(sum(1 for a in axes if a.design_shape == "straight")),
            "curved": int(sum(1 for a in axes if a.design_shape == "curved")),
            "polyline": int(sum(1 for a in axes if a.design_shape == "polyline")),
            "unknown": int(sum(1 for a in axes if a.design_shape == "unknown")),
        },
        "fit_mode_counts": {
            "straight_mode": int(sum(1 for a in axes if a.fit_mode == "straight_mode")),
            "curved_mode": int(sum(1 for a in axes if a.fit_mode == "curved_mode")),
            "polyline_mode": int(sum(1 for a in axes if a.fit_mode == "polyline_mode")),
            "auto_mode": int(sum(1 for a in axes if a.fit_mode == "auto_mode")),
        },
    }

    summary_json = out_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] source components: {len(axes)}")
    print(f"[DONE] merged components: {len(merged_axes)}")
    print(f"[DONE] removed noise axes: {len(removed_axes)}")
    print(f"[DONE] nodes: {len(nodes_df)}, expected_nodes: {expected_count}")
    print(f"[OUT] {axis_csv}")
    print(f"[OUT] {component_map_csv}")
    print(f"[OUT] {mapping_csv}")
    print(f"[OUT] {removed_axes_csv}")
    print(f"[OUT] {nodes_csv}")
    print(f"[OUT] {summary_json}")
    ifc_root = _resolve_ifc2mesh_root(out_dir)
    print(f"[NOTE] Result root (big folder under workspace): {ifc_root}")


if __name__ == "__main__":
    main()
