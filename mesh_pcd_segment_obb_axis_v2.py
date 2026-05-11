from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

import mesh_pcd_segment_obb_axis as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="点云-构件Mesh邻近性判定与语义标签转移（v2: 表面距离 + 法向 + 精确Mesh裁决）"
    )
    parser.add_argument(
        "--pcd",
        type=Path,
        default=Path("segment_pcd.pcd"),
        help="输入点云文件（已与IFC网格对齐），例如 segment_pcd.pcd",
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=Path("result_run_ifc311_world/meshes"),
        help="PLY三角面片目录",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("result_run_ifc311_world/ifc_mesh_report.json"),
        help="ifc2mesh导出的报告JSON，用于读取构件语义字段",
    )
    parser.add_argument(
        "--tolerance-mm",
        type=float,
        default=50.0,
        help="邻近阈值（毫米），常用30或50",
    )
    parser.add_argument(
        "--coord-unit",
        type=str,
        choices=["m", "mm"],
        default="m",
        help="点云与Mesh的坐标单位。m表示米，mm表示毫米。",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("result_run_ifc311_world/label_transfer_v2"),
        help="输出目录",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500000,
        help="候选点距离计算分块大小，点云很大时可降低内存峰值",
    )
    parser.add_argument(
        "--compute-device",
        type=str,
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="距离场计算设备：auto自动选择，cpu仅CPU，gpu优先GPU（不可用时回退CPU）",
    )
    parser.add_argument(
        "--distance-backend",
        type=str,
        choices=["open3d", "torch-nn"],
        default="open3d",
        help="距离计算后端：open3d为精确点到mesh距离，torch-nn为mesh采样点近邻距离（支持GPU）",
    )
    parser.add_argument(
        "--mesh-sample-factor",
        type=float,
        default=4.0,
        help="torch-nn模式下每个mesh采样点数系数，sample_n=triangle_count*factor",
    )
    parser.add_argument(
        "--mesh-sample-min",
        type=int,
        default=2000,
        help="torch-nn模式下每个mesh最小采样点数",
    )
    parser.add_argument(
        "--mesh-sample-max",
        type=int,
        default=50000,
        help="torch-nn模式下每个mesh最大采样点数",
    )
    parser.add_argument(
        "--torch-query-chunk",
        type=int,
        default=2048,
        help="torch-nn模式下查询点分块大小",
    )
    parser.add_argument(
        "--torch-ref-chunk",
        type=int,
        default=8192,
        help="torch-nn模式下参考点（mesh采样点）分块大小",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="导出逐点CSV（超大点云会非常大，默认不导出）",
    )
    parser.add_argument(
        "--transform-matrix-file",
        type=Path,
        default=None,
        help="CloudCompare导出的4x4配准矩阵文件（txt/csv等），用于将点云变换到Mesh坐标系",
    )
    parser.add_argument(
        "--transform-matrix-values",
        type=float,
        nargs=16,
        default=None,
        metavar=(
            "m00",
            "m01",
            "m02",
            "m03",
            "m10",
            "m11",
            "m12",
            "m13",
            "m20",
            "m21",
            "m22",
            "m23",
            "m30",
            "m31",
            "m32",
            "m33",
        ),
        help="直接传入4x4矩阵（按行展开16个数字）",
    )
    parser.add_argument(
        "--run-dual-tolerance",
        action="store_true",
        help="自动顺序执行30mm和50mm两档标签转移",
    )
    parser.add_argument(
        "--dual-tolerances-mm",
        type=float,
        nargs=2,
        default=[30.0, 50.0],
        metavar=("T1", "T2"),
        help="双阈值模式下使用的两档毫米阈值，默认30 50",
    )
    parser.add_argument(
        "--preview-voxel",
        type=float,
        default=0.02,
        help="overlay预览的点云下采样体素大小",
    )
    parser.add_argument(
        "--preview-max-points",
        type=int,
        default=500000,
        help="overlay预览中每类点云最大点数上限",
    )
    parser.add_argument(
        "--bbox-split-only",
        action="store_true",
        help="仅执行包围盒分割并导出逐构件点云，不执行距离场标签转移",
    )
    parser.add_argument(
        "--bbox-padding-mm",
        type=float,
        default=0.0,
        help="构件包围盒外扩容差（毫米），用于包围盒分割",
    )
    parser.add_argument(
        "--bbox-mode",
        type=str,
        choices=["aabb", "obb_axis"],
        default="aabb",
        help="包围盒候选模式：aabb为轴对齐包围盒，obb_axis为基于ply轴线的定向候选筛选",
    )
    parser.add_argument(
        "--bbox-axis-consistency",
        action="store_true",
        help="保留旧参数，v2不再依赖局部PCA主轴作为主要判据",
    )
    parser.add_argument(
        "--bbox-consistency-knn",
        type=int,
        default=30,
        help="保留旧参数，兼容旧命令行",
    )
    parser.add_argument(
        "--bbox-axis-weight",
        type=float,
        default=0.0,
        help="保留旧参数，v2中默认不使用局部PCA主轴评分",
    )
    parser.add_argument(
        "--bbox-normal-weight",
        type=float,
        default=0.25,
        help="相贯区重叠判别中法向一致性权重",
    )
    parser.add_argument(
        "--bbox-smart-score",
        action="store_true",
        help="启用完整评分分配：表面残差 + 法向 + 精确Mesh距离 + 邻域MRF平滑",
    )
    parser.add_argument(
        "--bbox-seed-ratio-threshold",
        type=float,
        default=1.3,
        help="保留旧参数，v2中仅作为调试输出",
    )
    parser.add_argument(
        "--bbox-line-sigma-mm",
        type=float,
        default=80.0,
        help="轴线距离评分的尺度参数sigma（毫米）",
    )
    parser.add_argument(
        "--bbox-score-axis-weight",
        type=float,
        default=0.0,
        help="保留旧参数，v2中不使用",
    )
    parser.add_argument(
        "--bbox-score-line-weight",
        type=float,
        default=0.0,
        help="保留旧参数，v2中不使用",
    )
    parser.add_argument(
        "--bbox-score-continuity-weight",
        type=float,
        default=0.0,
        help="保留旧参数，v2中不使用",
    )
    parser.add_argument(
        "--bbox-score-knn",
        type=int,
        default=30,
        help="重叠区MRF平滑中使用的KNN",
    )
    parser.add_argument(
        "--bbox-score-iterations",
        type=int,
        default=2,
        help="重叠区MRF迭代次数（>=1）",
    )
    parser.add_argument(
        "--bbox-cleanup-knn",
        type=int,
        default=16,
        help="后处理清理阶段的KNN",
    )
    parser.add_argument(
        "--bbox-cleanup-min-same-neighbors",
        type=int,
        default=3,
        help="后处理中若同标签邻居少于该阈值则尝试重分配",
    )
    parser.add_argument(
        "--bbox-coarse-center-align",
        action="store_true",
        help="分割前先做粗配准平移：将点云全局bbox中心平移到mesh全局bbox中心",
    )
    parser.add_argument(
        "--input-txt",
        type=Path,
        default=None,
        help="txt/csv格式点云路径（每行至少3列x y z），用于替代--pcd",
    )
    parser.add_argument(
        "--txt-voxel",
        type=float,
        default=0.0,
        help="txt点云体素降采样大小（与coord-unit一致），<=0表示不降采样",
    )
    parser.add_argument(
        "--radius-filter-mm",
        type=float,
        default=0.0,
        help="半径滤波半径（毫米），>0时启用 radius outlier removal",
    )
    parser.add_argument(
        "--radius-filter-min-n",
        type=int,
        default=16,
        help="半径滤波最小邻居数 (nb_points)，与 --radius-filter-mm 配合使用",
    )
    parser.add_argument(
        "--quick-validate",
        action="store_true",
        help="快速验证模式：限制txt读取点数，用更小样本先检查重叠区判别链",
    )
    parser.add_argument(
        "--txt-max-points",
        type=int,
        default=0,
        help="读取txt时最多保留的有效点数，<=0表示不截断；快验模式建议20万到50万",
    )
    parser.add_argument(
        "--txt-sample-stride",
        type=int,
        default=1,
        help="读取txt时按行抽样步长，1表示不抽样；仅对快速验证有意义",
    )
    parser.add_argument(
        "--bbox-surface-sigma-mm",
        type=float,
        default=60.0,
        help="管壁表面残差的高斯核尺度参数（毫米）",
    )
    parser.add_argument(
        "--bbox-exact-sigma-mm",
        type=float,
        default=25.0,
        help="精确Mesh距离的高斯核尺度参数（毫米）",
    )
    parser.add_argument(
        "--bbox-exact-mesh-weight",
        type=float,
        default=0.35,
        help="重叠区精确Mesh距离权重",
    )
    parser.add_argument(
        "--bbox-surface-weight",
        type=float,
        default=0.30,
        help="重叠区管壁表面残差权重",
    )
    parser.add_argument(
        "--bbox-mrf-weight",
        type=float,
        default=0.10,
        help="重叠区MRF平滑权重",
    )
    parser.add_argument(
        "--bbox-normal-knn",
        type=int,
        default=30,
        help="点云法向估计KNN",
    )
    parser.add_argument(
        "--bbox-mrf-smooth-sigma",
        type=float,
        default=0.35,
        help="MRF平滑中法向差异的尺度参数",
    )
    return parser.parse_args()


def _surface_distance_score(distance: float, sigma: float) -> float:
    return base._line_distance_score(distance, sigma)


def _load_points_from_txt_stream(
    txt_path: Path,
    max_points: int = 0,
    sample_stride: int = 1,
) -> np.ndarray:
    if not txt_path.exists():
        raise FileNotFoundError(f"txt点云文件不存在: {txt_path}")

    rows: list[list[float]] = []
    skipped = 0
    sample_stride = max(int(sample_stride), 1)
    max_points = max(int(max_points), 0)

    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_index, line in enumerate(f):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if sample_stride > 1 and (line_index % sample_stride) != 0:
                continue

            normalized = re.sub(r"[,;\t]+", " ", stripped)
            parts = normalized.split()
            if len(parts) < 3:
                skipped += 1
                continue

            try:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
            except ValueError:
                skipped += 1
                continue

            rows.append([x, y, z])
            if max_points > 0 and len(rows) >= max_points:
                break

    if not rows:
        raise RuntimeError(f"txt点云未解析到有效xyz数据: {txt_path}")

    points = np.asarray(rows, dtype=np.float64)
    if skipped > 0:
        print(f"[INFO] txt读取时跳过无效行: {skipped}")
    return points


def _build_exact_distance_cache(
    points: np.ndarray,
    overlap_candidate_map: dict[int, list[int]],
    mesh_infos: list[dict],
) -> dict[int, dict[int, float]]:
    mesh_by_id = {int(mi["mesh_id"]): mi for mi in mesh_infos}
    indices_by_mesh: dict[int, list[int]] = {}
    for pi, cands in overlap_candidate_map.items():
        for mid in cands:
            indices_by_mesh.setdefault(int(mid), []).append(int(pi))

    cache: dict[int, dict[int, float]] = {}
    for mid, indices in indices_by_mesh.items():
        mesh_info = mesh_by_id.get(mid)
        if mesh_info is None:
            continue
        mesh = o3d.io.read_triangle_mesh(str(mesh_info["mesh_path"]))
        if mesh.is_empty():
            continue
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        idx_arr = np.asarray(indices, dtype=np.int64)
        query = o3d.core.Tensor(points[idx_arr], dtype=o3d.core.Dtype.Float32)
        d = scene.compute_distance(query).numpy().astype(np.float64)
        cache[mid] = {int(pi): float(di) for pi, di in zip(idx_arr.tolist(), d.tolist())}
    return cache


def _mrf_refine_overlap_labels(
    points: np.ndarray,
    normals: np.ndarray,
    overlap_candidate_map: dict[int, list[int]],
    assigned_mesh_id: np.ndarray,
    candidate_score_map: dict[int, dict[int, float]],
    mrf_knn: int,
    mrf_iterations: int,
    mrf_weight: float,
    smooth_sigma: float,
) -> int:
    if not overlap_candidate_map:
        return 0

    mrf_knn = max(int(mrf_knn), 5)
    mrf_iterations = max(int(mrf_iterations), 1)
    smooth_sigma = max(float(smooth_sigma), 1e-6)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    kd_tree = o3d.geometry.KDTreeFlann(pcd)

    changed_total = 0
    for _ in range(mrf_iterations):
        changed = 0
        for pi, cands in overlap_candidate_map.items():
            _, nbr_idx, _ = kd_tree.search_knn_vector_3d(points[pi], mrf_knn)
            nbr_idx_arr = np.asarray(nbr_idx, dtype=np.int64)
            nbr_idx_arr = nbr_idx_arr[nbr_idx_arr != pi]
            if nbr_idx_arr.size == 0:
                continue

            nbr_labels = assigned_mesh_id[nbr_idx_arr]
            nbr_normals = normals[nbr_idx_arr] if normals.size > 0 else None
            current = int(assigned_mesh_id[pi])
            best_mid = current
            best_score = -np.inf

            for mid in cands:
                data_score = float(candidate_score_map.get(pi, {}).get(int(mid), 0.0))
                if nbr_normals is None:
                    smooth_score = float(np.count_nonzero(nbr_labels == int(mid)) / max(nbr_labels.size, 1))
                else:
                    same = nbr_labels == int(mid)
                    if np.any(same):
                        cos_sim = np.abs(np.sum(nbr_normals * normals[pi], axis=1))
                        angular_gap = 1.0 - np.clip(cos_sim, 0.0, 1.0)
                        support = np.exp(-(angular_gap * angular_gap) / (smooth_sigma * smooth_sigma))
                        smooth_score = float(np.mean(support[same]))
                    else:
                        smooth_score = 0.0

                score = data_score + mrf_weight * smooth_score
                if score > best_score:
                    best_score = score
                    best_mid = int(mid)

            if best_mid != current:
                assigned_mesh_id[pi] = best_mid
                changed += 1

        changed_total += changed
        if changed == 0:
            break

    return changed_total


def run_bbox_component_split_v2(
    pcd_path: Path,
    mesh_dir: Path,
    report_path: Path,
    out_dir: Path,
    coord_unit: str,
    bbox_padding_mm: float,
    bbox_mode: str = "aabb",
    use_axis_consistency: bool = False,
    consistency_knn: int = 30,
    axis_weight: float = 0.0,
    normal_weight: float = 0.25,
    use_smart_score: bool = False,
    seed_ratio_threshold: float = 1.3,
    line_sigma_mm: float = 80.0,
    score_axis_weight: float = 0.0,
    score_line_weight: float = 0.0,
    score_continuity_weight: float = 0.0,
    score_knn: int = 30,
    score_iterations: int = 2,
    cleanup_knn: int = 16,
    cleanup_min_same_neighbors: int = 3,
    coarse_center_align: bool = False,
    points_override: np.ndarray | None = None,
    input_source: str | None = None,
    surface_sigma_mm: float = 60.0,
    exact_sigma_mm: float = 25.0,
    surface_weight: float = 0.30,
    exact_mesh_weight: float = 0.35,
    mrf_weight: float = 0.10,
    normal_knn: int = 30,
    mrf_smooth_sigma: float = 0.35,
    radius_filter_mm: float = 0.0,
    radius_filter_n: int = 16,
) -> None:
    if points_override is None and not pcd_path.exists():
        raise FileNotFoundError(f"点云文件不存在: {pcd_path}")
    if not mesh_dir.exists():
        raise FileNotFoundError(f"Mesh目录不存在: {mesh_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    components_dir = out_dir / "components_pcd"
    components_dir.mkdir(parents=True, exist_ok=True)

    padding = bbox_padding_mm / 1000.0 if coord_unit == "m" else bbox_padding_mm
    report_map = base._build_report_map(report_path)
    mesh_infos = base._load_mesh_infos(mesh_dir, report_map)
    if not mesh_infos:
        raise RuntimeError(f"未在目录中找到可分割的ply: {mesh_dir}")

    if points_override is None:
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        points = np.asarray(pcd.points)
        source_text = str(pcd_path)
    else:
        points = points_override.astype(np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        source_text = input_source or str(pcd_path)
    if points.size == 0:
        raise RuntimeError("输入点云为空")

    # 半径滤波: 去除孤立离群点（以毫米为单位的用户输入）
    try:
        if float(radius_filter_mm) > 0.0:
            radius = float(radius_filter_mm) / 1000.0 if coord_unit == "m" else float(radius_filter_mm)
            nb = max(1, int(radius_filter_n))
            filtered_pcd, ind = pcd.remove_radius_outlier(nb_points=nb, radius=radius)
            kept = np.asarray(filtered_pcd.points).shape[0]
            print(f"[INFO] 半径滤波: radius={radius_filter_mm} {('mm' if coord_unit=='mm' else 'm')}, nb_points={nb}, kept={kept}")
            pcd = filtered_pcd
            points = np.asarray(pcd.points)
            filtered_path = out_dir / "scan_after_radius_filter.pcd"
            out_dir.mkdir(parents=True, exist_ok=True)
            o3d.io.write_point_cloud(str(filtered_path), pcd)
            print(f"[OUT] {filtered_path}")
    except Exception as e:
        print(f"[WARN] 半径滤波失败: {e}")

    if bbox_mode not in {"aabb", "obb_axis"}:
        raise ValueError(f"不支持的bbox_mode: {bbox_mode}")

    use_smart_score = bool(use_smart_score or use_axis_consistency)
    surface_sigma = surface_sigma_mm / 1000.0 if coord_unit == "m" else surface_sigma_mm
    exact_sigma = exact_sigma_mm / 1000.0 if coord_unit == "m" else exact_sigma_mm

    print(f"[INFO] 包围盒分割(v2): source={source_text}, mesh_count={len(mesh_infos)}")
    print(f"[INFO] bbox_mode = {bbox_mode}")
    print(f"[INFO] bbox_padding = {bbox_padding_mm:.3f} mm -> {padding:.6f} ({coord_unit})")
    print(f"[INFO] surface_sigma = {surface_sigma_mm:.2f} mm")
    print(f"[INFO] exact_sigma = {exact_sigma_mm:.2f} mm")
    print(f"[INFO] normal_weight = {normal_weight:.2f}, surface_weight = {surface_weight:.2f}, exact_mesh_weight = {exact_mesh_weight:.2f}, mrf_weight = {mrf_weight:.2f}")

    component_stats = []
    assigned_mesh_id = np.full(points.shape[0], -1, dtype=np.int32)
    global_assigned = np.zeros(points.shape[0], dtype=np.uint8)
    bbox_hit_mask = np.zeros(points.shape[0], dtype=bool)

    overlap_candidate_map: dict[int, list[int]] = {}
    mesh_axes: dict[int, np.ndarray] = {}
    mesh_axis_points: dict[int, np.ndarray] = {}
    mesh_bounds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    mesh_axis_models: dict[int, dict[str, np.ndarray | float]] = {}

    for mi in mesh_infos:
        mesh_path = mi["mesh_path"]
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if mesh.is_empty():
            print(f"[WARN] 跳过空网格: {mesh_path.name}")
            continue

        mid = int(mi["mesh_id"])
        mesh_axes[mid] = base._mesh_axis_direction(mesh)
        mesh_axis_points[mid] = np.asarray(mesh.vertices).mean(axis=0)

        aabb = mesh.get_axis_aligned_bounding_box()
        min_bound = np.asarray(aabb.min_bound) - padding
        max_bound = np.asarray(aabb.max_bound) + padding
        mesh_bounds[mid] = (min_bound, max_bound)
        mesh_axis_models[mid] = base._build_axis_candidate_model(mesh, padding)

        mask = base._build_candidate_mask(
            points=points,
            bbox_mode=bbox_mode,
            min_bound=min_bound,
            max_bound=max_bound,
            axis_model=mesh_axis_models[mid],
        )
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            continue

        bbox_hit_mask[indices] = True
        prev = assigned_mesh_id[indices]
        unassigned_mask = prev < 0
        if np.any(unassigned_mask):
            assigned_mesh_id[indices[unassigned_mask]] = mid

        overlap_local = np.flatnonzero(~unassigned_mask)
        for loc in overlap_local:
            pi = int(indices[loc])
            first_mid = int(prev[loc])
            cand = overlap_candidate_map.get(pi)
            if cand is None:
                cand = [first_mid]
            if mid not in cand:
                cand.append(mid)
            overlap_candidate_map[pi] = cand

    coarse_shift = np.zeros((3,), dtype=np.float64)
    coarse_aligned_pcd_path = ""
    mesh_global_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    mesh_global_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    for min_bound, max_bound in mesh_bounds.values():
        mesh_global_min = np.minimum(mesh_global_min, min_bound)
        mesh_global_max = np.maximum(mesh_global_max, max_bound)

    if np.all(np.isfinite(mesh_global_min)) and np.all(np.isfinite(mesh_global_max)):
        pcd_min = points.min(axis=0)
        pcd_max = points.max(axis=0)
        overlap = np.array(
            [
                base._overlap_1d(pcd_min[i], pcd_max[i], mesh_global_min[i], mesh_global_max[i])
                for i in range(3)
            ],
            dtype=np.float64,
        )
        print(f"[INFO] 全局PCD bbox min={pcd_min}, max={pcd_max}")
        print(f"[INFO] 全局Mesh bbox min={mesh_global_min}, max={mesh_global_max}")
        print(f"[INFO] 全局BBox overlap xyz={overlap}")
        if np.any(overlap <= 0.0):
            print("[WARN] 全局包围盒在至少一个轴向无重叠，当前输入点云很可能不在mesh坐标系中。")

    if coarse_center_align and np.all(np.isfinite(mesh_global_min)) and np.all(np.isfinite(mesh_global_max)):
        pcd_min = points.min(axis=0)
        pcd_max = points.max(axis=0)
        pcd_center = 0.5 * (pcd_min + pcd_max)
        mesh_center = 0.5 * (mesh_global_min + mesh_global_max)
        coarse_shift = mesh_center - pcd_center
        points = points + coarse_shift
        pcd.points = o3d.utility.Vector3dVector(points)

        assigned_mesh_id.fill(-1)
        bbox_hit_mask[:] = False
        overlap_candidate_map.clear()

        for mid, bounds in mesh_bounds.items():
            min_bound, max_bound = bounds
            axis_model = mesh_axis_models[mid]
            mask = base._build_candidate_mask(
                points=points,
                bbox_mode=bbox_mode,
                min_bound=min_bound,
                max_bound=max_bound,
                axis_model=axis_model,
            )
            indices = np.flatnonzero(mask)
            if indices.size == 0:
                continue

            bbox_hit_mask[indices] = True
            prev = assigned_mesh_id[indices]
            unassigned_mask = prev < 0
            if np.any(unassigned_mask):
                assigned_mesh_id[indices[unassigned_mask]] = int(mid)

            overlap_local = np.flatnonzero(~unassigned_mask)
            for loc in overlap_local:
                pi = int(indices[loc])
                first_mid = int(prev[loc])
                cand = overlap_candidate_map.get(pi)
                if cand is None:
                    cand = [first_mid]
                if int(mid) not in cand:
                    cand.append(int(mid))
                overlap_candidate_map[pi] = cand

        coarse_aligned_pcd = o3d.geometry.PointCloud()
        coarse_aligned_pcd.points = o3d.utility.Vector3dVector(points)
        coarse_aligned_pcd_path_obj = out_dir / "scan_coarse_aligned_for_bbox_split.pcd"
        o3d.io.write_point_cloud(str(coarse_aligned_pcd_path_obj), coarse_aligned_pcd)
        coarse_aligned_pcd_path = str(coarse_aligned_pcd_path_obj)
        print(f"[INFO] 粗配准平移向量({coord_unit}) = {coarse_shift}")
        print(f"[OUT] {coarse_aligned_pcd_path}")

    normals = np.empty((0, 3), dtype=np.float64)
    if points.shape[0] >= 3:
        normals_pcd = base._build_point_cloud_with_normals(points, knn=max(int(normal_knn), 10))
        normals = np.asarray(normals_pcd.normals, dtype=np.float64)
        if normals.shape[0] != points.shape[0]:
            normals = np.zeros((points.shape[0], 3), dtype=np.float64)

    overlap_points = int(len(overlap_candidate_map))
    exact_dist_cache: dict[int, dict[int, float]] = {}
    candidate_score_map: dict[int, dict[int, float]] = {}
    refined_reassigned = 0

    if overlap_points > 0:
        exact_dist_cache = _build_exact_distance_cache(points, overlap_candidate_map, mesh_infos)
        surface_sigma_safe = max(float(surface_sigma), 1e-9)
        exact_sigma_safe = max(float(exact_sigma), 1e-9)

        for pi, cands in overlap_candidate_map.items():
            p = points[pi]
            n = normals[pi] if normals.shape[0] == points.shape[0] else None
            candidate_score_map[pi] = {}

            best_mid = int(cands[0])
            best_score = -np.inf
            for mid in cands:
                axis = mesh_axes.get(int(mid), np.array([1.0, 0.0, 0.0], dtype=np.float64))
                axis_p = mesh_axis_points.get(int(mid), np.zeros((3,), dtype=np.float64))
                axis_model = mesh_axis_models.get(int(mid), {})
                radius = float(axis_model.get("radial_max", 0.0))

                line_dist = base._point_to_axis_distance(p, axis_p, axis)
                surface_residual = abs(line_dist - radius)
                surface_score = _surface_distance_score(surface_residual, surface_sigma_safe)

                exact_dist = exact_dist_cache.get(int(mid), {}).get(int(pi), np.inf)
                exact_score = _surface_distance_score(exact_dist, exact_sigma_safe) if np.isfinite(exact_dist) else 0.0

                if n is not None and n.size == 3:
                    normal_score = 1.0 - abs(float(np.dot(n, axis)))
                    normal_score = float(np.clip(normal_score, 0.0, 1.0))
                else:
                    normal_score = 0.5

                data_score = (
                    float(surface_weight) * surface_score
                    + float(exact_mesh_weight) * exact_score
                    + float(normal_weight) * normal_score
                )

                candidate_score_map[pi][int(mid)] = data_score
                if data_score > best_score:
                    best_score = data_score
                    best_mid = int(mid)

            assigned_mesh_id[pi] = best_mid

        refined_reassigned = _mrf_refine_overlap_labels(
            points=points,
            normals=normals,
            overlap_candidate_map=overlap_candidate_map,
            assigned_mesh_id=assigned_mesh_id,
            candidate_score_map=candidate_score_map,
            mrf_knn=score_knn,
            mrf_iterations=score_iterations,
            mrf_weight=mrf_weight,
            smooth_sigma=mrf_smooth_sigma,
        )

    points_by_mesh: dict[int, np.ndarray] = {}
    for mi in mesh_infos:
        mid = int(mi["mesh_id"])
        points_by_mesh[mid] = np.flatnonzero(assigned_mesh_id == mid)
    global_assigned[assigned_mesh_id >= 0] = 1

    for mi in mesh_infos:
        mesh_path = mi["mesh_path"]
        mid = int(mi["mesh_id"])
        bounds = mesh_bounds.get(mid)
        if bounds is None:
            continue
        min_bound, max_bound = bounds
        indices = points_by_mesh[mid]
        point_count = int(indices.size)

        label_name = base._safe_name(str(mi.get("name", mesh_path.stem)))
        guid = base._safe_name(str(mi.get("guid", "N_A")))
        ifc_type = base._safe_name(str(mi.get("ifc_type", "N_A")))
        file_stem = f"{int(mi['index']):04d}_{ifc_type}_{label_name}_{guid}"
        comp_pcd_path = components_dir / f"{file_stem}.pcd"

        if point_count > 0:
            comp_pcd = pcd.select_by_index(indices.tolist())
            o3d.io.write_point_cloud(str(comp_pcd_path), comp_pcd)
            global_assigned[indices] = 1

        component_stats.append(
            {
                "mesh_id": int(mi["mesh_id"]),
                "ifc_index": int(mi["index"]),
                "ifc_type": str(mi.get("ifc_type", "N/A")),
                "name": str(mi.get("name", "N/A")),
                "guid": str(mi.get("guid", "N/A")),
                "mesh_file": str(mesh_path),
                "bbox_min": min_bound.tolist(),
                "bbox_max": max_bound.tolist(),
                "point_count": point_count,
                "segmented_pcd": str(comp_pcd_path) if point_count > 0 else "",
            }
        )

    label_map_json = out_dir / "component_label_map.json"
    with label_map_json.open("w", encoding="utf-8") as f:
        json.dump(component_stats, f, ensure_ascii=False, indent=2)

    label_map_csv = out_dir / "component_label_map.csv"
    with label_map_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["mesh_id", "ifc_index", "ifc_type", "name", "guid", "mesh_file", "point_count", "segmented_pcd"])
        for row in component_stats:
            writer.writerow([row["mesh_id"], row["ifc_index"], row["ifc_type"], row["name"], row["guid"], row["mesh_file"], row["point_count"], row["segmented_pcd"]])

    summary = {
        "pcd_path": str(pcd_path),
        "input_source": source_text,
        "mesh_dir": str(mesh_dir),
        "report_path": str(report_path),
        "coord_unit": coord_unit,
        "bbox_mode": bbox_mode,
        "coarse_center_align_enabled": bool(coarse_center_align),
        "coarse_shift_in_coord_unit": coarse_shift.tolist(),
        "coarse_shift_mm": (coarse_shift * 1000.0).tolist() if coord_unit == "m" else coarse_shift.tolist(),
        "bbox_padding_mm": float(bbox_padding_mm),
        "bbox_padding_in_coord_unit": float(padding),
        "axis_consistency_enabled": bool(use_axis_consistency),
        "smart_score_enabled": bool(use_smart_score),
        "consistency_knn": int(consistency_knn),
        "axis_weight": float(axis_weight),
        "normal_weight": float(normal_weight),
        "seed_ratio_threshold": float(seed_ratio_threshold),
        "line_sigma_mm": float(line_sigma_mm),
        "line_sigma_in_coord_unit": float(line_sigma_mm / 1000.0 if coord_unit == "m" else line_sigma_mm),
        "surface_sigma_mm": float(surface_sigma_mm),
        "surface_sigma_in_coord_unit": float(surface_sigma),
        "exact_sigma_mm": float(exact_sigma_mm),
        "exact_sigma_in_coord_unit": float(exact_sigma),
        "surface_weight": float(surface_weight),
        "exact_mesh_weight": float(exact_mesh_weight),
        "mrf_weight": float(mrf_weight),
        "normal_knn": int(normal_knn),
        "mrf_iterations": int(score_iterations),
        "mrf_smooth_sigma": float(mrf_smooth_sigma),
        "overlap_points": int(overlap_points),
        "refined_reassigned_points": int(refined_reassigned),
        "input_point_count": int(points.shape[0]),
        "mesh_count": int(len(mesh_infos)),
        "component_count": int(len(component_stats)),
        "components_with_points": int(sum(1 for c in component_stats if c["point_count"] > 0)),
        "bbox_hit_points_union": int(np.count_nonzero(bbox_hit_mask)),
        "bbox_hit_ratio_union": float(np.count_nonzero(bbox_hit_mask) / points.shape[0]),
        "assigned_points_union": int(np.count_nonzero(global_assigned)),
        "assigned_ratio_union": float(np.count_nonzero(global_assigned) / points.shape[0]),
        "outputs": {
            "components_dir": str(components_dir),
            "coarse_aligned_pcd": coarse_aligned_pcd_path,
            "component_label_map_json": str(label_map_json),
            "component_label_map_csv": str(label_map_csv),
        },
    }
    summary_path = out_dir / "bbox_split_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] 包围盒分割完成(v2): components={summary['component_count']}, 非空={summary['components_with_points']}")
    print(f"[DONE] BBox命中点数(并集): {summary['bbox_hit_points_union']} / {summary['input_point_count']} ({summary['bbox_hit_ratio_union']:.2%})")
    print(f"[DONE] 赋值点数(并集): {summary['assigned_points_union']} / {summary['input_point_count']} ({summary['assigned_ratio_union']:.2%})")
    print(f"[DONE] 相贯区重分配点数: {summary['refined_reassigned_points']}")
    print(f"[OUT] {components_dir}")
    print(f"[OUT] {label_map_json}")
    print(f"[OUT] {label_map_csv}")
    print(f"[OUT] {summary_path}")


def main() -> None:
    args = parse_args()
    if args.input_txt is None and not args.pcd.exists():
        raise FileNotFoundError(f"点云文件不存在: {args.pcd}")
    if not args.mesh_dir.exists():
        raise FileNotFoundError(f"Mesh目录不存在: {args.mesh_dir}")

    if args.bbox_split_only:
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        split_out_dir = args.out_dir / f"subresult_bbox_split_v2_{run_stamp}"
        points_override = None
        input_source = None
        if args.input_txt is not None:
            if args.quick_validate:
                quick_max_points = args.txt_max_points if args.txt_max_points > 0 else 300000
                quick_stride = max(int(args.txt_sample_stride), 1)
                print(
                    "[INFO] 快速验证模式已启用: "
                    f"txt_max_points={quick_max_points}, txt_sample_stride={quick_stride}"
                )
                if args.txt_voxel <= 0.0:
                    print("[INFO] 快验模式未显式设置txt_voxel，仍使用抽样后的原始点")
                raw_txt_points = _load_points_from_txt_stream(
                    args.input_txt,
                    max_points=quick_max_points,
                    sample_stride=quick_stride,
                )
            else:
                raw_txt_points = _load_points_from_txt_stream(args.input_txt)
            raw_count = int(raw_txt_points.shape[0])
            ds_points = base._voxel_downsample_points(raw_txt_points, args.txt_voxel)
            ds_count = int(ds_points.shape[0])
            print(f"[INFO] txt原始点数: {raw_count}")
            print(f"[INFO] 当前点数(降采样后): {ds_count}")
            if args.txt_voxel > 0.0:
                print(f"[INFO] txt体素降采样: voxel={args.txt_voxel:.6f} ({args.coord_unit})")
            points_override = ds_points
            input_source = str(args.input_txt)

            split_out_dir.mkdir(parents=True, exist_ok=True)
            ds_pcd = o3d.geometry.PointCloud()
            ds_pcd.points = o3d.utility.Vector3dVector(ds_points)
            ds_pcd_path = split_out_dir / "scan_from_txt_downsampled_for_bbox_split.pcd"
            try:
                o3d.io.write_point_cloud(str(ds_pcd_path), ds_pcd)
                print(f"[OUT] {ds_pcd_path}")
            except Exception as ex:
                print(f"[WARN] 快验输入点云写出失败，继续运行: {ex}")

        run_bbox_component_split_v2(
            pcd_path=args.pcd,
            mesh_dir=args.mesh_dir,
            report_path=args.report,
            out_dir=split_out_dir,
            coord_unit=args.coord_unit,
            bbox_padding_mm=args.bbox_padding_mm,
            bbox_mode=args.bbox_mode,
            use_axis_consistency=args.bbox_axis_consistency,
            consistency_knn=args.bbox_consistency_knn,
            axis_weight=args.bbox_axis_weight,
            normal_weight=args.bbox_normal_weight,
            use_smart_score=args.bbox_smart_score,
            seed_ratio_threshold=args.bbox_seed_ratio_threshold,
            line_sigma_mm=args.bbox_line_sigma_mm,
            score_axis_weight=args.bbox_score_axis_weight,
            score_line_weight=args.bbox_score_line_weight,
            score_continuity_weight=args.bbox_score_continuity_weight,
            score_knn=args.bbox_score_knn,
            score_iterations=args.bbox_score_iterations,
            cleanup_knn=args.bbox_cleanup_knn,
            cleanup_min_same_neighbors=args.bbox_cleanup_min_same_neighbors,
            coarse_center_align=args.bbox_coarse_center_align,
            points_override=points_override,
            input_source=input_source,
            surface_sigma_mm=args.bbox_surface_sigma_mm,
            exact_sigma_mm=args.bbox_exact_sigma_mm,
            surface_weight=args.bbox_surface_weight,
            exact_mesh_weight=args.bbox_exact_mesh_weight,
            mrf_weight=args.bbox_mrf_weight,
            normal_knn=args.bbox_normal_knn,
            mrf_smooth_sigma=args.bbox_mrf_smooth_sigma,
            radius_filter_mm=args.radius_filter_mm,
            radius_filter_n=args.radius_filter_min_n,
        )
        return

    transform = base._load_transform_matrix(args.transform_matrix_file, args.transform_matrix_values)
    print("[INFO] 使用4x4变换矩阵:")
    print(transform)

    raw_points = None
    if args.input_txt is not None:
        raw_txt_points = _load_points_from_txt_stream(args.input_txt)
        raw_count = int(raw_txt_points.shape[0])
        ds_points = base._voxel_downsample_points(raw_txt_points, args.txt_voxel)
        ds_count = int(ds_points.shape[0])
        print(f"[INFO] txt原始点数: {raw_count}")
        print(f"[INFO] 当前点数(降采样后): {ds_count}")
        if args.txt_voxel > 0.0:
            print(f"[INFO] txt体素降采样: voxel={args.txt_voxel:.6f} ({args.coord_unit})")
        raw_points = ds_points
    else:
        raw_pcd = o3d.io.read_point_cloud(str(args.pcd))
        raw_points = np.asarray(raw_pcd.points)
    if raw_points.size == 0:
        raise RuntimeError("输入点云为空")

    # 半径滤波: 去除孤立离群点（以毫米为单位的用户输入）
    try:
        if float(args.radius_filter_mm) > 0.0:
            radius = float(args.radius_filter_mm) / 1000.0 if args.coord_unit == "m" else float(args.radius_filter_mm)
            nb = max(1, int(args.radius_filter_min_n))
            filter_pcd = o3d.geometry.PointCloud()
            filter_pcd.points = o3d.utility.Vector3dVector(raw_points)
            filtered_pcd, ind = filter_pcd.remove_radius_outlier(nb_points=nb, radius=radius)
            kept = np.asarray(filtered_pcd.points).shape[0]
            print(f"[INFO] 半径滤波: radius={args.radius_filter_mm} mm, nb_points={nb}, kept={kept}")
            raw_points = np.asarray(filtered_pcd.points)
    except Exception as e:
        print(f"[WARN] 半径滤波失败: {e}")

    transformed_points = base._apply_transform(raw_points, transform)

    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)
    transformed_pcd_path = args.out_dir / "scan_transformed_for_label_transfer.pcd"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(transformed_pcd_path), transformed_pcd)
    print(f"[OUT] {transformed_pcd_path}")

    preview_summary = base._build_alignment_preview(
        transformed_points=transformed_points,
        mesh_dir=args.mesh_dir,
        out_dir=args.out_dir,
        voxel_size=args.preview_voxel,
        max_points=args.preview_max_points,
    )
    print("[DONE] 已生成变换后overlay预览")
    print(f"[OUT] {preview_summary['outputs']['overlay_preview']}")
    print(f"[OUT] {preview_summary['summary_json']}")
    print(f"[INFO] 变换后BBox overlap xyz={np.array(preview_summary['bbox_overlap_xyz'])}")

    if np.any(np.array(preview_summary["bbox_overlap_xyz"]) <= 0.0):
        print("[WARN] 变换后仍存在轴向无重叠，请先复核矩阵方向/坐标单位。")

    if args.run_dual_tolerance:
        tolerances = [float(args.dual_tolerances_mm[0]), float(args.dual_tolerances_mm[1])]
    else:
        tolerances = [float(args.tolerance_mm)]

    for tol in tolerances:
        run_out_dir = args.out_dir / f"label_transfer_{int(round(tol))}mm"
        print(f"[RUN] 开始标签转移: tolerance={tol:.2f}mm -> {run_out_dir}")
        base.run_label_transfer(
            pcd_path=transformed_pcd_path,
            mesh_dir=args.mesh_dir,
            report_path=args.report,
            tolerance_mm=tol,
            coord_unit=args.coord_unit,
            out_dir=run_out_dir,
            chunk_size=args.chunk_size,
            write_csv=args.write_csv,
            compute_device=args.compute_device,
            distance_backend=args.distance_backend,
            mesh_sample_factor=args.mesh_sample_factor,
            mesh_sample_min=args.mesh_sample_min,
            mesh_sample_max=args.mesh_sample_max,
            torch_query_chunk=args.torch_query_chunk,
            torch_ref_chunk=args.torch_ref_chunk,
            points_override=transformed_points,
            transformed_pcd_path=transformed_pcd_path,
        )


if __name__ == "__main__":
    main()