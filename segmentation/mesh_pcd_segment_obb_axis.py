import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="点云-构件Mesh邻近性判定与语义标签转移（包围盒粗筛 + 距离场精算）"
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
        default=Path("result_run_ifc311_world/label_transfer"),
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
        help="在包围盒重叠区域启用轴线/法向量一致性判别，减少杆件相交处误分割",
    )
    parser.add_argument(
        "--bbox-consistency-knn",
        type=int,
        default=30,
        help="一致性判别的近邻点数量（用于法向量与局部轴线估计）",
    )
    parser.add_argument(
        "--bbox-axis-weight",
        type=float,
        default=0.7,
        help="一致性判别中‘局部轴线与构件轴线平行度’权重",
    )
    parser.add_argument(
        "--bbox-normal-weight",
        type=float,
        default=0.3,
        help="一致性判别中‘法向与构件轴线垂直度’权重",
    )
    parser.add_argument(
        "--bbox-smart-score",
        action="store_true",
        help="启用完整评分分配：种子判定 + 轴线平行 + 轴线距离 + 邻域连续性",
    )
    parser.add_argument(
        "--bbox-seed-ratio-threshold",
        type=float,
        default=1.3,
        help="重叠点种子判定阈值：次优轴距/最优轴距超过该值则直接赋予最优构件",
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
        default=0.55,
        help="完整评分中局部轴线平行项权重",
    )
    parser.add_argument(
        "--bbox-score-line-weight",
        type=float,
        default=0.30,
        help="完整评分中点到构件轴线距离项权重",
    )
    parser.add_argument(
        "--bbox-score-continuity-weight",
        type=float,
        default=0.15,
        help="完整评分中邻域标签连续性项权重",
    )
    parser.add_argument(
        "--bbox-score-knn",
        type=int,
        default=30,
        help="完整评分中局部轴线和邻域连续性使用的KNN",
    )
    parser.add_argument(
        "--bbox-score-iterations",
        type=int,
        default=1,
        help="完整评分迭代次数（>=1）",
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
    return parser.parse_args()


def _build_report_map(report_path: Path) -> dict[str, dict]:
    if not report_path.exists():
        print(f"[WARN] 报告文件不存在: {report_path}")
        return {}

    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    records = report.get("records", [])
    mapping = {}
    for rec in records:
        mesh_file = rec.get("mesh_file")
        if not mesh_file:
            continue
        mesh_name = Path(mesh_file).name
        mapping[mesh_name] = {
            "index": rec.get("index", -1),
            "ifc_type": rec.get("ifc_type", "N/A"),
            "name": rec.get("name", "N/A"),
            "guid": rec.get("guid", "N/A"),
            "profile": rec.get("profile", "N/A"),
        }
    return mapping


def _color_for_label(label_id: int) -> np.ndarray:
    # 通过标签ID生成稳定伪随机颜色，避免每次运行颜色变化。
    seed = (label_id * 1103515245 + 12345) & 0x7FFFFFFF
    r = ((seed >> 16) & 255) / 255.0
    g = ((seed >> 8) & 255) / 255.0
    b = (seed & 255) / 255.0
    return np.array([r, g, b], dtype=np.float64)


def _load_mesh_infos(mesh_dir: Path, report_map: dict[str, dict]) -> list[dict]:
    mesh_paths = sorted(mesh_dir.glob("*.ply"))
    mesh_infos = []
    for i, mesh_path in enumerate(mesh_paths):
        meta = report_map.get(
            mesh_path.name,
            {
                "index": i + 1,
                "ifc_type": "N/A",
                "name": mesh_path.stem,
                "guid": "N/A",
                "profile": "N/A",
            },
        )
        mesh_infos.append(
            {
                "mesh_id": i,
                "mesh_path": mesh_path,
                "index": int(meta.get("index", i + 1)),
                "ifc_type": str(meta.get("ifc_type", "N/A")),
                "name": str(meta.get("name", "N/A")),
                "guid": str(meta.get("guid", "N/A")),
                "profile": str(meta.get("profile", "N/A")),
            }
        )
    return mesh_infos


def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _parse_matrix_text(text: str) -> np.ndarray:
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)]
    if len(nums) < 16:
        raise ValueError("矩阵文件中可解析数值不足16个")
    mat = np.array(nums[-16:], dtype=np.float64).reshape(4, 4)
    return mat


def _load_transform_matrix(matrix_file: Path | None, matrix_values: list[float] | None) -> np.ndarray:
    if matrix_values is not None:
        return np.array(matrix_values, dtype=np.float64).reshape(4, 4)
    if matrix_file is not None:
        if not matrix_file.exists():
            raise FileNotFoundError(f"变换矩阵文件不存在: {matrix_file}")
        text = matrix_file.read_text(encoding="utf-8", errors="ignore")
        return _parse_matrix_text(text)
    return np.eye(4, dtype=np.float64)


def _to_homo(points: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    return np.hstack([points.astype(np.float64), ones])


def _apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homo = _to_homo(points)
    out = (homo @ transform.T)[:, :3]
    return out.astype(np.float64)


def _sample_to_limit(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    idx = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
    return points[idx]


def _build_alignment_preview(
    transformed_points: np.ndarray,
    mesh_dir: Path,
    out_dir: Path,
    voxel_size: float,
    max_points: int,
) -> dict:
    preview_dir = out_dir / "alignment_preview_after_transform"
    preview_dir.mkdir(parents=True, exist_ok=True)

    scan_pcd = o3d.geometry.PointCloud()
    scan_pcd.points = o3d.utility.Vector3dVector(transformed_points)
    scan_ds = scan_pcd.voxel_down_sample(voxel_size=voxel_size)
    scan_ds_pts = np.asarray(scan_ds.points)
    scan_ds_pts = _sample_to_limit(scan_ds_pts, max_points)

    mesh_sample_points = []
    mesh_global_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    mesh_global_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    mesh_paths = sorted(mesh_dir.glob("*.ply"))
    for mp in mesh_paths:
        mesh = o3d.io.read_triangle_mesh(str(mp))
        if mesh.is_empty():
            continue
        aabb = mesh.get_axis_aligned_bounding_box()
        mesh_global_min = np.minimum(mesh_global_min, np.asarray(aabb.min_bound))
        mesh_global_max = np.maximum(mesh_global_max, np.asarray(aabb.max_bound))
        tri_num = len(mesh.triangles)
        n = int(min(max(tri_num * 4, 800), 8000))
        smp = mesh.sample_points_uniformly(number_of_points=n)
        if len(smp.points) > 0:
            mesh_sample_points.append(np.asarray(smp.points))

    if not mesh_sample_points:
        raise RuntimeError("构建overlay预览失败：未采样到任何mesh点")

    mesh_pts = np.vstack(mesh_sample_points)
    mesh_pts = _sample_to_limit(mesh_pts, max_points)

    scan_color = np.tile(np.array([[0.1, 0.45, 0.95]], dtype=np.float64), (scan_ds_pts.shape[0], 1))
    mesh_color = np.tile(np.array([[0.95, 0.2, 0.2]], dtype=np.float64), (mesh_pts.shape[0], 1))

    scan_ds_out = o3d.geometry.PointCloud()
    scan_ds_out.points = o3d.utility.Vector3dVector(scan_ds_pts)
    scan_ds_out.colors = o3d.utility.Vector3dVector(scan_color)

    mesh_pcd_out = o3d.geometry.PointCloud()
    mesh_pcd_out.points = o3d.utility.Vector3dVector(mesh_pts)
    mesh_pcd_out.colors = o3d.utility.Vector3dVector(mesh_color)

    overlay_pts = np.vstack([scan_ds_pts, mesh_pts])
    overlay_cols = np.vstack([scan_color, mesh_color])
    overlay = o3d.geometry.PointCloud()
    overlay.points = o3d.utility.Vector3dVector(overlay_pts)
    overlay.colors = o3d.utility.Vector3dVector(overlay_cols)

    scan_path = preview_dir / "scan_transformed_downsample_blue.pcd"
    mesh_path = preview_dir / "mesh_sample_red.pcd"
    overlay_path = preview_dir / "overlay_after_transform_blue_scan_red_mesh.pcd"
    o3d.io.write_point_cloud(str(scan_path), scan_ds_out)
    o3d.io.write_point_cloud(str(mesh_path), mesh_pcd_out)
    o3d.io.write_point_cloud(str(overlay_path), overlay)

    pcd_min = transformed_points.min(axis=0)
    pcd_max = transformed_points.max(axis=0)
    overlap = np.array(
        [_overlap_1d(pcd_min[i], pcd_max[i], mesh_global_min[i], mesh_global_max[i]) for i in range(3)],
        dtype=np.float64,
    )

    summary = {
        "scan_point_count_raw": int(transformed_points.shape[0]),
        "scan_point_count_preview": int(scan_ds_pts.shape[0]),
        "mesh_sample_point_count_preview": int(mesh_pts.shape[0]),
        "pcd_bbox_min": pcd_min.tolist(),
        "pcd_bbox_max": pcd_max.tolist(),
        "mesh_bbox_min": mesh_global_min.tolist(),
        "mesh_bbox_max": mesh_global_max.tolist(),
        "bbox_overlap_xyz": overlap.tolist(),
        "outputs": {
            "scan_downsample_blue": str(scan_path),
            "mesh_sample_red": str(mesh_path),
            "overlay_preview": str(overlay_path),
        },
    }
    summary_path = preview_dir / "alignment_preview_after_transform_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    summary["summary_json"] = str(summary_path)
    return summary


def _safe_name(name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", name)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "UNNAMED"


def _load_points_from_txt(txt_path: Path) -> np.ndarray:
    if not txt_path.exists():
        raise FileNotFoundError(f"txt点云文件不存在: {txt_path}")

    rows: list[list[float]] = []
    skipped = 0
    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
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

    if not rows:
        raise RuntimeError(f"txt点云未解析到有效xyz数据: {txt_path}")

    points = np.asarray(rows, dtype=np.float64)
    if skipped > 0:
        print(f"[INFO] txt读取时跳过无效行: {skipped}")
    return points


def _voxel_downsample_points(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0.0:
        return points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    ds = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(ds.points, dtype=np.float64)


def _principal_axis_from_points(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 3:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    centered = points - points.mean(axis=0, keepdims=True)
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    n = float(np.linalg.norm(axis))
    if n <= 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return (axis / n).astype(np.float64)


def _mesh_axis_direction(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices)
    return _principal_axis_from_points(vertices)


def _build_point_cloud_with_normals(points: np.ndarray, knn: int) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=max(int(knn), 10)))
    return pcd


def _estimate_local_axis_for_point(
    points: np.ndarray,
    kd_tree: o3d.geometry.KDTreeFlann,
    point_index: int,
    knn: int,
) -> np.ndarray:
    k = max(int(knn), 10)
    _, idx, _ = kd_tree.search_knn_vector_3d(points[point_index], k)
    if len(idx) < 3:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    neigh = points[np.asarray(idx, dtype=np.int64)]
    return _principal_axis_from_points(neigh)


def _point_to_axis_distance(point: np.ndarray, axis_point: np.ndarray, axis_dir: np.ndarray) -> float:
    v = point - axis_point
    proj = float(np.dot(v, axis_dir))
    perp = v - proj * axis_dir
    return float(np.linalg.norm(perp))


def _line_distance_score(distance: float, sigma: float) -> float:
    s = max(float(sigma), 1e-9)
    return float(np.exp(-(distance * distance) / (s * s)))


def _build_axis_candidate_model(mesh: o3d.geometry.TriangleMesh, padding: float) -> dict[str, np.ndarray | float]:
    vertices = np.asarray(mesh.vertices)
    if vertices.size == 0:
        return {
            "center": np.zeros((3,), dtype=np.float64),
            "axis": np.array([1.0, 0.0, 0.0], dtype=np.float64),
            "axial_min": 0.0,
            "axial_max": 0.0,
            "radial_max": 0.0,
        }

    center = vertices.mean(axis=0)
    axis = _principal_axis_from_points(vertices)
    rel = vertices - center
    axial = rel @ axis
    axial_min = float(np.min(axial) - padding)
    axial_max = float(np.max(axial) + padding)
    radial_vec = rel - np.outer(axial, axis)
    radial = np.linalg.norm(radial_vec, axis=1)
    radial_max = float(np.max(radial) + padding)

    return {
        "center": center.astype(np.float64),
        "axis": axis.astype(np.float64),
        "axial_min": axial_min,
        "axial_max": axial_max,
        "radial_max": radial_max,
    }


def _build_candidate_mask(
    points: np.ndarray,
    bbox_mode: str,
    min_bound: np.ndarray,
    max_bound: np.ndarray,
    axis_model: dict[str, np.ndarray | float],
) -> np.ndarray:
    if bbox_mode == "aabb":
        return np.all((points >= min_bound) & (points <= max_bound), axis=1)

    center = np.asarray(axis_model["center"], dtype=np.float64)
    axis = np.asarray(axis_model["axis"], dtype=np.float64)
    axial_min = float(axis_model["axial_min"])
    axial_max = float(axis_model["axial_max"])
    radial_max = float(axis_model["radial_max"])

    rel = points - center
    axial = rel @ axis
    radial2 = np.sum(rel * rel, axis=1) - axial * axial
    radial2 = np.maximum(radial2, 0.0)
    radial = np.sqrt(radial2)

    return (axial >= axial_min) & (axial <= axial_max) & (radial <= radial_max)


def _resolve_open3d_device(compute_device: str) -> tuple[o3d.core.Device, bool, str]:
    cuda_available = hasattr(o3d.core, "cuda") and o3d.core.cuda.is_available()

    if compute_device == "cpu":
        return o3d.core.Device("CPU:0"), False, "CPU:0"

    if compute_device == "gpu":
        if cuda_available:
            return o3d.core.Device("CUDA:0"), True, "CUDA:0"
        print("[WARN] 用户指定GPU，但当前Open3D未检测到CUDA，已回退CPU")
        return o3d.core.Device("CPU:0"), False, "CPU:0"

    if cuda_available:
        print("[INFO] compute_device=auto，检测到CUDA，优先使用GPU")
        return o3d.core.Device("CUDA:0"), True, "CUDA:0"

    print("[INFO] compute_device=auto，未检测到CUDA，使用CPU")
    return o3d.core.Device("CPU:0"), False, "CPU:0"


def _try_import_torch() -> Any | None:
    try:
        import torch  # type: ignore

        return torch
    except Exception:
        return None


def _resolve_torch_device(compute_device: str, torch_mod: Any) -> tuple[str, bool, str]:
    cuda_available = bool(torch_mod.cuda.is_available())

    if compute_device == "cpu":
        return "cpu", False, "cpu"

    if compute_device == "gpu":
        if cuda_available:
            return "cuda", True, "cuda"
        print("[WARN] 用户指定GPU，但当前PyTorch未检测到CUDA，已回退CPU")
        return "cpu", False, "cpu"

    if cuda_available:
        print("[INFO] compute_device=auto，检测到PyTorch CUDA，优先使用GPU")
        return "cuda", True, "cuda"

    print("[INFO] compute_device=auto，未检测到PyTorch CUDA，使用CPU")
    return "cpu", False, "cpu"


def _sample_mesh_points_for_torch_nn(
    mesh: o3d.geometry.TriangleMesh,
    sample_factor: float,
    sample_min: int,
    sample_max: int,
) -> np.ndarray:
    tri_num = len(mesh.triangles)
    sample_n = int(tri_num * sample_factor)
    sample_n = max(sample_n, sample_min)
    sample_n = min(sample_n, sample_max)
    sample_n = max(sample_n, 128)
    sampled = mesh.sample_points_uniformly(number_of_points=sample_n)
    return np.asarray(sampled.points, dtype=np.float32)


def _torch_nn_min_distance(
    query_points: np.ndarray,
    ref_points: np.ndarray,
    torch_mod: Any,
    torch_device: str,
    ref_chunk: int,
) -> np.ndarray:
    if query_points.size == 0:
        return np.empty((0,), dtype=np.float64)
    if ref_points.size == 0:
        return np.full((query_points.shape[0],), np.inf, dtype=np.float64)

    ref_chunk = max(int(ref_chunk), 256)

    with torch_mod.no_grad():
        q = torch_mod.from_numpy(query_points.astype(np.float32, copy=False)).to(torch_device)
        q2 = (q * q).sum(dim=1, keepdim=True)
        best_d2 = torch_mod.full((q.shape[0],), float("inf"), device=torch_device, dtype=q.dtype)

        for r_start in range(0, ref_points.shape[0], ref_chunk):
            r_end = min(r_start + ref_chunk, ref_points.shape[0])
            r_np = ref_points[r_start:r_end]
            r = torch_mod.from_numpy(r_np.astype(np.float32, copy=False)).to(torch_device)
            r2 = (r * r).sum(dim=1).unsqueeze(0)
            d2 = q2 + r2 - 2.0 * (q @ r.transpose(0, 1))
            d2 = torch_mod.clamp(d2, min=0.0)
            local_min = d2.min(dim=1).values
            best_d2 = torch_mod.minimum(best_d2, local_min)

        dist = torch_mod.sqrt(best_d2)
        return dist.detach().cpu().numpy().astype(np.float64)


def run_bbox_component_split(
    pcd_path: Path,
    mesh_dir: Path,
    report_path: Path,
    out_dir: Path,
    coord_unit: str,
    bbox_padding_mm: float,
    bbox_mode: str = "aabb",
    use_axis_consistency: bool = False,
    consistency_knn: int = 30,
    axis_weight: float = 0.7,
    normal_weight: float = 0.3,
    use_smart_score: bool = False,
    seed_ratio_threshold: float = 1.3,
    line_sigma_mm: float = 80.0,
    score_axis_weight: float = 0.55,
    score_line_weight: float = 0.30,
    score_continuity_weight: float = 0.15,
    score_knn: int = 30,
    score_iterations: int = 1,
    cleanup_knn: int = 16,
    cleanup_min_same_neighbors: int = 3,
    coarse_center_align: bool = False,
    points_override: np.ndarray | None = None,
    input_source: str | None = None,
) -> None:
    if points_override is None and not pcd_path.exists():
        raise FileNotFoundError(f"点云文件不存在: {pcd_path}")
    if not mesh_dir.exists():
        raise FileNotFoundError(f"Mesh目录不存在: {mesh_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    components_dir = out_dir / "components_pcd"
    components_dir.mkdir(parents=True, exist_ok=True)

    padding = bbox_padding_mm / 1000.0 if coord_unit == "m" else bbox_padding_mm
    report_map = _build_report_map(report_path)
    mesh_infos = _load_mesh_infos(mesh_dir, report_map)
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

    if bbox_mode not in {"aabb", "obb_axis"}:
        raise ValueError(f"不支持的bbox_mode: {bbox_mode}")

    use_smart_score = bool(use_smart_score or use_axis_consistency)
    line_sigma = line_sigma_mm / 1000.0 if coord_unit == "m" else line_sigma_mm

    print(f"[INFO] 包围盒分割: source={source_text}, mesh_count={len(mesh_infos)}")
    print(f"[INFO] bbox_mode = {bbox_mode}")
    print(f"[INFO] bbox_padding = {bbox_padding_mm:.3f} mm -> {padding:.6f} ({coord_unit})")
    if use_smart_score:
        print(
            "[INFO] 启用完整评分分配: "
            f"seed_ratio={seed_ratio_threshold:.2f}, knn={score_knn}, iter={max(int(score_iterations), 1)}"
        )
        print(
            "[INFO] 评分权重: "
            f"axis={score_axis_weight:.2f}, line={score_line_weight:.2f}, continuity={score_continuity_weight:.2f}"
        )

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
        mesh_axis = _mesh_axis_direction(mesh)
        mesh_axes[mid] = mesh_axis
        mesh_axis_points[mid] = np.asarray(mesh.vertices).mean(axis=0)

        aabb = mesh.get_axis_aligned_bounding_box()
        min_bound = np.asarray(aabb.min_bound) - padding
        max_bound = np.asarray(aabb.max_bound) + padding
        mesh_bounds[mid] = (min_bound, max_bound)
        mesh_axis_models[mid] = _build_axis_candidate_model(mesh, padding)

        mask = _build_candidate_mask(
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

    if coarse_center_align and np.all(np.isfinite(mesh_global_min)) and np.all(np.isfinite(mesh_global_max)):
        pcd_min = points.min(axis=0)
        pcd_max = points.max(axis=0)
        pcd_center = 0.5 * (pcd_min + pcd_max)
        mesh_center = 0.5 * (mesh_global_min + mesh_global_max)
        coarse_shift = mesh_center - pcd_center
        points = points + coarse_shift
        pcd.points = o3d.utility.Vector3dVector(points)

        # 粗配准后，候选关系和分配结果需要重算。
        assigned_mesh_id.fill(-1)
        bbox_hit_mask[:] = False
        overlap_candidate_map.clear()

        for mid, bounds in mesh_bounds.items():
            min_bound, max_bound = bounds
            axis_model = mesh_axis_models[mid]
            mask = _build_candidate_mask(
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

    overlap_points = int(len(overlap_candidate_map))
    seed_assigned = 0
    score_assigned = 0
    cleanup_reassigned = 0

    if use_smart_score and overlap_points > 0:
        safe_axis_w = max(0.0, float(score_axis_weight))
        safe_line_w = max(0.0, float(score_line_weight))
        safe_cont_w = max(0.0, float(score_continuity_weight))
        w_sum = safe_axis_w + safe_line_w + safe_cont_w
        if w_sum <= 1e-12:
            safe_axis_w, safe_line_w, safe_cont_w = 0.55, 0.30, 0.15
            w_sum = 1.0
        safe_axis_w /= w_sum
        safe_line_w /= w_sum
        safe_cont_w /= w_sum

        score_knn = max(int(score_knn), 10)
        score_iterations = max(int(score_iterations), 1)
        pcd_for_knn = o3d.geometry.PointCloud()
        pcd_for_knn.points = o3d.utility.Vector3dVector(points)
        kd_tree = o3d.geometry.KDTreeFlann(pcd_for_knn)

        unresolved_points: list[tuple[int, list[int]]] = []
        for pi, cands in overlap_candidate_map.items():
            dist_pairs = []
            p = points[pi]
            for mid in cands:
                axis = mesh_axes.get(mid, np.array([1.0, 0.0, 0.0], dtype=np.float64))
                axis_p = mesh_axis_points.get(mid, np.zeros((3,), dtype=np.float64))
                d = _point_to_axis_distance(p, axis_p, axis)
                dist_pairs.append((d, mid))
            dist_pairs.sort(key=lambda x: x[0])

            if len(dist_pairs) >= 2:
                d1 = max(float(dist_pairs[0][0]), 1e-12)
                d2 = float(dist_pairs[1][0])
                if d2 / d1 >= float(seed_ratio_threshold):
                    assigned_mesh_id[pi] = int(dist_pairs[0][1])
                    seed_assigned += 1
                    continue

            unresolved_points.append((pi, cands))

        for _ in range(score_iterations):
            for pi, cands in unresolved_points:
                p = points[pi]
                local_axis = _estimate_local_axis_for_point(points, kd_tree, pi, score_knn)
                _, nbr_idx, _ = kd_tree.search_knn_vector_3d(p, score_knn)
                nbr_labels = assigned_mesh_id[np.asarray(nbr_idx, dtype=np.int64)] if len(nbr_idx) > 0 else np.array([], dtype=np.int32)
                valid_nbr = nbr_labels[nbr_labels >= 0]
                valid_nbr_count = int(valid_nbr.size)

                best_mid = int(cands[0])
                best_score = -1.0
                for mid in cands:
                    axis = mesh_axes.get(int(mid), np.array([1.0, 0.0, 0.0], dtype=np.float64))
                    axis_p = mesh_axis_points.get(int(mid), np.zeros((3,), dtype=np.float64))
                    axis_parallel = abs(float(np.dot(local_axis, axis)))
                    line_dist = _point_to_axis_distance(p, axis_p, axis)
                    line_score = _line_distance_score(line_dist, line_sigma)
                    if valid_nbr_count > 0:
                        continuity = float(np.count_nonzero(valid_nbr == int(mid)) / valid_nbr_count)
                    else:
                        continuity = 0.0
                    score = safe_axis_w * axis_parallel + safe_line_w * line_score + safe_cont_w * continuity
                    if score > best_score:
                        best_score = score
                        best_mid = int(mid)

                assigned_mesh_id[pi] = best_mid

        score_assigned = int(len(unresolved_points))

        cleanup_knn = max(int(cleanup_knn), 5)
        cleanup_min_same_neighbors = max(int(cleanup_min_same_neighbors), 1)
        for pi, cands in unresolved_points:
            _, nbr_idx, _ = kd_tree.search_knn_vector_3d(points[pi], cleanup_knn)
            if len(nbr_idx) == 0:
                continue
            nbr_labels = assigned_mesh_id[np.asarray(nbr_idx, dtype=np.int64)]
            valid = nbr_labels[nbr_labels >= 0]
            if valid.size == 0:
                continue
            current = int(assigned_mesh_id[pi])
            same_count = int(np.count_nonzero(valid == current))
            if same_count >= cleanup_min_same_neighbors:
                continue

            best_mid = current
            best_cnt = same_count
            for mid in cands:
                cnt = int(np.count_nonzero(valid == int(mid)))
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_mid = int(mid)
            if best_mid != current:
                assigned_mesh_id[pi] = best_mid
                cleanup_reassigned += 1

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

        label_name = _safe_name(str(mi.get("name", mesh_path.stem)))
        guid = _safe_name(str(mi.get("guid", "N_A")))
        ifc_type = _safe_name(str(mi.get("ifc_type", "N_A")))
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
        writer.writerow(
            [
                "mesh_id",
                "ifc_index",
                "ifc_type",
                "name",
                "guid",
                "mesh_file",
                "point_count",
                "segmented_pcd",
            ]
        )
        for row in component_stats:
            writer.writerow(
                [
                    row["mesh_id"],
                    row["ifc_index"],
                    row["ifc_type"],
                    row["name"],
                    row["guid"],
                    row["mesh_file"],
                    row["point_count"],
                    row["segmented_pcd"],
                ]
            )

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
        "line_sigma_in_coord_unit": float(line_sigma),
        "score_axis_weight": float(score_axis_weight),
        "score_line_weight": float(score_line_weight),
        "score_continuity_weight": float(score_continuity_weight),
        "score_knn": int(score_knn),
        "score_iterations": int(score_iterations),
        "cleanup_knn": int(cleanup_knn),
        "cleanup_min_same_neighbors": int(cleanup_min_same_neighbors),
        "overlap_points": int(overlap_points),
        "seed_assigned_overlap_points": int(seed_assigned),
        "scored_assigned_overlap_points": int(score_assigned),
        "cleanup_reassigned_points": int(cleanup_reassigned),
        "resolved_overlap_points": int(seed_assigned + score_assigned),
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

    print(f"[DONE] 包围盒分割完成: components={summary['component_count']}, 非空={summary['components_with_points']}")
    print(f"[DONE] BBox命中点数(并集): {summary['bbox_hit_points_union']} / {summary['input_point_count']} ({summary['bbox_hit_ratio_union']:.2%})")
    print(f"[DONE] 赋值点数(并集): {summary['assigned_points_union']} / {summary['input_point_count']} ({summary['assigned_ratio_union']:.2%})")
    print(
        "[DONE] 重叠点统计: "
        f"overlap={summary['overlap_points']}, "
        f"seed={summary['seed_assigned_overlap_points']}, "
        f"score={summary['scored_assigned_overlap_points']}, "
        f"cleanup={summary['cleanup_reassigned_points']}"
    )
    print(f"[OUT] {components_dir}")
    print(f"[OUT] {label_map_json}")
    print(f"[OUT] {label_map_csv}")
    print(f"[OUT] {summary_path}")


def run_label_transfer(
    pcd_path: Path,
    mesh_dir: Path,
    report_path: Path,
    tolerance_mm: float,
    coord_unit: str,
    out_dir: Path,
    chunk_size: int,
    write_csv: bool,
    compute_device: str,
    distance_backend: str,
    mesh_sample_factor: float,
    mesh_sample_min: int,
    mesh_sample_max: int,
    torch_query_chunk: int,
    torch_ref_chunk: int,
    points_override: np.ndarray | None = None,
    transformed_pcd_path: Path | None = None,
) -> None:
    if not pcd_path.exists():
        raise FileNotFoundError(f"点云文件不存在: {pcd_path}")
    if not mesh_dir.exists():
        raise FileNotFoundError(f"Mesh目录不存在: {mesh_dir}")

    tolerance_in_unit = tolerance_mm / 1000.0 if coord_unit == "m" else tolerance_mm
    out_dir.mkdir(parents=True, exist_ok=True)

    report_map = _build_report_map(report_path)
    mesh_infos = _load_mesh_infos(mesh_dir, report_map)
    if not mesh_infos:
        raise RuntimeError(f"在目录中未找到PLY网格: {mesh_dir}")

    if coord_unit == "m":
        print(f"[INFO] tolerance = {tolerance_mm:.2f} mm ({tolerance_in_unit:.6f} m)")
    else:
        print(f"[INFO] tolerance = {tolerance_mm:.2f} mm ({tolerance_in_unit:.6f} mm-unit)")
    print(f"[INFO] coord_unit = {coord_unit}")
    print(f"[INFO] mesh_count = {len(mesh_infos)}")

    effective_backend = distance_backend
    torch_mod: Any | None = None
    torch_device = "cpu"
    torch_uses_gpu = False
    open3d_device = o3d.core.Device("CPU:0")
    open3d_allow_gpu = False
    effective_device_str = "CPU:0"

    if distance_backend == "torch-nn":
        torch_mod = _try_import_torch()
        if torch_mod is None:
            print("[WARN] 未安装PyTorch，torch-nn不可用，自动回退open3d后端")
            effective_backend = "open3d"
        else:
            torch_device, torch_uses_gpu, effective_device_str = _resolve_torch_device(compute_device, torch_mod)
            print(f"[INFO] distance_backend(request={distance_backend}, effective={effective_backend})")
            print(f"[INFO] compute_device(request={compute_device}, effective={effective_device_str})")
            print(
                "[INFO] torch-nn采样参数: "
                f"factor={mesh_sample_factor}, min={mesh_sample_min}, max={mesh_sample_max}, "
                f"query_chunk={torch_query_chunk}, ref_chunk={torch_ref_chunk}"
            )

    if effective_backend == "open3d":
        open3d_device, open3d_allow_gpu, effective_device_str = _resolve_open3d_device(compute_device)
        print(f"[INFO] distance_backend(request={distance_backend}, effective={effective_backend})")
        print(f"[INFO] compute_device(request={compute_device}, effective={effective_device_str})")

    if points_override is None:
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        points = np.asarray(pcd.points)
    else:
        points = points_override
    if points.size == 0:
        raise RuntimeError("输入点云为空")

    point_count = points.shape[0]
    pcd_min = points.min(axis=0)
    pcd_max = points.max(axis=0)

    best_dist = np.full(point_count, np.inf, dtype=np.float64)
    best_mesh_id = np.full(point_count, -1, dtype=np.int32)

    total_bbox_candidates = 0
    evaluated_meshes = 0
    gpu_evaluated_meshes = 0
    gpu_fallback_happened = False
    mesh_global_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    mesh_global_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    for mi in mesh_infos:
        mesh_path = mi["mesh_path"]
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if mesh.is_empty():
            print(f"[WARN] 跳过空网格: {mesh_path.name}")
            continue

        aabb = mesh.get_axis_aligned_bounding_box()
        mesh_global_min = np.minimum(mesh_global_min, np.asarray(aabb.min_bound))
        mesh_global_max = np.maximum(mesh_global_max, np.asarray(aabb.max_bound))

        min_bound = np.asarray(aabb.min_bound) - tolerance_in_unit
        max_bound = np.asarray(aabb.max_bound) + tolerance_in_unit

        mask = np.all((points >= min_bound) & (points <= max_bound), axis=1)
        candidate_indices = np.flatnonzero(mask)
        if candidate_indices.size == 0:
            continue

        total_bbox_candidates += int(candidate_indices.size)

        distances = np.empty(candidate_indices.size, dtype=np.float64)
        if effective_backend == "torch-nn" and torch_mod is not None:
            mesh_ref_points = _sample_mesh_points_for_torch_nn(
                mesh=mesh,
                sample_factor=mesh_sample_factor,
                sample_min=mesh_sample_min,
                sample_max=mesh_sample_max,
            )
            if mesh_ref_points.size == 0:
                continue
            if torch_uses_gpu:
                gpu_evaluated_meshes += 1

            start = 0
            query_chunk = max(int(torch_query_chunk), 256)
            while start < candidate_indices.size:
                end = min(start + query_chunk, candidate_indices.size)
                part_idx = candidate_indices[start:end]
                part_points = points[part_idx]
                distances[start:end] = _torch_nn_min_distance(
                    query_points=part_points,
                    ref_points=mesh_ref_points,
                    torch_mod=torch_mod,
                    torch_device=torch_device,
                    ref_chunk=torch_ref_chunk,
                )
                start = end
        else:
            tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            scene_device = o3d.core.Device("CPU:0")
            scene_uses_gpu = False
            if open3d_allow_gpu:
                try:
                    scene = o3d.t.geometry.RaycastingScene(device=open3d_device)
                    _ = scene.add_triangles(tmesh.to(open3d_device))
                    scene_device = open3d_device
                    scene_uses_gpu = True
                    gpu_evaluated_meshes += 1
                except Exception as ex:
                    if not gpu_fallback_happened:
                        print(f"[WARN] GPU距离计算初始化失败，回退CPU。原因: {ex}")
                    gpu_fallback_happened = True
                    scene = o3d.t.geometry.RaycastingScene()
                    _ = scene.add_triangles(tmesh)
            else:
                scene = o3d.t.geometry.RaycastingScene()
                _ = scene.add_triangles(tmesh)

            start = 0
            while start < candidate_indices.size:
                end = min(start + chunk_size, candidate_indices.size)
                part_idx = candidate_indices[start:end]
                query_points = o3d.core.Tensor(
                    points[part_idx], dtype=o3d.core.Dtype.Float32, device=scene_device
                )
                part_dist = scene.compute_distance(query_points)
                if scene_uses_gpu:
                    part_dist = part_dist.to(o3d.core.Device("CPU:0"))
                distances[start:end] = part_dist.numpy().astype(np.float64)
                start = end

        previous = best_dist[candidate_indices]
        improve_mask = distances < previous
        if np.any(improve_mask):
            improved_indices = candidate_indices[improve_mask]
            best_dist[improved_indices] = distances[improve_mask]
            best_mesh_id[improved_indices] = int(mi["mesh_id"])

        evaluated_meshes += 1

    assigned_mask = (best_mesh_id >= 0) & (best_dist <= tolerance_in_unit)
    assigned_count = int(np.count_nonzero(assigned_mask))
    unassigned_count = int(point_count - assigned_count)

    info_by_mesh_id = {int(mi["mesh_id"]): mi for mi in mesh_infos}

    colors = np.full((point_count, 3), 0.6, dtype=np.float64)
    unique_ids = np.unique(best_mesh_id[assigned_mask])
    for mesh_id in unique_ids:
        idx_mask = assigned_mask & (best_mesh_id == mesh_id)
        colors[idx_mask] = _color_for_label(int(mesh_id))

    labeled_pcd = o3d.geometry.PointCloud()
    labeled_pcd.points = o3d.utility.Vector3dVector(points)
    labeled_pcd.colors = o3d.utility.Vector3dVector(colors)

    labeled_pcd_path = out_dir / "labeled_points_colored.pcd"
    o3d.io.write_point_cloud(str(labeled_pcd_path), labeled_pcd)

    labels_npz_path = out_dir / "labeled_points_arrays.npz"
    np.savez_compressed(
        labels_npz_path,
        points=points.astype(np.float32),
        assigned=assigned_mask.astype(np.uint8),
        best_mesh_id=best_mesh_id,
        best_distance_m=np.where(np.isfinite(best_dist), best_dist, -1.0).astype(np.float32),
    )

    csv_path = out_dir / "labeled_points.csv"
    if write_csv:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "x",
                    "y",
                    "z",
                    "assigned",
                    "mesh_id",
                    "ifc_index",
                    "ifc_type",
                    "name",
                    "guid",
                    "profile",
                    "mesh_file",
                    "distance_m",
                    "distance_mm",
                ]
            )

            for i in range(point_count):
                mesh_id = int(best_mesh_id[i])
                is_assigned = bool(assigned_mask[i])
                if is_assigned:
                    meta = info_by_mesh_id.get(mesh_id, {})
                    ifc_index = meta.get("index", -1)
                    ifc_type = meta.get("ifc_type", "N/A")
                    name = meta.get("name", "N/A")
                    guid = meta.get("guid", "N/A")
                    profile = meta.get("profile", "N/A")
                    mesh_file = str(meta.get("mesh_path", ""))
                    dist_m = float(best_dist[i])
                    dist_mm = dist_m * 1000.0
                else:
                    ifc_index = -1
                    ifc_type = "UNASSIGNED"
                    name = "UNASSIGNED"
                    guid = "UNASSIGNED"
                    profile = "UNASSIGNED"
                    mesh_file = ""
                    dist_m = float(best_dist[i]) if np.isfinite(best_dist[i]) else -1.0
                    dist_mm = dist_m * 1000.0 if dist_m >= 0.0 else -1.0

                writer.writerow(
                    [
                        float(points[i, 0]),
                        float(points[i, 1]),
                        float(points[i, 2]),
                        int(is_assigned),
                        mesh_id,
                        ifc_index,
                        ifc_type,
                        name,
                        guid,
                        profile,
                        mesh_file,
                        dist_m,
                        dist_mm,
                    ]
                )

    counts_by_mesh = {}
    for mesh_id in unique_ids:
        mesh_id_int = int(mesh_id)
        count = int(np.count_nonzero(assigned_mask & (best_mesh_id == mesh_id_int)))
        meta = info_by_mesh_id.get(mesh_id_int, {})
        counts_by_mesh[str(mesh_id_int)] = {
            "count": count,
            "ifc_index": int(meta.get("index", -1)),
            "ifc_type": str(meta.get("ifc_type", "N/A")),
            "name": str(meta.get("name", "N/A")),
            "guid": str(meta.get("guid", "N/A")),
            "mesh_file": str(meta.get("mesh_path", "")),
        }

    summary = {
        "pcd_path": str(pcd_path),
        "pcd_transformed_path": str(transformed_pcd_path) if transformed_pcd_path else "",
        "mesh_dir": str(mesh_dir),
        "report_path": str(report_path),
        "tolerance_mm": float(tolerance_mm),
        "coord_unit": coord_unit,
        "tolerance_in_coord_unit": float(tolerance_in_unit),
        "point_count": int(point_count),
        "assigned_count": assigned_count,
        "unassigned_count": unassigned_count,
        "assigned_ratio": float(assigned_count / point_count),
        "mesh_count": int(len(mesh_infos)),
        "evaluated_meshes": int(evaluated_meshes),
        "gpu_evaluated_meshes": int(gpu_evaluated_meshes),
        "distance_backend_request": distance_backend,
        "distance_backend_effective": effective_backend,
        "compute_device_request": compute_device,
        "compute_device_effective": effective_device_str,
        "gpu_fallback_happened": bool(gpu_fallback_happened),
        "mesh_sample_factor": float(mesh_sample_factor),
        "mesh_sample_min": int(mesh_sample_min),
        "mesh_sample_max": int(mesh_sample_max),
        "torch_query_chunk": int(torch_query_chunk),
        "torch_ref_chunk": int(torch_ref_chunk),
        "total_bbox_candidates": int(total_bbox_candidates),
        "mean_candidate_per_mesh": float(total_bbox_candidates / max(evaluated_meshes, 1)),
        "outputs": {
            "colored_pcd": str(labeled_pcd_path),
            "labeled_arrays_npz": str(labels_npz_path),
            "labeled_csv": str(csv_path) if write_csv else "",
        },
        "counts_by_mesh": counts_by_mesh,
    }

    summary_path = out_dir / "label_transfer_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] 点云总数: {point_count}")
    print(f"[DONE] 已赋标签: {assigned_count} ({assigned_count / point_count:.2%})")
    print(f"[DONE] 未赋标签: {unassigned_count}")

    if np.all(np.isfinite(mesh_global_min)) and np.all(np.isfinite(mesh_global_max)):
        overlap = np.array(
            [
                _overlap_1d(pcd_min[i], pcd_max[i], mesh_global_min[i], mesh_global_max[i])
                for i in range(3)
            ],
            dtype=np.float64,
        )
        print(f"[INFO] PCD bbox min={pcd_min}, max={pcd_max}")
        print(f"[INFO] Mesh bbox min={mesh_global_min}, max={mesh_global_max}")
        print(f"[INFO] BBox overlap xyz={overlap}")
        if np.any(overlap <= 0.0):
            print("[WARN] 点云与Mesh全局包围盒在至少一个轴向无重叠，可能仍未在同一坐标系。")

    finite_dist = best_dist[np.isfinite(best_dist)]
    if finite_dist.size > 0:
        print(
            "[INFO] 最近距离统计: "
            f"min={float(np.min(finite_dist)):.6f}, "
            f"p50={float(np.percentile(finite_dist, 50)):.6f}, "
            f"p95={float(np.percentile(finite_dist, 95)):.6f}"
        )
    if assigned_count == 0:
        print(
            "[WARN] 当前阈值下未匹配到任何点。若模型/点云坐标单位为毫米，请尝试 --coord-unit mm；"
            "或适当增大 --tolerance-mm。"
        )

    print(f"[OUT] {labeled_pcd_path}")
    print(f"[OUT] {labels_npz_path}")
    if write_csv:
        print(f"[OUT] {csv_path}")
    print(f"[OUT] {summary_path}")


def main() -> None:
    args = parse_args()
    if args.input_txt is None and not args.pcd.exists():
        raise FileNotFoundError(f"点云文件不存在: {args.pcd}")
    if not args.mesh_dir.exists():
        raise FileNotFoundError(f"Mesh目录不存在: {args.mesh_dir}")

    if args.bbox_split_only:
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        split_out_dir = args.out_dir / f"subresult_bbox_split_{run_stamp}"
        points_override = None
        input_source = None
        if args.input_txt is not None:
            raw_txt_points = _load_points_from_txt(args.input_txt)
            raw_count = int(raw_txt_points.shape[0])
            ds_points = _voxel_downsample_points(raw_txt_points, args.txt_voxel)
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
            o3d.io.write_point_cloud(str(ds_pcd_path), ds_pcd)
            print(f"[OUT] {ds_pcd_path}")

        run_bbox_component_split(
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
        )
        return

    transform = _load_transform_matrix(args.transform_matrix_file, args.transform_matrix_values)
    print("[INFO] 使用4x4变换矩阵:")
    print(transform)

    raw_pcd = o3d.io.read_point_cloud(str(args.pcd))
    raw_points = np.asarray(raw_pcd.points)
    if raw_points.size == 0:
        raise RuntimeError("输入点云为空")

    transformed_points = _apply_transform(raw_points, transform)

    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)
    transformed_pcd_path = args.out_dir / "scan_transformed_for_label_transfer.pcd"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(transformed_pcd_path), transformed_pcd)
    print(f"[OUT] {transformed_pcd_path}")

    preview_summary = _build_alignment_preview(
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
        run_label_transfer(
            pcd_path=args.pcd,
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