#!/usr/bin/env python3
"""
从标签转移结果的 npz 数组提取各构件的独立点云 PCD。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def extract_component_pcds(
    label_transfer_dir: Path,
    output_subdir: str = "component_pcds",
) -> None:
    """
    从 labeled_points_arrays.npz 提取各构件独立点云。
    
    Args:
        label_transfer_dir: 标签转移输出目录（包含 labeled_points_arrays.npz）
        output_subdir: 输出子目录名称
    """
    label_transfer_dir = Path(label_transfer_dir).resolve()
    if not label_transfer_dir.exists():
        raise FileNotFoundError(f"标签转移目录不存在: {label_transfer_dir}")

    npz_path = label_transfer_dir / "labeled_points_arrays.npz"
    summary_path = label_transfer_dir / "label_transfer_summary.json"

    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ 文件不存在: {npz_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"汇总 JSON 不存在: {summary_path}")

    # 读取点云数据与标签
    data = np.load(npz_path)
    points = data["points"].astype(np.float64)
    assigned = data["assigned"].astype(bool)
    best_mesh_id = data["best_mesh_id"].astype(np.int32)

    # 读取构件元数据
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    counts_by_mesh = summary.get("counts_by_mesh", {})

    # 创建输出目录
    out_dir = label_transfer_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 逐构件输出 PCD
    saved_count = 0
    for mesh_id_str, meta in counts_by_mesh.items():
        mesh_id = int(mesh_id_str)
        ifc_index = meta.get("ifc_index", -1)
        ifc_type = meta.get("ifc_type", "N/A")
        name = meta.get("name", "N/A")
        guid = meta.get("guid", "N/A")
        count = meta.get("count", 0)

        if count == 0:
            continue

        # 提取该构件的点
        mask = (best_mesh_id == mesh_id) & assigned
        if not np.any(mask):
            print(f"[SKIP] mesh_id={mesh_id} 未找到有效点")
            continue

        component_points = points[mask]

        # 清理中文字符，生成文件名
        safe_name = name.encode("utf-8", errors="ignore").decode("utf-8")
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in safe_name)[:80]
        safe_guid = guid[:16] if guid else "NOGUID"

        pcd_filename = f"{ifc_index:04d}_{ifc_type}_{safe_name}_{safe_guid}.pcd"
        pcd_path = out_dir / pcd_filename

        # 写出 PCD
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(component_points)
        o3d.io.write_point_cloud(str(pcd_path), pcd)

        print(f"[OUT] {pcd_filename} -> {count} points")
        saved_count += 1

    print(f"\n[DONE] 提取完成: {saved_count} 个构件的独立 PCD 已保存至 {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从标签转移 NPZ 提取各构件的独立点云 PCD"
    )
    parser.add_argument(
        "--label-transfer-dir",
        type=Path,
        required=True,
        help="标签转移输出目录（包含 labeled_points_arrays.npz 和 label_transfer_summary.json）",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="component_pcds",
        help="输出子目录名称，默认 component_pcds",
    )

    args = parser.parse_args()
    extract_component_pcds(args.label_transfer_dir, args.output_subdir)


if __name__ == "__main__":
    main()
