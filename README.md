# ifc2mesh

`ifc2mesh` 用于将 IFC 结构模型转换为可处理的三角网格与点云语义信息，并进一步完成轴线合并、节点拟合以及设计节点对比分析。

## 功能概览

- IFC 转网格与语义报告
- 点云与网格的配准后标签转移
- 基于轴线一致性与表面距离的构件分割
- 构件轴线合并与节点拟合
- DXF 线结构提取设计轴线与设计节点

## 目录说明

- `ifc2mesh.py`：IFC 转网格、导出语义报告
- `mesh_pcd_segment.py`：点云到网格的基础语义转移
- `mesh_pcd_segment_obb_axis.py`：加入轴线一致性约束的分割流程
- `mesh_pcd_segment_obb_axis_v2.py`：进一步加入表面距离与法向约束
- `run_axis_merge_and_node_fit_v3.py`：轴线合并与节点拟合主流程
- `dxfline2point/line2point.py`：DXF 设计轴线与设计节点生成

## 依赖

- Python 3.10+
- `ifcopenshell`
- `open3d`
- `numpy`
- `matplotlib`（部分可视化）
- `torch`（仅在启用 GPU 距离后端时需要）

## 建议

- 代码仓库建议只保留脚本、少量示例配置和说明文档。
- 大体量点云、网格、IFC、DXF 与中间结果建议存放在仓库外部，或后续改用 Git LFS 管理。
