ifc2mesh 文件夹概要（2026-05-08）

一、用途概览
- 目标：将 IFC 结构模型转成可处理的三角网格与点云，并把构件语义转移到实测点云上，后续做构件轴线合并与节点拟合。
- 主流程：IFC -> 网格导出 + 语义报告 -> 点云与网格配准 -> 点云/网格邻近性分割与标签转移 -> 轴线合并与节点拟合 -> 可视化。

二、核心脚本与职责
- IFC 转网格与报告：ifc2mesh/ifc2mesh.py
	- 调用 IfcOpenShell + Open3D 导出 PLY 网格、记录构件语义字段、并做拓扑分布检查（用于判断是否在世界坐标系）。
- IFC 轴线/节点抽取：ifc2mesh/ifc2keypoint.py
	- 从 IfcBeam/IfcColumn 的 Axis 表示或 ObjectPlacement 中提取标准化几何信息。
	- 导出轴线与节点 CSV，并可使用 matplotlib 做简易可视化。
- 点云-网格语义转移：
	- ifc2mesh/mesh_pcd_segment.py：包围盒粗筛 + 距离场精算，支持 AABB/OBB 筛选、GPU/CPU、双阈值运行等。
	- ifc2mesh/mesh_pcd_segment_obb_axis.py：在包围盒候选中加入“轴线一致性”判别，降低交叉杆件误分割。
	- ifc2mesh/mesh_pcd_segment_obb_axis_v2.py：在 v1 基础上加入表面距离 + 法向 + 精确 Mesh 裁决的流程。
- 轴线合并与节点拟合：
	- ifc2mesh/axis_node_exaction/run_axis_merge_and_node_fit.py
		- 合并近似共线构件轴线，估计交点节点，并导出结果到 node_fit_results。
	- ifc2mesh/axis_node_exaction/visualize_fitted_nodes.py
		- 读取合并轴线与节点 CSV，输出 Open3D/Matplotlib 可视化。

三、输入输出与目录约定
- IFC 与网格输出：在 ifc2mesh/ifc2mesh.py 中生成 mesh 与报告 JSON（默认输出目录参考命令行参数）。
- 标签转移输出：默认输出到 result_run_ifc311_world/label_transfer 或 label_transfer_v2。
- 节点拟合输出：默认输出到 ifc2mesh/node_fit_results。
- 结果目录规范详见 ifc2mesh/RESULT_LOCATION_NOTE.txt。

四、配准与辅助数据
- 文件夹中包含配准矩阵与中间数据（例如 segment_pcd.pcd、原始 ply/pcd 配准文件）。
- 可通过命令行参数传入 CloudCompare 的 4x4 变换矩阵，以将点云变换到 Mesh 坐标系。

五、依赖与环境要点
- Python 主要依赖：IfcOpenShell、Open3D、NumPy；部分可视化依赖 Matplotlib。
- 如果使用 GPU 距离后端（torch-nn），需要对应的 PyTorch 环境。

六、常见注意事项
- 网格导出阶段需关注是否启用了 use-world-coords 以避免局部坐标问题。
- 点云坐标单位需要与 Mesh 一致（m 或 mm），否则阈值含义会错。
- 大点云建议使用分块距离计算与可视化下采样参数。

七、dxfline2point（DXF 线结构 -> 设计节点）
- 目标：从 DXF 中抽取设计轴线段，并据轴线交点生成设计节点，用于与实测节点对比偏差。
- 输入：DXF 文件（支持 LINE、LWPOLYLINE、POLYLINE 实体）。
- 核心脚本：ifc2mesh/dxfline2point/line2point.py
	- 轴线抽取：读取 DXF modelspace，提取线段并按最小长度过滤（min_axis_length）。
	- 节点生成：计算轴线段两两最近点，距离小于阈值且投影参数在段内即视为交点。
	- 节点融合：按容差对交点聚类去重（node_merge_tolerance），并统计关联轴线数。
	- 多轴交点：当一个节点关联轴线数 >= 3 时，用最小二乘伪交点优化节点位置。
	- 输出：轴线 CSV 与节点 CSV，可进一步做缩放、筛选、与实测节点比对。
- 偏差分析与可视化：
	- ifc2mesh/dxfline2point/view_node_deviation_open3d.py
		- 读取偏差报告 JSON，加载设计节点与实测节点，Open3D 交互展示偏差向量。
	- line2point.py 内置 Matplotlib/Open3D 可视化函数，适合快速检查轴线与节点分布。
