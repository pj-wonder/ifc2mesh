from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_expected_module_directories_exist() -> None:
    for relative in ("pipelines", "segmentation", "axis_fit", "design_nodes"):
        assert (ROOT / relative).is_dir()


def test_compat_wrappers_point_to_new_modules() -> None:
    wrapper_expectations = {
        "ifc2mesh.py": "from pipelines.ifc2mesh import *",
        "mesh_pcd_segment_obb_axis_v2.py": "from segmentation.mesh_pcd_segment_obb_axis_v2 import *",
        "run_axis_merge_and_node_fit_v3.py": "from axis_fit.run_axis_merge_and_node_fit_v3 import *",
        "dxfline2point/line2point.py": "from design_nodes.line2point import *",
    }
    for relative, expected in wrapper_expectations.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert expected in text
