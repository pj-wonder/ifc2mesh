from __future__ import annotations

import argparse
import importlib
from typing import Dict, Sequence


COMMANDS: Dict[str, tuple[str, str]] = {
    "ifc-export": ("pipelines.ifc2mesh", "Export IFC elements to meshes and semantic reports."),
    "extract-components": ("pipelines.extract_component_pcds", "Split transferred labels into per-component point clouds."),
    "segment-basic": ("segmentation.mesh_pcd_segment", "Transfer labels from meshes to point clouds."),
    "segment-axis": ("segmentation.mesh_pcd_segment_obb_axis", "Segment with axis-consistency filtering."),
    "segment-axis-v2": ("segmentation.mesh_pcd_segment_obb_axis_v2", "Segment with axis, surface-distance, and normal checks."),
    "fit-axis-nodes": ("axis_fit.run_axis_merge_and_node_fit_v3", "Merge component axes and fit junction nodes."),
    "fit-curves": ("axis_fit.run_axis_merge_and_node_fit_curve_v2_0", "Fit curved axes from segmented components."),
    "fit-curves-prior": ("axis_fit.run_axis_merge_and_node_fit_curve_v2_prior", "Fit curved axes with prior-assisted mode selection."),
    "design-nodes": ("design_nodes.line2point", "Extract design axes and nodes from DXF files."),
    "design-dxf2ply": ("design_nodes.dxf2ply.dxf2ply", "Sample DXF linework to PLY and compare against measured nodes."),
    "design-view-deviation": ("design_nodes.view_node_deviation_open3d", "Open an Open3D viewer for node deviation reports."),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified CLI for IFC export, segmentation, axis fitting, and design-node workflows."
    )
    parser.add_argument("command", choices=sorted(COMMANDS.keys()), help="Pipeline command to run.")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the selected command.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    parsed = parser.parse_args(argv)
    forwarded = list(parsed.args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    module_name = COMMANDS[parsed.command][0]
    module = importlib.import_module(module_name)
    module.main(forwarded)


if __name__ == "__main__":
    main()
