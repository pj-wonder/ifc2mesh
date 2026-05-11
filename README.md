# ifc2mesh

`ifc2mesh` is a script-oriented toolbox for converting IFC structural models into
triangle meshes, transferring component semantics onto measured point clouds,
and fitting merged axes and junction nodes for downstream analysis.

## What the repository contains

- `ifc2mesh.py`: export IFC elements as mesh files plus a semantic report
- `mesh_pcd_segment.py`: transfer labels from mesh components onto point clouds
- `mesh_pcd_segment_obb_axis.py`: segmentation with axis-consistency filtering
- `mesh_pcd_segment_obb_axis_v2.py`: segmentation with axis, surface-distance,
  and normal constraints
- `extract_component_pcds.py`: split labeled point clouds into per-component files
- `run_axis_merge_and_node_fit_v3.py`: merge nearby component axes and fit nodes
- `run_axis_merge_and_node_fit_curve_v2_0.py`: curve-aware axis fitting workflow
- `run_axis_merge_and_node_fit_curve_v2_prior.py`: prior-assisted curved-axis workflow
- `dxfline2point/line2point.py`: extract design axes and nodes from DXF files
- `dxfline2point/dxf2ply/dxf2ply.py`: sample DXF linework into PLY and compare
  against measured nodes

## Recommended environment

- Python 3.10 or newer
- Core dependencies from `requirements.txt`
- Optional GPU backend: `torch`

Install the common environment with:

```bash
pip install -r requirements.txt
```

For development utilities:

```bash
pip install -r requirements-dev.txt
```

## Repository layout

The codebase is still primarily organized as executable scripts, but it now
includes a small amount of project scaffolding to make long-term maintenance
easier:

- `requirements.txt` / `requirements-dev.txt`: reproducible environments
- `pyproject.toml`: project metadata and test-path configuration
- `LICENSE`: repository license
- `dxfline2point/__init__.py`: package markers for DXF utilities
- `run_axis_merge_and_node_fit_curve_v2.py`: compatibility shim for older imports

## Typical workflow

1. Run `ifc2mesh.py` to export IFC meshes and semantic metadata.
2. Align measured point clouds into the mesh coordinate frame.
3. Run one of the `mesh_pcd_segment*.py` scripts to transfer labels.
4. Extract component-level point clouds when needed.
5. Run the axis/node fitting scripts to merge axes and estimate junctions.
6. Use the DXF utilities when comparing design nodes against measured nodes.

## Notes on large data

This repository intentionally tracks code and lightweight documentation only.
Large IFC, DXF, PLY, PCD, and result folders should stay outside the repo or be
managed with Git LFS if they must be versioned later.
