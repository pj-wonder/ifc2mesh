# ifc2mesh

`ifc2mesh` is a script-oriented toolbox for converting IFC structural models into
triangle meshes, transferring component semantics onto measured point clouds,
and fitting merged axes and junction nodes for downstream analysis.

## What the repository contains

- `pipelines/`: IFC export and preprocessing workflows
- `segmentation/`: point-cloud and mesh label-transfer workflows
- `axis_fit/`: axis merging and node fitting workflows
- top-level script names such as `ifc2mesh.py` and `mesh_pcd_segment.py`: compatibility entry points
- `dxfline2point/`: DXF-based design-axis and design-node utilities

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

The implementation files now live in module directories, while the original
top-level filenames remain as thin wrappers so existing commands do not break:

- `pipelines/ifc2mesh.py`: IFC mesh export implementation
- `pipelines/extract_component_pcds.py`: per-component point-cloud extraction
- `segmentation/mesh_pcd_segment*.py`: segmentation implementations
- `axis_fit/run_axis_merge_and_node_fit*.py`: axis and node fitting implementations
- `requirements.txt` / `requirements-dev.txt`: reproducible environments
- `pyproject.toml`: project metadata and test-path configuration
- `LICENSE`: repository license
- `dxfline2point/__init__.py`: package markers for DXF utilities

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
