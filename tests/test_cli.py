from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cli


def test_cli_exposes_expected_commands() -> None:
    expected = {
        "ifc-export",
        "extract-components",
        "segment-basic",
        "segment-axis",
        "segment-axis-v2",
        "fit-axis-nodes",
        "fit-curves",
        "fit-curves-prior",
        "design-nodes",
        "design-dxf2ply",
        "design-view-deviation",
    }
    assert expected.issubset(set(cli.COMMANDS))


def test_cli_forwards_arguments(monkeypatch) -> None:
    called = {}

    class FakeModule:
        @staticmethod
        def main(argv=None) -> None:
            called["argv"] = list(argv or [])

    def fake_import(name: str):
        called["module"] = name
        return FakeModule

    monkeypatch.setattr(cli.importlib, "import_module", fake_import)
    monkeypatch.setitem(cli.COMMANDS, "ifc-export", ("pipelines.ifc2mesh", "fake"))
    cli.main(["ifc-export", "--", "--help"])
    assert called["module"] == "pipelines.ifc2mesh"
    assert called["argv"] == ["--help"]
