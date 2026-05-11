from __future__ import annotations

"""Compatibility shim for the historical curve-v2 import path.

The repository currently keeps the maintained implementation in
`run_axis_merge_and_node_fit_curve_v2_0.py`, while some workflows still
import `run_axis_merge_and_node_fit_curve_v2`.
"""

from axis_fit.run_axis_merge_and_node_fit_curve_v2_0 import *  # noqa: F401,F403
