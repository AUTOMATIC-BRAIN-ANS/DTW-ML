"""
Stage `beats`: Beat-to-beat metrics (SPO, SPP, DN, DPP, RR): {condition}/..._PP_F/{ABP,CBFV}.csv
Public interface: ComputeBeatMetrics (stage.py), run by the orchestrator or on its own.
"""

from ddtw_ml.stages.beats.stage import ComputeBeatMetrics

__all__ = ["ComputeBeatMetrics"]
