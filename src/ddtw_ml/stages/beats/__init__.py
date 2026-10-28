"""
Stage `beats`: Beat-to-beat metrics (SPO, SPP, DN, DPP, RR): {condition}/..._PP_F/{ABP,CBFV}.csv
Public interface: ComputeBeatMetrics (stage.py), run by the orchestrator or on its own; FromSignal (metrics.py), the
beat detection, also used for the SNR of the beats in recordings_metadata.
"""

from ddtw_ml.stages.beats.metrics import FromSignal
from ddtw_ml.stages.beats.stage import ComputeBeatMetrics

__all__ = ["ComputeBeatMetrics", "FromSignal"]
