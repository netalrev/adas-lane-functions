"""
src/pipeline/__init__.py
==========================
Batch orchestration layer for pipeline_input.py: segment discovery, one-time
inference-engine construction, and the per-segment frame loop.
"""
from .segment_resolver import resolve_segments
from .engine_factory   import build_engines, PipelineEngines
from .segment_runner    import SegmentRunner

__all__ = [
    "resolve_segments",
    "build_engines",
    "PipelineEngines",
    "SegmentRunner",
]
