"""MicroBrain 0.9.52 FIXED current decoder.

This module deliberately restores the multi-scale temporal decoder as the active
path.  The simulator's event clock is still native 1 ms; *decoding* compares the
same recurrent route-event train simultaneously at 0.5, 1.0, 2.0, and 4.0 ms
scales.  Do not collapse this back to a single 1 ms token representation.

The historical 0.9.38 causal-birth dictionary that reached 30/30 on one fresh
30-word development validation cohort is preserved under history/, but it is not
exported here as the current architecture because it discards multi-scale route
timing information.
"""
from multiscale_temporal_decoder import (
    MultiScaleTemporalMemory,
    SCALES,
    SCALE_WEIGHTS,
    score,
    temporal_signature_similarity,
)

EXPECTED_SCALES = (0.50, 1.00, 2.00, 4.00)
if tuple(SCALES) != EXPECTED_SCALES:
    raise RuntimeError(f"Temporal-scale regression: expected {EXPECTED_SCALES}, got {SCALES}")

CurrentDevelopmentDecoder = MultiScaleTemporalMemory
CurrentDecoder = MultiScaleTemporalMemory
