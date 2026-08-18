"""Observation helpers for world model training.

The recommended import path for ``preprocess_obs`` is
``torchwm.utils.preprocess_obs`` (or ``torchwm.preprocess_obs``).
"""

from torchwm.observations.dreamer_v1_obs import (
    ObservationModel,
    SymbolicObservationModel,
    VisualObservationModel,
)

__all__ = [
    "ObservationModel",
    "SymbolicObservationModel",
    "VisualObservationModel",
]
