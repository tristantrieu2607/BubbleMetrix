"""Global configuration and calibration constants for bubble sizing."""

from dataclasses import dataclass, field
from typing import Tuple

IGNORE_OVERLAP_THRESHOLD = 0.5

@dataclass
class BubbleConfig:
    # Optical calibration for pulp probe
    # 1024 px = 3.37 mm field of view => 303.858 px/mm => 3.291 um/px
    px_per_mm: float = 303.858
    um_per_px: float = 1000.0 / 303.858  # 3.29098... um

    # Microbubble spectrum constraints (in pixels)
    # 12 px = ~39.5 um (just above window fouling specks)
    # 110 px = ~362 um (largest single in-focus bubbles before coalescence)
    min_diameter_px: float = 12.0
    max_diameter_px: float = 250.0   # labels reach ~206 px and recall at 140+ px is 1.0; 120 cut real bubbles from measure_video

    # Optical FOV geometry
    fov_center_xy: Tuple[int, int] = (524, 508)
    fov_radius_px: int = 601

    # RPN anchors, by named scheme -- see models/mask_rcnn_v2.ANCHOR_SCHEMES.
    # The scheme is saved into every checkpoint and read back at load time, so a
    # model is always rebuilt with the anchors it was trained on. Checkpoints
    # without the field (everything before 2026-09-29) load as "v1".
    anchor_scheme: str = "round3"

    # Target class configuration
    num_classes: int = 2  # Background (0), Bubble (1)

    # Frame extraction decorrelation parameters
    rotor_frame_gap: int = 5
    venturi_frame_gap: int = 20

    # Training parameters
    batch_size: int = 2
    num_workers: int = 2
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    num_epochs: int = 12          # val loss bottomed at epoch 10 and rose after (runs/rotor_raw)

    # Scale jitter, training only. Log-uniform zoom so the ~73 px median bubble also
    # appears at 100-140 px, the band the detector starves in. Instances pushed
    # below the 12 px floor move into the ignore mask.
    scale_jitter_prob: float = 0.5
    scale_jitter_range: Tuple[float, float] = (0.75, 2.0)
