"""Pytest verification suite for BubMask core modules."""

import pytest
import numpy as np
import cv2
from bubble_rcnn.config import BubbleConfig
from bubble_rcnn.reconstruction.circle_fit import ShapeReconstructor
from bubble_rcnn.sizing.metrology import BubbleMetrology

def test_config_calibration():
    cfg = BubbleConfig()
    assert abs(cfg.px_per_mm - 303.858) < 1e-3
    assert abs(cfg.um_per_px - 3.291) < 1e-2
    from bubble_rcnn.models.mask_rcnn_v2 import ANCHOR_SCHEMES
    assert cfg.anchor_scheme in ANCHOR_SCHEMES
    assert all(len(sz) == 5 for sz, _ in ANCHOR_SCHEMES.values())

def test_shape_reconstructor_circle():
    # Synthetic circle of radius 20 at center (50, 50)
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 20, 1, -1)
    
    rec = ShapeReconstructor.reconstruct_from_binary_mask(mask, method="min_enclosing_circle")
    assert rec is not None
    assert rec["shape_type"] == "circle"
    assert abs(rec["diameter_px"] - 40.0) <= 2.0
    assert abs(rec["center_xy"][0] - 50.0) <= 1.0
    assert abs(rec["center_xy"][1] - 50.0) <= 1.0

def test_shape_reconstructor_ellipse_fallback_on_few_points():
    # 2x2 mask has 4 contour points (< 5): fit_ellipse must fall back to min enclosing circle
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:12, 10:12] = 1

    rec = ShapeReconstructor.reconstruct_from_binary_mask(mask, method="fit_ellipse")
    assert rec is not None
    assert rec["shape_type"] == "circle"  # Fell back cleanly

def test_metrology_sauter_d32():
    met = BubbleMetrology()
    # If all bubbles have same diameter d, D32 must be d
    d32 = met.compute_sauter_d32([100.0, 100.0, 100.0])
    assert abs(d32 - 100.0) < 1e-5

    # Hand calculation: d = [10, 20]
    # sum(d^3) = 1000 + 8000 = 9000
    # sum(d^2) = 100 + 400 = 500
    # D32 = 9000 / 500 = 18.0
    d32_two = met.compute_sauter_d32([10.0, 20.0])
    assert abs(d32_two - 18.0) < 1e-5

def test_model_anchor_replacement():
    import torch
    from bubble_rcnn.models.mask_rcnn_v2 import build_bubble_maskrcnn
    
    cfg = BubbleConfig()
    model = build_bubble_maskrcnn(config=cfg, pretrained=False)
    
    # Check anchor generator configuration
    assert len(model.rpn.anchor_generator.sizes) == 5
    assert model.rpn.anchor_generator.sizes[0] == (8,)
    assert model.rpn.anchor_generator.sizes[-1] == (128,)

    # Run dummy forward pass
    model.eval()
    x = [torch.rand(3, 512, 512)]
    with torch.no_grad():
        out = model(x)
    assert "boxes" in out[0]
    assert "masks" in out[0]

def test_no_resize_transform():
    import torch
    from bubble_rcnn.models.mask_rcnn_v2 import NoResizeTransform
    
    t = NoResizeTransform()
    img = torch.rand(3, 1024, 1024)
    target = {"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]])}
    out_img, out_target = t.resize(img, target)
    assert out_img.shape == (3, 1024, 1024), "NoResizeTransform altered image dimensions!"
    assert torch.equal(out_target["boxes"], target["boxes"]), "NoResizeTransform altered target coordinates!"

def test_tiled_dataset():
    from bubble_rcnn.dataset.real_dataset import RealBubbleDataset, TiledDataset
    
    ds = RealBubbleDataset("data/crops/pulp", "data/labels/pulp")
    tiled = TiledDataset(ds, tile=512)
    assert len(tiled) >= len(ds)
    # Pull first tile
    img, target = tiled[0]
    assert img.shape[1] <= 512 and img.shape[2] <= 512
    assert "boxes" in target
    assert "masks" in target

