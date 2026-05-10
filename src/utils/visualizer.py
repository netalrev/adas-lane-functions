import cv2
import numpy as np

# Color map per Waymo object type (BGR)
_TYPE_COLORS = {
    1: (255, 100, 50),   # Vehicle  — blue
    2: (50, 255, 100),   # Pedestrian — green
    3: (255, 255, 255),  # Sign — white
    4: (50, 220, 255),   # Cyclist — yellow
}
_DEFAULT_COLOR = (255, 255, 255)

_TYPE_NAMES = {
    1: "Vehicle",
    2: "Pedestrian",
    3: "Sign",
    4: "Cyclist",
}

# HUD layout constants
_HUD_MARGIN = 10
_HUD_LINE_HEIGHT = 22
_HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
_HUD_FONT_SCALE = 0.6
_HUD_THICKNESS = 1
_HUD_ALPHA = 0.55  # transparency of HUD background


def draw_annotations(img: np.ndarray, gt_data: dict, step: int) -> np.ndarray:
    """
    Draws 2D bounding boxes and a HUD overlay onto a copy of the input image.

    Args:
        img:     BGR numpy array (H, W, 3), uint8. Never mutated.
        gt_data: Ground truth dict with keys 'boxes_2d', 'ego_speed_kmh'.
        step:    Current frame index.

    Returns:
        Annotated BGR numpy array.
    """
    canvas = img.copy()

    boxes = gt_data.get("boxes_2d", [])
    ego_speed = gt_data.get("ego_speed_kmh", 0.0)

    # --- Draw bounding boxes ---
    for box in boxes:
        cx = box["center_x"]
        cy = box["center_y"]
        w = box["length"]   # pixel width  (Waymo 'length' = horizontal)
        h = box["width"]    # pixel height (Waymo 'width'  = vertical)

        x1 = int(cx - w / 2.0)
        y1 = int(cy - h / 2.0)
        x2 = int(cx + w / 2.0)
        y2 = int(cy + h / 2.0)

        obj_type = box.get("type", 0)
        color = _TYPE_COLORS.get(obj_type, _DEFAULT_COLOR)
        class_name = _TYPE_NAMES.get(obj_type, "Unknown")
        short_id = box["id"][:4]
        label = f"{class_name} {short_id}"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas, label,
            (x1, max(y1 - 5, 10)),
            _HUD_FONT, 0.5, color, _HUD_THICKNESS, cv2.LINE_AA
        )

    # --- Draw semi-transparent HUD ---
    hud_lines = [
        f"Frame:   {step}",
        f"Speed:   {ego_speed:.1f} km/h",
        f"Objects: {len(boxes)}",
    ]

    hud_w = 210
    hud_h = _HUD_MARGIN * 2 + len(hud_lines) * _HUD_LINE_HEIGHT
    x0, y0 = _HUD_MARGIN, _HUD_MARGIN

    # Semi-transparent dark background via addWeighted
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + hud_w, y0 + hud_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, _HUD_ALPHA, canvas, 1 - _HUD_ALPHA, 0, canvas)

    for i, line in enumerate(hud_lines):
        text_y = y0 + _HUD_MARGIN + (i + 1) * _HUD_LINE_HEIGHT - 4
        cv2.putText(
            canvas, line,
            (x0 + 8, text_y),
            _HUD_FONT, _HUD_FONT_SCALE, (255, 255, 255), _HUD_THICKNESS, cv2.LINE_AA
        )

    return canvas
