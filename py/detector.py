from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

@dataclass
class DetectorParams:
    red_h_low_max: int = 10
    red_h_high_min: int = 170
    red_s_min: int = 90
    red_v_min: int = 50
    blue_h_min: int = 95
    blue_h_max: int = 135
    blue_s_min: int = 90
    blue_v_min: int = 50
    dominant_ratio: float = 0.8
    interleave_ratio: float = 0.18
    layer_split_ratio: float = 0.5
    layer_size_tol: float = 0.45
    blur_kernel: int = 5
    morph_kernel: int = 5
    min_area_px: int = 200
    real_edge_m: float = 0.05
    corner_offset_m: float = 0.0
    fx: float = 600.0
    fy: float = 600.0
    cx: float = 320.0
    cy: float = 240.0
    distance_mode: str = 'pixel_height'
    cam_height_m: float = 0.35
    cam_pitch_deg: float = 30.0

    def to_dict(self) -> Dict[str, object]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

@dataclass
class Detection:
    block_class: str = 'unknown'
    edge_length_m: float = 0.0
    edge_length_px: float = 0.0
    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.0
    u_px: float = 0.0
    v_px: float = 0.0
    confidence: float = 0.0
    box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    ratio_red: float = 0.0
    ratio_blue: float = 0.0
    layer_ok: bool = False
    debug: Dict[str, float] = field(default_factory=dict)

def _odd(k: int) -> int:
    k = int(k)
    if k < 1:
        k = 1
    return k if k % 2 == 1 else k + 1

def build_masks(hsv: np.ndarray, p: DetectorParams) -> Tuple[np.ndarray, np.ndarray]:
    hsv_b = cv2.GaussianBlur(hsv, (_odd(p.blur_kernel),) * 2, 0)
    lower_red_1 = np.array([0, p.red_s_min, p.red_v_min], dtype=np.uint8)
    upper_red_1 = np.array([p.red_h_low_max, 255, 255], dtype=np.uint8)
    lower_red_2 = np.array([p.red_h_high_min, p.red_s_min, p.red_v_min], dtype=np.uint8)
    upper_red_2 = np.array([179, 255, 255], dtype=np.uint8)
    mask_red = cv2.bitwise_or(cv2.inRange(hsv_b, lower_red_1, upper_red_1), cv2.inRange(hsv_b, lower_red_2, upper_red_2))
    lower_blue = np.array([p.blue_h_min, p.blue_s_min, p.blue_v_min], dtype=np.uint8)
    upper_blue = np.array([p.blue_h_max, 255, 255], dtype=np.uint8)
    mask_blue = cv2.inRange(hsv_b, lower_blue, upper_blue)
    mk = _odd(p.morph_kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
    for m in (mask_red, mask_blue):
        cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, dst=m)
        cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, dst=m)
    return (mask_red, mask_blue)

def pixel_edge_for_distance(rect: Tuple[Tuple[float, float], Tuple[float, float], float]) -> float:
    (_c, (w, h), _a) = rect
    return float(max(w, h))

def classify(n_red: float, n_blue: float, p: DetectorParams, mask_red_blob: np.ndarray, mask_blue_blob: np.ndarray, box: Tuple[int, int, int, int]) -> Tuple[str, float, float, float, bool]:
    total = n_red + n_blue
    if total <= 0:
        return ('unknown', 0.0, 0.0, 0.0, False)
    r_red = n_red / total
    r_blue = n_blue / total
    if r_red >= p.dominant_ratio and r_blue < p.interleave_ratio:
        return ('red', r_red, r_red, r_blue, False)
    if r_blue >= p.dominant_ratio and r_red < p.interleave_ratio:
        return ('blue', r_blue, r_red, r_blue, False)
    if r_red >= p.interleave_ratio and r_blue >= p.interleave_ratio:
        conf = min(r_red, r_blue) * 2.0
        ok = layer_check(mask_red_blob, mask_blue_blob, box, p)
        if ok:
            return ('red_blue', conf, r_red, r_blue, True)
        if r_red >= r_blue:
            return ('red', r_red, r_red, r_blue, False)
        return ('blue', r_blue, r_red, r_blue, False)
    return ('unknown', 0.0, r_red, r_blue, False)

def layer_check(mask_red: np.ndarray, mask_blue: np.ndarray, box: Tuple[int, int, int, int], p: DetectorParams) -> bool:
    (x, y, w, h) = box
    if h < 8 or w < 8:
        return False
    y0 = max(0, y)
    y1 = min(mask_red.shape[0], y + h)
    x0 = max(0, x)
    x1 = min(mask_red.shape[1], x + w)
    split = y0 + int((y1 - y0) * p.layer_split_ratio)
    if split <= y0 or split >= y1:
        return False
    up_r = int(np.count_nonzero(mask_red[y0:split, x0:x1]))
    up_b = int(np.count_nonzero(mask_blue[y0:split, x0:x1]))
    lo_r = int(np.count_nonzero(mask_red[split:y1, x0:x1]))
    lo_b = int(np.count_nonzero(mask_blue[split:y1, x0:x1]))
    red_top = up_r > up_b and lo_b > lo_r
    blue_top = up_b > up_r and lo_r > lo_b
    layered = red_top or blue_top
    dom_up = up_r if red_top else up_b
    dom_low = lo_b if red_top else lo_r
    if dom_up + dom_low <= 0:
        return False
    balance = min(dom_up, dom_low) / max(dom_up, dom_low)
    return bool(layered and balance >= p.layer_size_tol)

def detect(bgr: np.ndarray, p: DetectorParams) -> Detection:
    out = Detection()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    (mask_red, mask_blue) = build_masks(hsv, p)
    mask_any = cv2.bitwise_or(mask_red, mask_blue)
    _gap = max(7, _odd(p.morph_kernel) * 2 + 1)
    mask_any = cv2.morphologyEx(mask_any, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_gap, _gap)))
    (contours, _) = cv2.findContours(mask_any, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return out
    areas = [cv2.contourArea(c) for c in contours]
    best = int(np.argmax(areas))
    if areas[best] < p.min_area_px:
        return out
    cnt = contours[best]
    box = cv2.boundingRect(cnt)
    only = np.zeros(mask_any.shape, dtype=np.uint8)
    cv2.drawContours(only, [cnt], -1, 255, cv2.FILLED)
    blob_red = cv2.bitwise_and(mask_red, only)
    blob_blue = cv2.bitwise_and(mask_blue, only)
    n_red = float(np.count_nonzero(blob_red))
    n_blue = float(np.count_nonzero(blob_blue))
    if n_red + n_blue < 1:
        return out
    (cls, conf, r_red, r_blue, layer_ok) = classify(n_red, n_blue, p, blob_red, blob_blue, box)
    rect = cv2.minAreaRect(cnt)
    px_edge = pixel_edge_for_distance(rect)
    z = x = y = 0.0
    if px_edge > 1.0:
        if p.distance_mode == 'ground_plane':
            z = ground_plane_distance(box, p)
        else:
            z = p.fy * p.real_edge_m / px_edge
        uc = box[0] + box[2] * 0.5
        vc = box[1] + box[3] * 0.5
        x = (uc - p.cx) * z / p.fx
        y = (vc - p.cy) * z / p.fy
    if abs(p.corner_offset_m) > 1e-09:
        x -= p.corner_offset_m
    out.block_class = cls
    out.edge_length_m = p.real_edge_m if px_edge > 1.0 else 0.0
    out.edge_length_px = px_edge
    (out.x_m, out.y_m, out.z_m) = (x, y, z)
    out.u_px = box[0] + box[2] * 0.5
    out.v_px = box[1] + box[3] * 0.5
    out.confidence = conf
    out.box = tuple((int(v) for v in box))
    (out.ratio_red, out.ratio_blue) = (r_red, r_blue)
    out.layer_ok = layer_ok
    out.debug = {'n_red': n_red, 'n_blue': n_blue, 'area': areas[best], 'rect_w': rect[1][0], 'rect_h': rect[1][1]}
    return out

def ground_plane_distance(box: Tuple[int, int, int, int], p: DetectorParams) -> float:
    v_bottom = box[1] + box[3]
    alpha = math.atan((v_bottom - p.cy) / max(1.0, p.fy))
    theta = math.radians(p.cam_pitch_deg)
    z = p.cam_height_m * math.tan(theta + alpha)
    return z if math.isfinite(z) and z > 0 else 0.0

def draw_result(bgr: np.ndarray, det: Detection, params: DetectorParams) -> np.ndarray:
    canvas = bgr.copy()
    (h, w) = canvas.shape[:2]
    cv2.drawMarker(canvas, (w // 2, h // 2), (0, 0, 255), cv2.MARKER_CROSS, 26, 2)
    (x, y, bw, bh) = det.box
    if bw > 0 and bh > 0:
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        lines = [f'{det.block_class}  conf={det.confidence:.2f}', f'z={det.z_m:.4f}m  x={det.x_m:+.4f}m  y={det.y_m:+.4f}m', f'edge={det.edge_length_m:.4f}m ({det.edge_length_px:.1f}px)', f'R={det.ratio_red:.2f} B={det.ratio_blue:.2f} layer={det.layer_ok}']
        ty = max(18, y - 8 - 16 * (len(lines) - 1))
        for (i, s) in enumerate(lines):
            org = (x, ty + 16 * i)
            cv2.putText(canvas, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(canvas, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        cv2.putText(canvas, 'no target', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return canvas
DEFAULT_PARAMS = DetectorParams()
