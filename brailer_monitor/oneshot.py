"""One-shot brailer detection from manual reference polygons (no training)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .labeling import sam_polygon

logger = logging.getLogger(__name__)

# HSV histogram bins for back-projection (H, S channels; V excluded for lighting robustness).
_HIST_BINS = (30, 32)
_HIST_RANGES = (0, 180, 0, 256)


@dataclass
class ReferenceSignature:
    """Visual signature derived from one or more reference brailer polygons."""

    hist_hs: np.ndarray
    hist_hs_backproj: np.ndarray
    hu_moments: np.ndarray
    area_ratio_min: float
    area_ratio_max: float
    aspect_min: float
    aspect_max: float
    center_cx: float
    center_cy: float
    center_tol: float = 0.25
    orb_descriptors: np.ndarray | None = None
    # Interior color: brailer fill is darker than background, especially at the bottom (tuna).
    interior_v_mean: float = 80.0
    interior_v_tol: float = 35.0
    dark_pixel_ratio: float = 0.5
    dark_pixel_ratio_min: float = 0.2
    bottom_top_v_delta: float = -15.0
    bottom_top_v_delta_max: float = 5.0
    # Teardrop / droplet shape: wider at bottom, narrower at top.
    width_ratio_bottom_top: float = 1.3
    width_ratio_min: float = 1.0
    width_ratio_max: float = 2.5
    widest_y_frac: float = 0.7

    def to_dict(self) -> dict[str, Any]:
        return {
            "area_ratio": [self.area_ratio_min, self.area_ratio_max],
            "aspect": [self.aspect_min, self.aspect_max],
            "center": [self.center_cx, self.center_cy],
            "center_tol": self.center_tol,
            "interior_v_mean": self.interior_v_mean,
            "dark_pixel_ratio": self.dark_pixel_ratio,
            "bottom_top_v_delta": self.bottom_top_v_delta,
            "width_ratio_bottom_top": self.width_ratio_bottom_top,
        }


@dataclass(frozen=True)
class DetectionResult:
    polygon_norm: list[tuple[float, float]]
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]


def _polygon_to_mask(
    polygon_norm: list[tuple[float, float]],
    width: int,
    height: int,
) -> np.ndarray:
    pts = np.array(
        [[int(x * width), int(y * height)] for x, y in polygon_norm],
        dtype=np.int32,
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _mask_metrics(mask: np.ndarray) -> tuple[float, float, float, float]:
    h, w = mask.shape[:2]
    frame_area = h * w
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0, 1.0, 0.5, 0.5
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
    area_ratio = cv2.countNonZero(mask) / frame_area
    aspect = bw / bh
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    return area_ratio, aspect, cx, cy


def _compute_hist_hs(frame: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    raw = cv2.calcHist([hsv], [0, 1], mask, _HIST_BINS, _HIST_RANGES)
    normed = raw.copy()
    cv2.normalize(normed, normed, 0, 1, cv2.NORM_MINMAX)
    return normed, raw


def _compute_hu_moments(mask: np.ndarray) -> np.ndarray:
    moments = cv2.moments(mask.astype(np.uint8))
    hu = cv2.HuMoments(moments).flatten()
    # Log-scale for stability.
    for i in range(len(hu)):
        if hu[i] != 0:
            hu[i] = -np.sign(hu[i]) * np.log10(abs(hu[i]))
    return hu


def _mask_band_width(mask: np.ndarray, frac_low: float, frac_high: float) -> float:
    """Horizontal span of mask pixels within a vertical band of the mask bbox."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0
    y1, y2 = int(ys.min()), int(ys.max())
    h = max(y2 - y1, 1)
    band_y1 = y1 + h * frac_low
    band_y2 = y1 + h * frac_high
    in_band = (ys >= band_y1) & (ys < band_y2)
    if not np.any(in_band):
        return 0.0
    band_xs = xs[in_band]
    return float(band_xs.max() - band_xs.min())


def _interior_color_profile(frame: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Brightness profile inside a mask (dark fill, darker toward bottom)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return {
            "mean_v": 128.0,
            "dark_ratio": 0.0,
            "bottom_top_delta": 0.0,
        }

    v_values = hsv[ys, xs, 2].astype(np.float32)
    y1, y2 = int(ys.min()), int(ys.max())
    h = max(y2 - y1, 1)
    top_cut = y1 + h * 0.33
    bot_cut = y1 + h * 0.66

    top_vals = v_values[ys <= top_cut]
    bot_vals = v_values[ys >= bot_cut]
    top_mean = float(top_vals.mean()) if len(top_vals) else float(v_values.mean())
    bot_mean = float(bot_vals.mean()) if len(bot_vals) else float(v_values.mean())

    dark_thresh = min(95.0, float(np.percentile(v_values, 55)))
    dark_ratio = float((v_values < dark_thresh).mean())

    return {
        "mean_v": float(v_values.mean()),
        "dark_ratio": dark_ratio,
        "bottom_top_delta": bot_mean - top_mean,
    }


def _teardrop_profile(mask: np.ndarray) -> dict[str, float]:
    """Droplet shape: wider bottom, narrow top; widest point in lower half."""
    width_top = _mask_band_width(mask, 0.0, 0.33)
    width_mid = _mask_band_width(mask, 0.33, 0.66)
    width_bottom = _mask_band_width(mask, 0.66, 1.0)
    width_ratio = width_bottom / max(width_top, 1.0)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return {"width_ratio_bottom_top": 1.0, "widest_y_frac": 0.5, "width_mid": 0.0}

    y1, y2 = int(ys.min()), int(ys.max())
    h = max(y2 - y1, 1)
    row_widths: list[tuple[float, float]] = []
    for y in range(y1, y2 + 1, max(1, h // 20)):
        row_xs = xs[ys == y]
        if len(row_xs) < 2:
            continue
        row_widths.append(((y - y1) / h, float(row_xs.max() - row_xs.min())))

    if row_widths:
        widest_y_frac = max(row_widths, key=lambda item: item[1])[0]
    else:
        widest_y_frac = 0.5

    return {
        "width_ratio_bottom_top": width_ratio,
        "width_mid": width_mid,
        "widest_y_frac": widest_y_frac,
    }


def _extract_orb(frame: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    orb = cv2.ORB_create(nfeatures=200)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, mask)
    if descriptors is None or len(descriptors) == 0:
        return None
    return descriptors


def build_reference(
    samples: list[tuple[np.ndarray, list[tuple[float, float]]]],
    *,
    center_tol: float = 0.25,
) -> ReferenceSignature:
    """Build a combined signature from reference (image, polygon_norm) pairs."""
    if not samples:
        raise ValueError("At least one reference sample is required")

    hists: list[np.ndarray] = []
    hists_backproj: list[np.ndarray] = []
    hu_list: list[np.ndarray] = []
    area_ratios: list[float] = []
    aspects: list[float] = []
    centers: list[tuple[float, float]] = []
    orb_descs: list[np.ndarray] = []
    interior_profiles: list[dict[str, float]] = []
    teardrop_profiles: list[dict[str, float]] = []

    for frame, polygon_norm in samples:
        h, w = frame.shape[:2]
        mask = _polygon_to_mask(polygon_norm, w, h)
        hist_norm, hist_raw = _compute_hist_hs(frame, mask)
        hists.append(hist_norm)
        hists_backproj.append(hist_raw)
        hu_list.append(_compute_hu_moments(mask))
        ar, asp, cx, cy = _mask_metrics(mask)
        area_ratios.append(ar)
        aspects.append(asp)
        centers.append((cx, cy))
        interior_profiles.append(_interior_color_profile(frame, mask))
        teardrop_profiles.append(_teardrop_profile(mask))
        desc = _extract_orb(frame, mask)
        if desc is not None:
            orb_descs.append(desc)

    avg_hist = np.mean(hists, axis=0)
    cv2.normalize(avg_hist, avg_hist, 0, 1, cv2.NORM_MINMAX)
    avg_hist_backproj = np.mean(hists_backproj, axis=0)
    avg_hu = np.mean(hu_list, axis=0)

    margin_area = max(0.002, np.mean(area_ratios) * 0.5)
    margin_aspect = max(0.2, np.mean(aspects) * 0.35)

    orb_combined = None
    if orb_descs:
        orb_combined = np.vstack(orb_descs)

    cx_mean = float(np.mean([c[0] for c in centers]))
    cy_mean = float(np.mean([c[1] for c in centers]))

    mean_v = float(np.mean([p["mean_v"] for p in interior_profiles]))
    dark_ratios = [p["dark_ratio"] for p in interior_profiles]
    bottom_deltas = [p["bottom_top_delta"] for p in interior_profiles]
    width_ratios = [p["width_ratio_bottom_top"] for p in teardrop_profiles]
    widest_fracs = [p["widest_y_frac"] for p in teardrop_profiles]

    return ReferenceSignature(
        hist_hs=avg_hist,
        hist_hs_backproj=avg_hist_backproj,
        hu_moments=avg_hu,
        area_ratio_min=max(0.001, min(area_ratios) - margin_area),
        area_ratio_max=min(0.15, max(area_ratios) + margin_area),
        aspect_min=max(0.2, min(aspects) - margin_aspect),
        aspect_max=min(3.5, max(aspects) + margin_aspect),
        center_cx=cx_mean,
        center_cy=cy_mean,
        center_tol=center_tol,
        orb_descriptors=orb_combined,
        interior_v_mean=mean_v,
        interior_v_tol=max(25.0, float(np.std([p["mean_v"] for p in interior_profiles])) + 20.0),
        dark_pixel_ratio=float(np.mean(dark_ratios)),
        dark_pixel_ratio_min=max(0.1, min(dark_ratios) - 0.15),
        bottom_top_v_delta=float(np.mean(bottom_deltas)),
        bottom_top_v_delta_max=min(10.0, max(bottom_deltas) + 10.0),
        width_ratio_bottom_top=float(np.mean(width_ratios)),
        width_ratio_min=max(0.9, min(width_ratios) - 0.35),
        width_ratio_max=min(3.5, max(width_ratios) + 0.5),
        widest_y_frac=float(np.mean(widest_fracs)),
    )


def _upper_center_roi_mask(height: int, width: int) -> np.ndarray:
    """Restrict search to upper-center region (same prior as frame_extractor)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    y1 = int(height * 0.48)
    x0 = int(width * 0.22)
    x1 = int(width * 0.78)
    mask[0:y1, x0:x1] = 255
    return mask


def _collect_region_candidates(
    mask: np.ndarray,
    signature: ReferenceSignature,
    *,
    max_candidates: int = 8,
) -> list[tuple[int, int, int, int]]:
    h, w = mask.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = h * w
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < frame_area * 0.0015 or area > frame_area * 0.12:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 25 or bh < 25:
            continue
        aspect = bw / max(bh, 1)
        if aspect < signature.aspect_min * 0.6 or aspect > signature.aspect_max * 1.4:
            continue
        cy = (y + bh / 2) / h
        if cy > 0.58:
            continue
        candidates.append((area, (x, y, x + bw, y + bh)))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [bbox for _, bbox in candidates[:max_candidates]]


def _dark_blob_mask(frame: np.ndarray) -> np.ndarray:
    """Fallback mask similar to frame_extractor dark-blob heuristic."""
    h, w = frame.shape[:2]
    y1 = int(h * 0.48)
    x0 = int(w * 0.22)
    x1 = int(w * 0.78)
    roi = frame[0:y1, x0:x1]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 105))
    mask = cv2.bitwise_and(mask, cv2.inRange(gray, 0, 115))

    full = np.zeros((h, w), dtype=np.uint8)
    full[0:y1, x0:x1] = mask
    return full


def propose_regions(
    frame: np.ndarray,
    signature: ReferenceSignature,
    *,
    max_candidates: int = 8,
) -> list[tuple[int, int, int, int]]:
    """Propose candidate bounding boxes via color back-projection."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    backproj = cv2.calcBackProject(
        [hsv], [0, 1], signature.hist_hs_backproj, _HIST_RANGES, scale=1.0
    )

    roi_mask = _upper_center_roi_mask(h, w)
    backproj = cv2.bitwise_and(backproj, roi_mask)

    if cv2.countNonZero(backproj) > 0:
        _, thresh = cv2.threshold(backproj, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        thresh = np.zeros((h, w), dtype=np.uint8)

    candidates = _collect_region_candidates(thresh, signature, max_candidates=max_candidates)
    if candidates:
        return candidates

    fallback = _dark_blob_mask(frame)
    return _collect_region_candidates(fallback, signature, max_candidates=max_candidates)


def _score_color(frame: np.ndarray, mask: np.ndarray, signature: ReferenceSignature) -> float:
    hist_norm, _ = _compute_hist_hs(frame, mask)
    corr = cv2.compareHist(signature.hist_hs, hist_norm, cv2.HISTCMP_CORREL)
    return float(np.clip((corr + 1) / 2, 0, 1))


def _score_interior_color(
    frame: np.ndarray,
    mask: np.ndarray,
    signature: ReferenceSignature,
) -> float:
    """Score dark interior fill and darker-bottom gradient (tuna vs background)."""
    profile = _interior_color_profile(frame, mask)
    scores: list[float] = []

    v_dist = abs(profile["mean_v"] - signature.interior_v_mean)
    scores.append(float(np.clip(1.0 - v_dist / signature.interior_v_tol, 0, 1)))

    if profile["dark_ratio"] >= signature.dark_pixel_ratio_min:
        scores.append(
            float(
                np.clip(
                    1.0 - abs(profile["dark_ratio"] - signature.dark_pixel_ratio) / 0.35,
                    0,
                    1,
                )
            )
        )
    else:
        scores.append(float(np.clip(profile["dark_ratio"] / max(signature.dark_pixel_ratio_min, 0.01), 0, 0.5)))

    # Bottom should be darker than top (negative delta).
    if profile["bottom_top_delta"] <= signature.bottom_top_v_delta_max:
        delta_score = 1.0 - abs(profile["bottom_top_delta"] - signature.bottom_top_v_delta) / 30.0
        scores.append(float(np.clip(delta_score, 0, 1)))
    else:
        scores.append(0.2)

    return float(np.mean(scores))


def _score_teardrop(mask: np.ndarray, signature: ReferenceSignature) -> float:
    """Score droplet shape: wider bottom, widest point in lower region."""
    profile = _teardrop_profile(mask)
    scores: list[float] = []

    ratio = profile["width_ratio_bottom_top"]
    if signature.width_ratio_min <= ratio <= signature.width_ratio_max:
        scores.append(
            float(
                np.clip(
                    1.0 - abs(ratio - signature.width_ratio_bottom_top) / 0.8,
                    0,
                    1,
                )
            )
        )
    else:
        dist = min(abs(ratio - signature.width_ratio_min), abs(ratio - signature.width_ratio_max))
        scores.append(float(np.clip(1.0 - dist / 0.8, 0, 0.4)))

    # Widest horizontal span should sit in lower half (typical hanging brailer).
    widest_y = profile["widest_y_frac"]
    ref_widest = signature.widest_y_frac
    if widest_y >= 0.45:
        scores.append(float(np.clip(1.0 - abs(widest_y - ref_widest) / 0.35, 0, 1)))
    else:
        scores.append(0.2)

    # Mid band should be between top and bottom width (smooth teardrop taper).
    width_top = _mask_band_width(mask, 0.0, 0.33)
    width_bottom = _mask_band_width(mask, 0.66, 1.0)
    width_mid = profile["width_mid"]
    if width_top < width_mid < width_bottom or width_top < width_bottom:
        taper = (width_mid - width_top) / max(width_bottom - width_top, 1.0)
        scores.append(float(np.clip(taper, 0.3, 1.0)))
    else:
        scores.append(0.3)

    return float(np.mean(scores))


def _score_shape(mask: np.ndarray, signature: ReferenceSignature) -> float:
    moments = cv2.moments(mask.astype(np.uint8))
    if moments["m00"] == 0:
        return 0.0
    hu = cv2.HuMoments(moments).flatten()
    for i in range(len(hu)):
        if hu[i] != 0:
            hu[i] = -np.sign(hu[i]) * np.log10(abs(hu[i]))
    diff = np.abs(hu - signature.hu_moments)
    # Lower diff -> higher score.
    return float(np.clip(1.0 - np.mean(diff) / 2.0, 0, 1))


def _score_geometry(mask: np.ndarray, signature: ReferenceSignature) -> float:
    area_ratio, aspect, cx, cy = _mask_metrics(mask)
    scores: list[float] = []

    if signature.area_ratio_min <= area_ratio <= signature.area_ratio_max:
        scores.append(1.0)
    else:
        dist = min(
            abs(area_ratio - signature.area_ratio_min),
            abs(area_ratio - signature.area_ratio_max),
        )
        scores.append(float(np.clip(1.0 - dist / 0.05, 0, 1)))

    if signature.aspect_min <= aspect <= signature.aspect_max:
        scores.append(1.0)
    else:
        dist = min(abs(aspect - signature.aspect_min), abs(aspect - signature.aspect_max))
        scores.append(float(np.clip(1.0 - dist / 0.8, 0, 1)))

    center_dist = ((cx - signature.center_cx) ** 2 + (cy - signature.center_cy) ** 2) ** 0.5
    scores.append(float(np.clip(1.0 - center_dist / signature.center_tol, 0, 1)))

    return float(np.mean(scores))


def _score_orb(
    frame: np.ndarray,
    mask: np.ndarray,
    signature: ReferenceSignature,
) -> float:
    if signature.orb_descriptors is None:
        return 0.5
    desc = _extract_orb(frame, mask)
    if desc is None:
        return 0.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(signature.orb_descriptors, desc)
    if not matches:
        return 0.0
    good = [m for m in matches if m.distance < 50]
    ratio = len(good) / max(len(signature.orb_descriptors), 1)
    return float(np.clip(ratio * 3.0, 0, 1))


def score_region(
    frame: np.ndarray,
    mask: np.ndarray,
    signature: ReferenceSignature,
) -> float:
    """Score a candidate mask against the reference signature (0..1)."""
    color = _score_color(frame, mask, signature)
    interior = _score_interior_color(frame, mask, signature)
    teardrop = _score_teardrop(mask, signature)
    shape = _score_shape(mask, signature)
    geom = _score_geometry(mask, signature)
    orb = _score_orb(frame, mask, signature)
    return float(
        0.20 * color
        + 0.25 * interior
        + 0.20 * teardrop
        + 0.15 * shape
        + 0.12 * geom
        + 0.08 * orb
    )


def _mask_from_bbox(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    mask[y1:y2, x1:x2] = 255
    return mask


def detect_oneshot(
    frame: np.ndarray,
    signature: ReferenceSignature,
    sam_model: Any | None = None,
    *,
    threshold: float = 0.55,
) -> DetectionResult | None:
    """Detect brailer in a frame using reference signature; optional SAM refinement."""
    candidates = propose_regions(frame, signature)
    if not candidates:
        return None

    h, w = frame.shape[:2]
    best_score = 0.0
    best_polygon: list[tuple[float, float]] | None = None
    best_bbox: tuple[int, int, int, int] | None = None

    for bbox in candidates:
        polygon: list[tuple[float, float]] | None = None
        if sam_model is not None:
            try:
                polygon = sam_polygon(frame, bbox, sam_model)
            except Exception as exc:
                logger.debug("SAM polygon failed for bbox %s: %s", bbox, exc)
                polygon = None
        if polygon is None:
            # Fallback: use bbox as rectangle polygon.
            x1, y1, x2, y2 = bbox
            polygon = [
                (x1 / w, y1 / h),
                (x2 / w, y1 / h),
                (x2 / w, y2 / h),
                (x1 / w, y2 / h),
            ]

        mask = _polygon_to_mask(polygon, w, h)
        score = score_region(frame, mask, signature)
        if score > best_score:
            best_score = score
            best_polygon = polygon
            best_bbox = bbox

    if best_polygon is None or best_score < threshold:
        return None

    assert best_bbox is not None
    return DetectionResult(
        polygon_norm=best_polygon,
        confidence=best_score,
        bbox_xyxy=best_bbox,
    )


def load_sam_model(model_path: str) -> Any:
    """Load ultralytics SAM model from local path."""
    from ultralytics import SAM

    return SAM(model_path)
