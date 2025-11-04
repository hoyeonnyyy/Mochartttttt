"""
chart_tools.py — Chart image tooling (OCR-first), with optional SAM via local ckpt or Hugging Face.

What's inside
-------------
Core tools (via run_tool):
  - detect_legend_text(image, max_items=20)
  - annotate_legend(image, legend)              # legend 없으면 자동으로 annotate_legend_auto로 우회
  - annotate_legend_auto(image, max_items=20)   # OCR로 legend 추출 후 라벨링
  - clean_chart_image(image, title=None, legend=None, pad_expand_px=6)
  - segment_and_mark(image, segmentation_model="SAM", ...)
        * SAM 사용 시: sam_checkpoint or (sam_hf_repo + sam_hf_filename) 로 해결
        * 그 외엔 adaptive threshold fallback으로 CPU만으로 동작
  - axis_localizer(image, axis="x", axis_threshold=0.2, axis_tickers=None)
  - interpolate_pixel_to_value(pixel, axis_values, axis_pixel_positions)
  - arithmetic(a, b, operation="percentage")
  - get_marker_rgb(legend_image, bbox_mapping, text_of_interest | label_of_interest)

Dependencies
------------
Required: pillow, opencv-python, numpy
OCR: install ONE of [pytesseract (+OS tesseract), easyocr]
SAM (optional): torch, segment-anything
Hugging Face auto-download (optional): huggingface_hub

Notes on SAM
------------
- segment-anything 라이브러리는 보통 .pth 체크포인트 경로가 필요합니다.
- 본 모듈은 sam_hf_repo / sam_hf_filename 인자를 주면 huggingface_hub로 자동 다운로드 후 사용합니다.
  (예) sam_hf_repo="facebook/sam-vit-b", sam_hf_filename="sam_vit_b_01ec64.pth"
- GPU가 없으면 device="cpu"로도 가능하나 느릴 수 있습니다. (fallback은 CPU로 충분히 빠름)
"""

from __future__ import annotations
from dataclasses import dataclass
from tkinter import TRUE
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import importlib
import os
import torch

# ------------------------------
# Params & Small Utils
# ------------------------------

def _legend_candidate_slices(cv, paddings=(0.3, 0.2, 0.25)):
    """하단/상단/우측 후보 영역 잘라서 [(y0,x0,y1,x1), panel] 리스트 반환"""
    H, W = cv.shape[:2]
    out = []
    # bottom
    r = float(paddings[0])
    y0 = int(H*(1.0-r)); out.append(((y0,0,H,W), cv[y0:H, 0:W]))
    # top
    r = float(paddings[1])
    y1 = int(H*r); out.append(((0,0,y1,W), cv[0:y1, 0:W]))
    # right
    r = float(paddings[2])
    x0 = int(W*(1.0-r)); out.append(((0,x0,H,W), cv[0:H, x0:W]))
    return out

def detect_legend_text_robust(image: Image.Image, max_items: int = 20, ocr: str = "auto"):
    """
    색-마커/텍스트 양쪽 탐지로 범례를 더 강하게 찾는다.
    반환: (legend_list, legend_panel_pil or None)
    """
    cv = pil_to_cv(image)
    H, W = cv.shape[:2]

    best = ([], None)

    for (y0,x0,y1,x1), panel in _legend_candidate_slices(cv):
        ph, pw = panel.shape[:2]
        if ph < 30 or pw < 80:
            continue

        # --- 업스케일로 작은 텍스트 보강 ---
        panel_pil = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        up = panel_pil.resize((panel_pil.width*2, panel_pil.height*2), Image.BICUBIC)

        # ---------- (A) 마커 우선 ----------
        legend_A = []
        try:
            gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 3)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,
                                       dp=1.2, minDist=28,
                                       param1=100, param2=15,
                                       minRadius=6, maxRadius=20)
        except Exception:
            circles = None

        regs = _ocr_text_regions(up, backend=ocr)
        regs_scaled = []
        for (bx1,by1,bx2,by2), txt in regs:
            regs_scaled.append(((bx1//2,by1//2,bx2//2,by2//2), txt.strip()))

        if circles is not None:
            C = np.uint16(np.around(circles[0]))  # (x,y,r)
            used = set()
            for (cx, cy, r) in C:
                best_i, best_d = -1, 1e18
                for i, (bb, txt) in enumerate(regs_scaled):
                    if not txt or i in used: continue
                    (ax1,ay1,ax2,ay2) = bb
                    dx = (ax1+ax2)/2 - cx; dy = (ay1+ay2)/2 - cy
                    d = dx*dx + dy*dy
                    if d < best_d:
                        best_d, best_i = d, i
                if best_i >= 0:
                    used.add(best_i)
                    legend_A.append(regs_scaled[best_i][1])

        # ---------- (B) 텍스트 우선 ----------
        legend_B = []
        if regs_scaled:
            # 짧은 토큰 위주로 필터
            cand = []
            for (bb, txt) in regs_scaled:
                if len(txt) <= 8 or txt.isdigit() or (len(txt.split()) <= 2):
                    cand.append((bb, txt))
            # y기반 행 그룹핑
            cand.sort(key=lambda t: ( (t[0][1]+t[0][3])//2 ))
            rows = []
            for (bb, txt) in cand:
                cy = (bb[1]+bb[3])//2
                if not rows or abs(cy - rows[-1][0]) > 18:
                    rows.append([cy, [(bb,txt)]])
                else:
                    rows[-1][1].append((bb,txt))
            # 각 행에 좌/우 작은 컬러 패치 있는지 확인
            panel_hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
            for _, items in rows:
                # 행 전체 bbox
                xs = [b[0][0] for b in items] + [b[0][2] for b in items]
                ys = [b[0][1] for b in items] + [b[0][3] for b in items]
                xL, xR = max(min(xs)-10,0), min(max(xs)+10, panel.shape[1]-1)
                yT, yB = max(min(ys)-10,0), min(max(ys)+10, panel.shape[0]-1)
                strip = panel_hsv[yT:yB, max(0,xL-30):min(panel.shape[1]-1,xR+30)]
                if strip.size == 0: 
                    continue
                # "유색 픽셀" 존재하면 범례 행으로 인정
                sat = strip[...,1]; val = strip[...,2]
                colored_ratio = np.mean((sat>40) & (val>80))
                if colored_ratio > 0.03:
                    # 행의 문자열을 공백으로 합침 (연속 토큰을 하나로)
                    row_text = " ".join(t for (_bb,t) in items if t)
                    legend_B.append(row_text.strip())

        # ---------- A/B 중 더 좋은 쪽 선택 ----------
        candidate = legend_A if len(legend_A) >= len(legend_B) else legend_B
        candidate = [t for t in candidate if t][:max_items]
        if len(candidate) > len(best[0]):
            best = (candidate, panel_pil)

    # 최소 2개 미만이면 범례 없음
    if len(best[0]) < 2:
        return [], None
    return best

@dataclass
class LegendParams:
    max_legend_area_ratio: float = 0.35
    min_legend_area_ratio: float = 0.02
    pref_right_bias: float = 0.6
    canny1: int = 80
    canny2: int = 160
    row_min_height: int = 8
    row_max_height: int = 120
    row_merge_gap: int = 4
    marker_min_w: int = 6
    marker_max_w: int = 60
    marker_min_h: int = 6
    marker_max_h: int = 60
    marker_sat_thresh: int = 40
    marker_gray_tol: int = 18
    annotate_font_size: int = 16

PARAMS = LegendParams()

def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def _bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    if inter == 0: return 0.0
    area_a = (ax2-ax1)*(ay2-ay1)
    area_b = (bx2-bx1)*(by2-by1)
    return inter / float(area_a + area_b - inter + 1e-8)


_EASYOCR_READER = None
_PADDLE_OCR = None

def _get_easyocr_reader():
    """lazy singleton for easyocr"""
    global _EASYOCR_READER
    if _EASYOCR_READER is not None:
        return _EASYOCR_READER
    try:
        import torch, easyocr
        use_gpu = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        _EASYOCR_READER = easyocr.Reader(['en'], gpu=use_gpu)
    except Exception:
        _EASYOCR_READER = None
    return _EASYOCR_READER

def _get_paddle_ocr():
    """lazy singleton for PaddleOCR (PP-OCRv4/5 계열)"""
    global _PADDLE_OCR
    if _PADDLE_OCR is not None:
        return _PADDLE_OCR
    try:
        import torch
        from paddleocr import PaddleOCR
        use_gpu = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        # lang='en'로 영어 차트에 최적화, cls=False로 속도↑
        _PADDLE_OCR = PaddleOCR(
            lang='en', det=True, rec=True, use_angle_cls=False,
            use_gpu=use_gpu, show_log=False
        )
    except Exception:
        _PADDLE_OCR = None
    return _PADDLE_OCR

def _ocr_text_regions(pil_img, backend: str = "auto"):
    """
    차트용 OCR 추상화:
      backend in {"auto","paddle","tesseract","easyocr"}
    반환: [((x1,y1,x2,y2), text), ...]  (PIL 좌표계, int)
    """
    # 1) backend 우선순위 구성
    if backend == "auto":
        order = ["paddle", "tesseract", "easyocr"]
    else:
        order = [backend]

    img_np = np.array(pil_img)

    for b in order:
        try:
            if b == "paddle":
                ocr = _get_paddle_ocr()
                if ocr is None:
                    raise RuntimeError("PaddleOCR not available")
                res = ocr.ocr(img_np, cls=False)  # [ [ (poly, (text, conf)), ... ] ]
                regs = []
                if res and len(res) > 0 and res[0]:
                    for poly, (txt, conf) in res[0]:
                        if not txt: 
                            continue
                        xs = [int(p[0]) for p in poly]
                        ys = [int(p[1]) for p in poly]
                        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                        regs.append(((x1, y1, x2, y2), txt.strip()))
                if regs:
                    return regs

            elif b == "tesseract":
                import pytesseract
                d = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
                regs = []
                for i in range(len(d["text"])):
                    txt = (d["text"][i] or "").strip()
                    if not txt:
                        continue
                    x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
                    regs.append(((int(x), int(y), int(x+w), int(y+h)), txt))
                if regs:
                    return regs

            elif b == "easyocr":
                reader = _get_easyocr_reader()
                if reader is None:
                    raise RuntimeError("EasyOCR not available")
                result = reader.readtext(img_np, detail=1)  # [(bbox, text, conf),...]
                regs = []
                for (bbox, txt, conf) in result:
                    if not txt:
                        continue
                    (x1,y1),(x2,y2),(x3,y3),(x4,y4) = bbox
                    xs = [x1,x2,x3,x4]; ys = [y1,y2,y3,y4]
                    regs.append(((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))), txt.strip()))
                if regs:
                    return regs
        except Exception:
            # 다음 백엔드로 폴백
            continue

    return []
# ------------------------------
# Legend panel detection & annotate
# ------------------------------

def _find_legend_panel(cv_img: np.ndarray, params: LegendParams) -> Tuple[Tuple[int,int,int,int], np.ndarray]:
    H, W = cv_img.shape[:2]
    area = H * W
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, params.canny1, params.canny2)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_score = -1.0
    best_box = (0,0,W,H)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        a = w*h
        if a < area*params.min_legend_area_ratio or a > area*params.max_legend_area_ratio:
            continue
        ar = w / max(h,1)
        if not (0.3 <= ar <= 6.5):
            continue
        rightness = x / W
        topness = 1.0 - (y / H)
        loc_score = params.pref_right_bias * (0.6*rightness + 0.4*topness)
        roi = edges[y:y+h, x:x+w]
        density = roi.mean() / 255.0
        score = density + loc_score
        if score > best_score:
            best_score = score
            best_box = (x,y,x+w,y+h)
    x1,y1,x2,y2 = best_box
    return best_box, cv_img[y1:y2, x1:x2].copy()

def _split_rows_by_projection(panel: np.ndarray, params: LegendParams) -> List[Tuple[int,int]]:
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    edges = cv2.Canny(norm, 60, 120)
    proj = edges.sum(axis=1)
    th = max(8, int(0.1 * (proj.max() if proj.max() > 0 else 1)))
    on = (proj > th).astype(np.uint8)

    rows = []
    s = None
    for i, v in enumerate(on):
        if v and s is None:
            s = i
        elif not v and s is not None:
            if (i - s) >= params.row_min_height and (i - s) <= params.row_max_height:
                rows.append((s, i))
            s = None
    if s is not None and (len(on)-s) >= params.row_min_height:
        rows.append((s, len(on)-1))

    merged = []
    for r in rows:
        if not merged or r[0] - merged[-1][1] > params.row_merge_gap:
            merged.append(list(r))
        else:
            merged[-1][1] = r[1]
    return [(a,b) for a,b in merged]

def _detect_marker_in_row(panel: np.ndarray, row: Tuple[int,int], params: LegendParams) -> Optional[Tuple[int,int,int,int]]:
    r1, r2 = row
    band = panel[r1:r2, :, :]
    bgr = band.copy()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat_mask = (hsv[:,:,1] > params.marker_sat_thresh).astype(np.uint8)
    b,g,r = cv2.split(bgr)
    grayish = (np.abs(r-g) + np.abs(g-b) < params.marker_gray_tol).astype(np.uint8)
    color_mask = cv2.bitwise_and(sat_mask, 1 - grayish)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), iterations=1)
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = -1
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < params.marker_min_w or w > params.marker_max_w: 
            continue
        if h < params.marker_min_h or h > params.marker_max_h:
            continue
        a = w*h
        if a > best_area:
            best_area = a
            best = (x, y, x+w, y+h)

    if best is None:
        return None
    x1,y1,x2,y2 = best
    return (x1, y1 + r1, x2, y2 + r1)

def annotate_legend(
    image: Image.Image,
    legend: Optional[List[str]] = None,
    params: LegendParams = PARAMS,
) -> Tuple[Image.Image, Image.Image, Dict[int, Tuple[str, Tuple[int,int,int,int]]]]:
    """
    legend 리스트가 없으면(=None 또는 빈 리스트)만 OCR 자동 경로로 우회.
    legend가 주어지면 '절대' annotate_legend_auto를 다시 호출하지 않고,
    여기서 직접 라벨 렌더링을 완료한다.
    """
    # --------- 빈/누락 범례면 자동 우회 ---------
    if legend is None or (isinstance(legend, (list, tuple)) and len(legend) == 0):
        return annotate_legend_auto(image)

    # --------- 여기부터는 legend가 '있는' 정상 렌더링 경로 ---------
    cv_img = pil_to_cv(image)
    (x1, y1, x2, y2), panel = _find_legend_panel(cv_img, params)
    legend_panel = cv_to_pil(panel)

    rows = _split_rows_by_projection(panel, params)
    draw = ImageDraw.Draw(legend_panel)
    try:
        font = ImageFont.truetype("arial.ttf", params.annotate_font_size)
    except Exception:
        font = ImageFont.load_default()

    bbox_mapping: Dict[int, Tuple[str, Tuple[int,int,int,int]]] = {}
    label_id = 1
    for i, row in enumerate(rows):
        if label_id > len(legend):
            break
        box = _detect_marker_in_row(panel, row, params)
        if box is None:
            continue
        bx1, by1, bx2, by2 = box
        # legend_panel 기준 상대 좌표로 변환
        rel_box = (bx1, by1 - y1, bx2, by2 - y1)

        # 원형 라벨 표시
        cx, cy = int((rel_box[0] + rel_box[2]) // 2), int((rel_box[1] + rel_box[3]) // 2)
        r = max(10, int(0.6 * min(rel_box[2] - rel_box[0], rel_box[3] - rel_box[1])))
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(0, 0, 0), width=2)

        # 텍스트 그리기 (Pillow 버전별 호환)
        try:
            w, h = draw.textbbox((0, 0), str(label_id), font=font)[2:]
        except Exception:
            w, h = draw.textlength(str(label_id), font=font), params.annotate_font_size
        draw.text((cx - int(w) // 2, cy - int(h) // 2), str(label_id), fill=(0, 0, 0), font=font)

        # id → (라벨 문자열, bbox) 매핑 저장 (bbox는 legend_panel 기준)
        bbox_mapping[label_id] = (
            legend[label_id - 1],
            (rel_box[0], rel_box[1], rel_box[2], rel_box[3]),
        )
        label_id += 1

    labeled = legend_panel.copy()
    return legend_panel, labeled, bbox_mapping


# ------------------------------
# OCR-first: detect legend text
# ------------------------------

# def _ocr_text_regions(pil_img: Image.Image):
#     """Return list of (text, (x,y,w,h)) using pytesseract or easyocr on a PIL image."""
#     cv_img = pil_to_cv(pil_img)
#     out = []
#     if importlib.util.find_spec("pytesseract") is not None:
#         import pytesseract
#         data = pytesseract.image_to_data(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB), output_type=pytesseract.Output.DICT)
#         for i in range(len(data['text'])):
#             txt = (data['text'][i] or "").strip()
#             if not txt:
#                 continue
#             x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
#             out.append((txt, (x, y, w, h)))
#         return out
#     if importlib.util.find_spec("easyocr") is not None:
#         import easyocr
#         use_gpu = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
#         reader = easyocr.Reader(['en'], gpu=use_gpu)
#         results = reader.readtext(cv_img)
#         for (bbox, txt, conf) in results:
#             xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
#             x, y, w, h = min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)
#             out.append((txt.strip(), (x,y,w,h)))
#         return out
#     raise RuntimeError("No OCR backend found. Install pytesseract or easyocr to auto-detect legend text.")

def detect_legend_text(image: Image.Image, max_items: int = 20, ocr: str = "auto"):
    cv_img = pil_to_cv(image)
    try:
        (x1, y1, x2, y2), panel = _find_legend_panel(cv_img, PARAMS)
    except Exception:
        panel = None

    if panel is None:
        return [], None

    rows = _split_rows_by_projection(panel, PARAMS)
    H, W = panel.shape[:2]
    if len(rows) < 2 or H < 30 or W < 60 or (H / max(1, W) < 0.15):
        return [], None

    legend_panel = cv_to_pil(panel)

    # --- 업스케일로 OCR 안정화 (작은 글씨 대응) ---
    w, h = legend_panel.size
    scale = 1.6
    legend_panel_up = legend_panel.resize((int(w*scale), int(h*scale)), Image.BICUBIC)

    regs = _ocr_text_regions(legend_panel_up, backend=ocr)
    # 업스케일 좌표를 원본 크기로 환원 (현재는 텍스트만 쓰므로 좌표는 필요 없지만, 유지)
    text_list = [t for (_bb, t) in regs][:max_items]

    if not text_list:
        return [], None

    return text_list, legend_panel


def annotate_legend_auto(image: Image.Image, max_items: int = 20, ocr: str = "auto"):
    # 1) 강화 탐지 먼저
    legend_list, legend_img = detect_legend_text_robust(image, max_items=max_items, ocr=ocr)
    # 2) 실패 시 기존 경로 재시도
    if not legend_list or legend_img is None:
        legend_list, legend_img = detect_legend_text(image, max_items=max_items, ocr=ocr)

    if not legend_list or legend_img is None:
        safe = image
        return safe, safe.copy(), {}

    legend_panel, labeled, bbox_map = annotate_legend(image, legend_list, PARAMS)
    return legend_panel, labeled, bbox_map


# ------------------------------
# Clean chart image
# ------------------------------

def clean_chart_image(image: Image.Image, title: Optional[str]=None, legend: Optional[List[str]]=None,
                      pad_expand_px: int = 6) -> Image.Image:
    cv_img = pil_to_cv(image)
    H, W = cv_img.shape[:2]
    out = cv_img.copy()

    # Remove title (simple heuristic)
    top_h = max(10, int(0.2*H))
    top = out[:top_h]
    edges = cv2.Canny(cv2.cvtColor(top, cv2.COLOR_BGR2GRAY), 60, 120)
    proj = edges.sum(axis=1)
    if proj.max() > 0:
        cutoff = np.argmax(proj < 0.15 * proj.max())
        if cutoff > 5:
            y2 = min(top_h, cutoff + pad_expand_px)
            out = out[y2:, :]

    # Remove legend (if labels provided)
    if legend:
        try:
            (x1,y1,x2,y2), panel = _find_legend_panel(cv_img, PARAMS)
            y1b, y2b = max(0,y1-pad_expand_px), min(H,y2+pad_expand_px)
            x1b, x2b = max(0,x1-pad_expand_px), min(W,x2+pad_expand_px)
            mask = np.zeros_like(out)
            mask[y1b:y2b, x1b:x2b] = 255
            out = cv2.inpaint(out, cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), 3, cv2.INPAINT_TELEA)
        except Exception:
            pass

    return cv_to_pil(out)

# ------------------------------
# SAM checkpoint resolver
# ------------------------------

def _resolve_sam_checkpoint(
    sam_checkpoint: Optional[str],
    sam_hf_repo: Optional[str],
    sam_hf_filename: Optional[str],
    hf_token: Optional[str] = None
) -> Optional[str]:
    """
    If local checkpoint path is provided, return it (if exists).
    Else, if Hugging Face repo+filename are provided and huggingface_hub is installed,
    download to cache and return local path.
    """
    if sam_checkpoint and os.path.exists(sam_checkpoint):
        return sam_checkpoint

    if (sam_checkpoint and not os.path.exists(sam_checkpoint)):
        # Warn but try HF if provided
        pass

    if sam_hf_repo and sam_hf_filename:
        if importlib.util.find_spec("huggingface_hub") is None:
            raise RuntimeError(
                "huggingface_hub is not installed. Install it or provide a local sam_checkpoint."
            )
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=sam_hf_repo,
            filename=sam_hf_filename,
            token=hf_token,
            local_dir=None
        )
        return path

    return sam_checkpoint if sam_checkpoint else None

# ------------------------------
# Segment & mark (SAM optional)
# ------------------------------

def _load_sam_model(checkpoint: str, model_name: str = "vit_b", device: str = "cuda"):
    from segment_anything import sam_model_registry
    sam = sam_model_registry[model_name](checkpoint=checkpoint)
    sam.to(device=device)
    return sam

def segment_and_mark(
    image: Image.Image,
    segmentation_model: str = "SAM",
    min_area: int = 5000,
    iou_thresh_unique: float = 0.9,
    iou_thresh_composite: float = 0.98,
    white_ratio_thresh: float = 0.95,
    remove_background_color: bool = True,
    sam_checkpoint: Optional[str] = None,
    sam_model_name: str = "vit_b",
    device: str = "cuda",
    sam_hf_repo: Optional[str] = None,      # NEW: HF repo id
    sam_hf_filename: Optional[str] = None,  # NEW: HF filename
    hf_token: Optional[str] = None          # NEW: HF token (optional/private repos)
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    cv_img = pil_to_cv(image)
    H, W = cv_img.shape[:2]

    masks_raw = []

    if segmentation_model.upper() == "SAM":
        # Try to resolve checkpoint (local or HF)
        resolved = _resolve_sam_checkpoint(sam_checkpoint, sam_hf_repo, sam_hf_filename, hf_token)
        if resolved:
            from segment_anything import SamAutomaticMaskGenerator
            sam = _load_sam_model(resolved, sam_model_name, device)
            mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=256,
            )
            sam_masks = mask_generator.generate(cv_img)
            for m in sam_masks:
                seg = (m['segmentation'].astype(np.uint8) * 255)
                masks_raw.append(seg)
        else:
            # No checkpoint → fallback
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 35, 5)
            masks_raw = [thr]
    else:
        # Fallback (non-SAM)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 35, 5)
        masks_raw = [thr]

    cleaned: List[Dict[str, Any]] = []
    label_img = np.zeros_like(cv_img)

    for raw in masks_raw:
        cnts, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            area = w*h
            if area < min_area:
                continue
            mask = np.zeros((H,W), dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)
            if remove_background_color:
                roi = cv_img[y:y+h, x:x+w]
                white_ratio = (roi > 240).mean()
                if white_ratio > white_ratio_thresh:
                    continue
            cleaned.append({'mask': (mask>0), 'bbox': (x,y,x+w,y+h), 'area': int(area)})

    # De-duplicate by IoU (greedy)
    dedup: List[Dict[str, Any]] = []
    for m in sorted(cleaned, key=lambda d: -d['area']):
        keep = True
        for d in dedup:
            iou = _bbox_iou(m['bbox'], d['bbox'])
            if iou > iou_thresh_unique or iou > iou_thresh_composite:
                keep = False; break
        if keep:
            dedup.append(m)

    for i, m in enumerate(dedup, start=1):
        x1,y1,x2,y2 = m['bbox']
        cv2.rectangle(label_img, (x1,y1), (x2,y2), (0,0,255), 2)
        cv2.putText(label_img, str(i), (x1, max(0,y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        m['label'] = i

    labeled_pil = cv_to_pil(cv2.addWeighted(cv_img, 0.8, label_img, 0.7, 0))
    return labeled_pil, dedup

# ------------------------------
# Axes & mapping
# ------------------------------

def axis_localizer(
    image: Image.Image,
    axis: str = "x",
    axis_threshold: float = 0.2,
    axis_tickers: Optional[List[str]] = None,
) -> Tuple[List[float], List[int]]:
    axis = axis.lower()
    if axis not in {'x','y','left_y','right_y'}:
        raise ValueError("axis must be one of 'x', 'y', 'left_y', 'right_y'")
    cv_img = pil_to_cv(image)
    H,W = cv_img.shape[:2]

    frac = max(0.05, min(0.45, axis_threshold))
    if axis == 'x':
        strip = cv_img[int(H*(1-frac)):H, :]
        axis_dir = 'x'
    else:
        if axis == 'right_y':
            strip = cv_img[:, int(W*(1-frac)):W]
        else:
            strip = cv_img[:, :int(W*frac)]
        axis_dir = 'y'

    text_boxes = []
    if importlib.util.find_spec("pytesseract") is not None:
        import pytesseract
        data = pytesseract.image_to_data(cv2.cvtColor(strip, cv2.COLOR_BGR2RGB), output_type=pytesseract.Output.DICT)
        for i in range(len(data['text'])):
            txt = (data['text'][i] or "").strip()
            if not txt:
                continue
            try:
                val = float(txt.replace(',', ''))
            except Exception:
                continue
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            text_boxes.append((val, x, y, w, h))
    elif importlib.util.find_spec("easyocr") is not None:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False)
        results = reader.readtext(strip)
        for (bbox, txt, conf) in results:
            txt = (txt or "").strip()
            try:
                val = float(txt.replace(',', ''))
            except Exception:
                continue
            xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
            x, y, w, h = min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)
            text_boxes.append((val, x, y, w, h))
    else:
        raise RuntimeError("No OCR backend found. Install pytesseract or easyocr.")

    axis_values, axis_pixels = [], []
    for (val, x, y, w, h) in text_boxes:
        cx = int(x + w/2); cy = int(y + h/2)
        if axis_dir == 'x':
            px = cx
            if 0 <= px < W:
                axis_values.append(val); axis_pixels.append(px)
        else:
            py = cy
            if 0 <= py < H:
                axis_values.append(val); axis_pixels.append(py)

    order = np.argsort(axis_pixels)
    axis_values = [float(axis_values[i]) for i in order]
    axis_pixels = [int(axis_pixels[i]) for i in order]
    return axis_values, axis_pixels

def interpolate_pixel_to_value(pixel: float, axis_values: List[float], axis_pixel_positions: List[int]) -> float:
    if len(axis_values) != len(axis_pixel_positions) or len(axis_values) < 2:
        raise ValueError(
            "interpolate_pixel_to_value needs axis_values and axis_pixel_positions (length>=2). "
            "Tip: Get them from axis_localizer()."
        )
    pts = sorted(zip(axis_pixel_positions, axis_values))
    pxs, vals = zip(*pts)
    if pixel <= pxs[0]:
        return float(vals[0])
    if pixel >= pxs[-1]:
        return float(vals[-1])
    for i in range(1, len(pxs)):
        if pixel <= pxs[i]:
            x0, x1 = pxs[i-1], pxs[i]
            y0, y1 = vals[i-1], vals[i]
            t = (pixel - x0) / (x1 - x0 + 1e-8)
            return float(y0 + t*(y1 - y0))
    return float(vals[-1])

# ------------------------------
# Arithmetic
# ------------------------------

def arithmetic(a: float, b: float, operation: str="percentage") -> float:
    if a is None or b is None:
        raise ValueError("arithmetic needs numeric a and b. Example: {'a':300,'b':1200,'operation':'percentage'}")
    op = operation.lower()
    if op == "add":
        return float(a + b)
    if op == "subtract":
        return float(a - b)
    if op == "multiply":
        return float(a * b)
    if op == "divide":
        if b == 0: raise ValueError("Division by zero")
        return float(a / b)
    if op == "percentage":
        if b == 0: raise ValueError("Division by zero")
        return float(100.0 * a / b)
    if op == "ratio":
        if b == 0: raise ValueError("Division by zero")
        return float(a / b)
    raise ValueError(f"Unsupported operation: {operation}")

# ------------------------------
# Dispatcher
# ------------------------------

def run_tool(image_path: str, tool: str, **kwargs) -> Any:
    img = Image.open(image_path).convert("RGB")
    tool = tool.lower()

    if tool == "detect_legend_text":
        # 콘솔 출력용으로 리스트만 보고 싶으면 이 툴을 쓰세요
        texts, legend_img = detect_legend_text(img,
                                               max_items=kwargs.get("max_items", 20),
                                               ocr=kwargs.get("ocr", "auto"))
        return texts, legend_img

    if tool == "annotate_legend":
        # NOTE: legend 미제공 시 자동 OCR로 우회
        return annotate_legend(img, kwargs.get("legend"), kwargs.get("params", PARAMS))

    if tool == "annotate_legend_auto":
        return annotate_legend_auto(img,
                                    max_items=kwargs.get("max_items", 20),
                                    ocr=kwargs.get("ocr", "auto"))

    if tool == "get_marker_rgb":
        return get_marker_rgb(kwargs["legend_image"], kwargs["bbox_mapping"],
                              kwargs.get("text_of_interest"), kwargs.get("label_of_interest"),
                              kwargs.get("distance_between_text_and_marker", 5))

    if tool == "clean_chart_image":
        return clean_chart_image(img, kwargs.get("title"), kwargs.get("legend"), kwargs.get("pad_expand_px", 6))

    if tool == "segment_and_mark":
        return segment_and_mark(
            img,
            kwargs.get("segmentation_model","SAM"),
            kwargs.get("min_area", 5000),
            kwargs.get("iou_thresh_unique", 0.9),
            kwargs.get("iou_thresh_composite", 0.98),
            kwargs.get("white_ratio_thresh", 0.95),
            kwargs.get("remove_background_color", True),
            kwargs.get("sam_checkpoint"),
            kwargs.get("sam_model_name","vit_b"),
            kwargs.get("device","cuda"),
            kwargs.get("sam_hf_repo"),           # NEW
            kwargs.get("sam_hf_filename"),       # NEW
            kwargs.get("hf_token")               # NEW
        )

    if tool == "axis_localizer":
        return axis_localizer(img,
                              kwargs.get("axis","x"),
                              kwargs.get("axis_threshold",0.2),
                              kwargs.get("axis_tickers"))

    if tool == "interpolate_pixel_to_value":
        return interpolate_pixel_to_value(kwargs.get("pixel"), kwargs.get("axis_values"), kwargs.get("axis_pixel_positions"))

    if tool == "arithmetic":
        return arithmetic(kwargs.get("a"), kwargs.get("b"), kwargs.get("operation","percentage"))

    raise ValueError(f"Unknown tool: {tool}")

# ------------------------------
# Color from legend marker
# ------------------------------

def get_marker_rgb(
    legend_image: Image.Image,
    bbox_mapping: Dict[int, Tuple[str, Tuple[int,int,int,int]]],
    text_of_interest: Optional[str] = None,
    label_of_interest: Optional[int] = None,
    distance_between_text_and_marker: int = 5,
) -> Tuple[int,int,int]:
    """Dominant RGB color of a legend marker, either by label or text."""
    if (text_of_interest is None) == (label_of_interest is None):
        raise ValueError("Provide exactly one of text_of_interest or label_of_interest.")
    target_label = None
    if label_of_interest is not None:
        if label_of_interest not in bbox_mapping:
            raise KeyError(f"label {label_of_interest} not found")
        target_label = label_of_interest
    else:
        lo = text_of_interest.lower()
        for k,(txt,_) in bbox_mapping.items():
            if lo in txt.lower():
                target_label = k
                break
        if target_label is None:
            raise KeyError(f"text '{text_of_interest}' not found in bbox_mapping")

    _, (x1,y1,x2,y2) = bbox_mapping[target_label]
    cv_panel = pil_to_cv(legend_image)
    roi = cv_panel[max(0,y1):y2, max(0,x1):x2, :]
    if roi.size == 0:
        return (0,0,0)
    h, w = roi.shape[:2]
    mx1, mx2 = int(0.1*w), int(0.9*w)
    my1, my2 = int(0.1*h), int(0.9*h)
    roi_c = roi[my1:my2, mx1:mx2, :]
    hsv = cv2.cvtColor(roi_c, cv2.COLOR_BGR2HSV)
    med = np.median(hsv.reshape(-1,3), axis=0).astype(np.uint8)
    rgb = cv2.cvtColor(med.reshape(1,1,3), cv2.COLOR_HSV2BGR)[0,0]
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))
