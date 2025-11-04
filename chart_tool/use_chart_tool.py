#!/usr/bin/env python

"""
use_chart_tool.py — Minimal CLI for chart_tools.run_tool

이 스크립트는 chart_tools.run_tool()에 이미지를 넘겨
원하는 차트 유틸리티를 실행하는 간단한 CLI입니다.

================================================================================
AVAILABLE TOOLS (run_tool의 tool 인자값)
================================================================================

[Legend 관련]
(1) annotate_legend_auto
    - 이미지에서 범례 패널을 자동 탐색 → OCR로 항목 텍스트 추출 →
      각 항목(마커) 위치에 번호 라벨 오버레이.
    - kwargs:
        ocr: "auto" | "paddle" | "tesseract" | "easyocr"   (기본 "auto")
        max_items: int                                     (기본 20)
      ※ chart_tools.py에서 detect_legend_text_robust → detect_legend_text 순으로 시도.
    - return: (legend_crop_img: PIL.Image, legend_labeled_img: PIL.Image,
               bbox_map: Dict[int, Tuple[str, (x1,y1,x2,y2)]])

(2) detect_legend_text
    - 범례 패널만 찾아 OCR 텍스트 리스트와 패널 crop 이미지를 반환.
    - kwargs:
        ocr: "auto" | "paddle" | "tesseract" | "easyocr"   (기본 "auto")
        max_items: int                                     (기본 20)
    - return: (list_text: List[str], legend_crop_img: PIL.Image or None)

[세그멘테이션/마킹]
(4) segment_and_mark
    - SAM 없이 컨투어 기반 빠른 세그멘테이션 + 라벨 마킹 이미지를 생성.
    - kwargs: (없음)   *SAM 고급 옵션은 chart_tools.py의 확장판 참고
    - return: (segmented_labeled_img: PIL.Image, cleaned_masks: List[ndarray] or similar)

[축/스케일]
(5) axis_localizer
    - x/y 축 영역에서 tick 텍스트를 OCR → 픽셀 좌표와 값 배열을 추출.
    - kwargs:
        axis: "x" | "y"                                    (기본 "x")
        axis_threshold: float                              (기본 0.2)
        ocr: "auto" | "paddle" | "tesseract" | "easyocr"   (기본 "auto")
    - return: (axis_values: List[float or str], axis_pixel_positions: List[int])

(6) interpolate_pixel_to_value
    - (axis_localizer 결과를 사용) 주어진 픽셀 위치 → 선형 보간된 값으로 변환.
    - kwargs:
        pixel: int                                         (필수)
        axis_values: List[float]                           (필수)
        axis_pixel_positions: List[int]                    (필수)
    - return: float

[수치 계산]
(7) arithmetic
    - 간단한 수치 연산 도구.
    - kwargs:
        a: float (필수), b: float (필수),
        operation: "percentage" | "add" | "subtract" | "multiply" | "divide" | "ratio"
    - return: float

[수평 막대 전용(옵션)]
(8) extract_horizontal_bar_items
    - 범례 패널이 없이 좌측 라벨·우측 값이 있는 수평 막대 차트에서
      (label, value, color)를 행 단위로 추출.
    - kwargs:
        ocr: "auto" | "paddle" | "tesseract" | "easyocr"   (기본 "auto")
        max_rows: int                                      (기본 50)
    - return: List[{"label": str, "value": float, "color": (r,g,b)}]

[색상/전처리]
(9) get_marker_rgb
    - 범례 마커(원/사각/라인 조각) 또는 막대/포인트의 대표 색(RGB)을 추출.
    - kwargs:
        bbox: [x1, y1, x2, y2]                             (선택)
        method: "median" | "kmeans"                        (기본 "median")
        k: int                                             (기본 2; method="kmeans"에서 사용)
        ignore_bg: bool                                    (기본 True)
        sample_ratio: float                                (기본 1.0)
    - return: (r, g, b)  정수 3-튜플

(10) clean_chart_image
    - 차트 분석 친화적 전처리(여백 트림, 노이즈 억제, 그리드 약화, 미세 선명화).
    - kwargs:
        trim_bg: bool                                      (기본 True)
        denoise: "none" | "mild" | "strong"                (기본 "mild")
        degrid: bool                                       (기본 True)
        sharpen: bool                                      (기본 True)
        keep_size: bool                                    (기본 False)
    - return: cleaned_img: PIL.Image

--------------------------------------------------------------------------------
NOTE
- OCR 백엔드("ocr" kwarg)는 chart_tools.py에서 플러그형으로 동작합니다.
  "auto"(기본)는 paddle → tesseract → easyocr 순으로 시도합니다.
- 범례가 실제로 없는 차트에서 annotate_legend_auto를 호출하면
  빈 결과(또는 원본 반환)로 처리하는 것이 정상입니다.
- 실행 예시는 파일 하단의 "Examples" 섹션을 참고하세요.
================================================================================
"""

import argparse, json, os, sys
from PIL import Image
from chart_tools import run_tool


def _ensure_outdir(path: str):
    if not path:
        return "."
    os.makedirs(path, exist_ok=True)
    return path


def _save_outputs(out, outdir: str):
    """
    run_tool 반환값을 최대한 친절하게 저장:
      - PIL.Image: output.png (또는 output_{i}.png)
      - (tuple/list 안) PIL.Image: output_{i}.png
      - dict/list 등 JSON 직렬화 가능한 객체: output.json
      - 그 외: 문자열로 output.txt
    """
    saved_any = False
    # 1) 단일 이미지
    if hasattr(out, "save"):
        out_path = os.path.join(outdir, "output.png")
        out.save(out_path)
        print(f"Saved image: {out_path}")
        return True

    # 2) tuple/list 복합
    if isinstance(out, (tuple, list)):
        non_images = []
        for i, item in enumerate(out):
            if hasattr(item, "save"):
                p = os.path.join(outdir, f"output_{i}.png")
                try:
                    item.save(p)
                    print(f"Saved image: {p}")
                    saved_any = True
                except Exception as e:
                    print(f"[warn] failed to save image index {i}: {e}")
            else:
                non_images.append(item)
        # 비-이미지 묶음은 JSON으로 저장 시도
        if non_images:
            try:
                import numpy as _np
                def _to_jsonable(x):
                    if isinstance(x, _np.ndarray):
                        return x.tolist()
                    return x
                j = [ _to_jsonable(x) for x in non_images ]
                jp = os.path.join(outdir, "output.json")
                with open(jp, "w", encoding="utf-8") as f:
                    json.dump(j, f, ensure_ascii=False, indent=2)
                print(f"Saved JSON: {jp}")
                saved_any = True
            except Exception as e:
                print(f"[warn] failed to save JSON: {e}")
        return saved_any

    # 3) dict → JSON
    if isinstance(out, dict):
        try:
            jp = os.path.join(outdir, "output.json")
            with open(jp, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print(f"Saved JSON: {jp}")
            return True
        except Exception as e:
            print(f"[warn] failed to save JSON: {e}")

    # 4) 기타 → TXT
    try:
        tp = os.path.join(outdir, "output.txt")
        with open(tp, "w", encoding="utf-8") as f:
            f.write(str(out))
        print(f"Saved text: {tp}")
        return True
    except Exception as e:
        print(f"[warn] failed to save text: {e}")

    return False


def main():
    p = argparse.ArgumentParser(description="Run chart_tools.run_tool on an image.")
    p.add_argument("--image", required=True, help="Path to chart image")
    p.add_argument("--tool", required=True, help="Tool name (see file header)")
    p.add_argument("--kwargs", default="{}", help="JSON dict of keyword args")
    p.add_argument("--outdir", default=".", help="Directory to save outputs (default: .)")
    args = p.parse_args()

    outdir = _ensure_outdir(args.outdir)

    try:
        kwargs = json.loads(args.kwargs)
        if not isinstance(kwargs, dict):
            raise ValueError("--kwargs must be a JSON object (e.g., '{\"ocr\":\"paddle\"}')")
    except Exception as e:
        print(f"[error] failed to parse --kwargs JSON: {e}")
        sys.exit(1)

    # 실행
    out = run_tool(args.image, args.tool, **kwargs)

    # 저장 & 요약 출력
    saved = _save_outputs(out, outdir)
    if not saved:
        # 저장할 게 없으면 표준출력에 내용 찍기
        print(out)

    # 간단 요약
    try:
        typ = type(out).__name__
        print(f"Return type: {typ}")
        if isinstance(out, (tuple, list)):
            print("Parts:", [type(x).__name__ for x in out])
    except Exception:
        pass


if __name__ == "__main__":
    main()

"""
Examples
--------

# 1) 범례 자동 주석 (OCR 자동 폴백)
python use_chart_tool.py --image chart.png --tool annotate_legend_auto

# 2) 범례 자동 주석 (PaddleOCR 강제)
python use_chart_tool.py --image chart.png --tool annotate_legend_auto --kwargs '{"ocr":"paddle","max_items":12}'

# 3) 범례 텍스트만 보고 싶을 때 (강건 탐지 경로)
python use_chart_tool.py --image chart.png --tool detect_legend_text_robust --kwargs '{"ocr":"auto"}'

# 4) 세그멘테이션 라벨링
python use_chart_tool.py --image chart.png --tool segment_and_mark --outdir ./outs

# 5) 축 로컬라이즈 후 보간 값 구하기 (예: x=512 픽셀의 값)
#    1단계: 축 값/좌표 추출
python use_chart_tool.py --image chart.png --tool axis_localizer --kwargs '{"axis":"x","ocr":"auto"}' --outdir ./outs
#    생성된 outs/output.json 내용을 참고해 interpolate 호출:
python use_chart_tool.py --image chart.png --tool interpolate_pixel_to_value \
  --kwargs '{"pixel":512,"axis_values":[0,10,20,30],"axis_pixel_positions":[100,300,500,700]}'

# 6) 수평 막대 전용 추출 (범례 없는 형식)
python use_chart_tool.py --image chart.png --tool extract_horizontal_bar_items --kwargs '{"ocr":"paddle"}' --outdir ./outs

# 7) 마커 대표 색 추출 (bbox 지정)
python use_chart_tool.py --image chart.png --tool get_marker_rgb \
  --kwargs '{"bbox":[1020,940,1048,968], "method":"kmeans", "k":3}'

# 8) 차트 전처리 클린업
python use_chart_tool.py --image chart.png --tool clean_chart_image \
  --kwargs '{"denoise":"mild","degrid":true,"sharpen":true,"trim_bg":true}' --outdir ./clean
"""
