import os
import time
import json
import base64
import cv2
import numpy as np
import gradio as gr
from fastapi import FastAPI
from contextlib import asynccontextmanager
from PIL import Image
import io

os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = "0.0"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"

local_tmp_dir = os.path.join(os.getcwd(), "gradio_tmp")
os.makedirs(local_tmp_dir, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = local_tmp_dir

# --- Configuration ---
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddleocr import PaddleOCR, PPStructureV3, TextRecognition

# --- Global Model Cache ---
model_cache = {"ocr": {}, "structure": None}


def get_ocr_model(language="german"):
    if language not in model_cache["ocr"]:
        print(f"⚡ Loading OCR model for language: {language}...")
        model_cache["ocr"][language] = PaddleOCR(
            lang=language,
            device="gpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
        )
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        try:
            list(model_cache["ocr"][language].predict(dummy))
        except:
            pass
        print(f"✅ OCR model loaded for {language}.")
    return model_cache["ocr"][language]


def get_ocr_model_mobile(language="german"):
    key = f"{language}_mobile"
    if key not in model_cache["ocr"]:
        print(f"⚡ Loading Mobile OCR model for language: {language}...")
        model_cache["ocr"][key] = PaddleOCR(
            lang=language,
            device="gpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            text_detection_model_name="PP-OCRv5_mobile_det",
        )
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        try:
            list(model_cache["ocr"][key].predict(dummy))
        except:
            pass
        print(f"✅ Mobile OCR model loaded for {language}.")
    return model_cache["ocr"][key]


def get_structure_model():
    if not model_cache["structure"]:
        print("⚡ Loading Layout/Structure model...")
        model_cache["structure"] = PPStructureV3(
            device="gpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        try:
            list(model_cache["structure"].predict(dummy))
        except:
            pass
        print("✅ Layout/Structure model loaded.")
    return model_cache["structure"]

def get_rec_model(lang="german"):
    key = f"{lang}_rec"
    if key not in model_cache["ocr"]:
        print(f"⚡ Loading recognition-only model for: {lang}...")
        model_name = "PP-OCRv5_server_rec"
        model_cache["ocr"][key] = TextRecognition(
            model_name=model_name,
            device="gpu",
        )
        dummy = np.zeros((32, 200, 3), dtype=np.uint8)
        try:
            list(model_cache["ocr"][key].predict(input=dummy, batch_size=1))
        except:
            pass
        print(f"✅ Recognition model loaded for {lang}.")
    return model_cache["ocr"][key]

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_ocr_model("german")
    get_ocr_model_mobile("german")
    get_structure_model()
    get_rec_model("german")
    yield

# ============================================================
# Interactive HTML Visualization
# ============================================================

def _img_to_data_uri(image_rgb: np.ndarray, max_width=1400) -> tuple[str, int, int]:
    """Convert numpy RGB image to a base64 data URI, optionally downscale."""
    h, w = image_rgb.shape[:2]
    scale = 1.0
    if w > max_width:
        scale = max_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w

    pil_img = Image.fromarray(image_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}", w, h, scale


def _conf_color(conf: float) -> str:
    """Return HSL color string: red(0) -> yellow(0.5) -> green(1.0)."""
    hue = int(conf * 120)  # 0=red, 60=yellow, 120=green
    return f"hsl({hue}, 90%, 45%)"


def _conf_color_fill(conf: float) -> str:
    hue = int(conf * 120)
    return f"hsla({hue}, 90%, 50%, 0.12)"


LAYOUT_COLORS = {
    "paragraph_title": "#e64e4e",
    "text": "#3dae3d",
    "image": "#4e5ee6",
    "table": "#d9922b",
    "figure": "#a84ee6",
    "header": "#2db8b8",
    "footer": "#b8b82d",
    "reference": "#8a8a8a",
    "equation": "#e65eaa",
}
LAYOUT_DEFAULT = "#ff6464"


def _escape_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")

def build_interactive_html(
    image_rgb: np.ndarray,
    detections: list,
    mode: str,
    inference_ms: float = 0,
    viz_theme: str = "Dark",
) -> str:
    """
    Build HTML with tooltips and adaptive theme support.
    Note: CSS braces are doubled {{ }} to work inside the Python f-string.
    """
    orig_h, orig_w = image_rgb.shape[:2]
    data_uri, disp_w, disp_h, scale = _img_to_data_uri(image_rgb)

    # Prepare theme class for the root div
    theme_class = viz_theme.lower().replace(" ", "-")

    overlay_divs = []

    for i, det in enumerate(detections):
        # 1. Extract bbox and content
        if mode == "layout":
            label = det.get("label", "unknown")
            score = det.get("score", 0)
            bbox = det.get("bbox", [0, 0, 0, 0])
            content_preview = det.get("content", "")
            color = LAYOUT_COLORS.get(label, LAYOUT_DEFAULT)
            tooltip_header = f'<div class="tt-label" style="color:{color}">{_escape_html(label.upper())}</div>'
        else:
            text = det.get("text", "")
            conf = det.get("confidence", 0)
            poly = det.get("polygon", det.get("box", []))
            if not poly or len(poly) < 3: continue
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
            content_preview = text
            stroke_color = _conf_color(conf)
            fill_color = _conf_color_fill(conf)
            tooltip_header = f'<div class="tt-conf" style="color:{stroke_color}">Conf: <b>{conf:.1%}</b></div>'

        # 2. Calculate percentages
        x1, y1, x2, y2 = bbox
        left_pct, top_pct = (x1 / orig_w) * 100, (y1 / orig_h) * 100
        w_pct, h_pct = ((x2 - x1) / orig_w) * 100, ((y2 - y1) / orig_h) * 100

        # 3. Positioning Logic
        v_pos_class = "tt-down" if top_pct < 15 else "tt-up"
        h_pos_class = "tt-right" if left_pct > 75 else "tt-left"

        if content_preview and len(content_preview) > 120:
            content_preview = content_preview[:120] + "…"
        
        if mode == "layout":
            tooltip_inner = f'{tooltip_header}<div class="tt-conf">Conf: <b>{score:.0%}</b></div>'
            if content_preview:
                tooltip_inner += f'<div class="tt-text">{_escape_html(content_preview)}</div>'
        else:
            tooltip_inner = f'<div class="tt-text">{_escape_html(content_preview)}</div>{tooltip_header}'

        style_box = f'left:{left_pct:.3f}%;top:{top_pct:.3f}%;width:{w_pct:.3f}%;height:{h_pct:.3f}%;'

        if mode == "layout":
            overlay_divs.append(
                f'<div class="det-box" style="{style_box} border-color:{color};background:{color}18;">'
                f'<div class="tt {v_pos_class} {h_pos_class}">{tooltip_inner}</div>'
                f'</div>'
            )
        else:
            box_w, box_h = (x2 - x1) or 1, (y2 - y1) or 1
            svg_points = " ".join(f"{(p[0] - x1):.1f},{(p[1] - y1):.1f}" for p in poly)
            overlay_divs.append(
                f'<div class="det-box det-poly" style="{style_box} border-color:transparent;">'
                f'<svg class="poly-svg" viewBox="0 0 {box_w:.1f} {box_h:.1f}" preserveAspectRatio="none">'
                f'<polygon points="{svg_points}" fill="{fill_color}" stroke="{stroke_color}" stroke-width="2" vector-effect="non-scaling-stroke"/>'
                f'</svg>'
                f'<div class="tt {v_pos_class} {h_pos_class}">{tooltip_inner}</div>'
                f'</div>'
            )

    overlays_html = "\n".join(overlay_divs)
    mode_label = mode.replace("_", " ").title()

    # --- Pre-build Legend to avoid backslash issues in f-string ---
    legend_html = ""
    if mode == "layout":
        labels_found = set(d.get("label", "") for d in detections)
        legend_items = []
        for lbl in labels_found:
            clr = LAYOUT_COLORS.get(lbl, LAYOUT_DEFAULT)
            legend_items.append(f'<div class="legend-item"><div class="legend-dot" style="background:{clr}"></div>{lbl}</div>')
        legend_html = f'<div class="viz-legend">{"".join(legend_items)}</div>'

    return f"""
<div class="ocr-viz-root {theme_class}">
<style>
  .ocr-viz-root {{
    --bg: #0e0e11; --surface: #18181c; --border: #2a2a32;
    --text-primary: #e8e8ec; --text-secondary: #8e8e9a;
    --tt-bg: rgba(24, 24, 28, 0.95); --tt-border: #3a3a48;
    --tt-shadow: rgba(0, 0, 0, 0.5);
    font-family: 'JetBrains Mono', monospace;
    background: var(--bg); border-radius: 12px; overflow: hidden;
  }}

  /* LIGHT THEME OVERRIDES */
  .ocr-viz-root.light {{
    --bg: #f4f4f7; 
    --surface: #ffffff; 
    --border: #d1d1db;
    --text-primary: #1a1a1e;      /* Dark text for headers */
    --text-secondary: #62626e;    /* Gray text for headers */
    --tt-bg: rgba(255, 255, 255, 0.98); 
    --tt-border: #bcbccb;
    --tt-shadow: rgba(0, 0, 0, 0.1);
  }}

  .ocr-viz-root.light .tt {{
    color: #1a1a1e !important;   /* Force dark text for everything in tooltip */
  }}

  .ocr-viz-root.light .tt-text {{
    color: #1a1a1e !important;
    border-top: 1px solid #d1d1db; /* Use a dark border for the separator */
  }}

  .ocr-viz-root.light .tt-conf {{
    /* Keep the dynamic colors (green/red) but ensure they are dark enough */
    filter: brightness(0.8); 
  }}

  .ocr-viz-root.high-contrast {{
    --bg: #ffffff; --surface: #f0f0f3; --border: #cccccc;
    --text-primary: #000000; --text-secondary: #444444;
    --tt-bg: rgba(15, 15, 20, 0.98); --tt-border: #444455;
    --tt-shadow: rgba(0, 0, 0, 0.6);
  }}

  .viz-header {{ display: flex; justify-content: space-between; padding: 12px 18px; background: var(--surface); border-bottom: 1px solid var(--border); }}
  .viz-canvas {{ position: relative; overflow: auto; max-height: 84vh; background: var(--bg); }}
  .viz-img-wrap {{ position: relative; display: inline-block; width: 100%; }}
  .viz-img-wrap img {{ display: block; width: 100%; }}
  
  .det-box {{ position: absolute; border: 1px solid; z-index: 10; }}
  .det-box:hover {{ z-index: 100; border-width: 2px; cursor: crosshair; }}
  .poly-svg {{ position: absolute; width: 100%; height: 100%; pointer-events: none; }}

  .tt {{
    display: none; position: absolute; z-index: 9999;
    background: var(--tt-bg); backdrop-filter: blur(8px);
    border: 1px solid var(--tt-border); border-radius: 6px;
    padding: 8px 10px; width: max-content; max-width: 320px;
    box-shadow: 0 4px 12px var(--tt-shadow); pointer-events: none;
    flex-direction: column; gap: 4px; align-items: flex-start;
  }}
  
  .high-contrast .tt {{ --text-primary: #eeeeee; --text-secondary: #bbbbbb; }}

  .det-box:hover > .tt {{ display: flex; }}
  .tt.tt-up {{ bottom: 100%; top: auto; margin-bottom: 6px; }}
  .tt.tt-down {{ top: 100%; bottom: auto; margin-top: 6px; }}
  .tt.tt-left {{ left: 0; right: auto; }}
  .tt.tt-right {{ right: 0; left: auto; }}
  
  .tt-label {{ font-size: 11px; font-weight: 800; text-transform: uppercase; color: var(--text-primary); }}
  .tt-conf {{ font-size: 10px; color: var(--text-secondary); }}
  .tt-text {{ 
    font-size: 11px; color: var(--text-primary); display: block; 
    line-height: 1.4; padding-top: 4px; margin-top: 2px;
    border-top: 1px solid var(--border); width: 100%; 
  }}
  
  .viz-legend {{ padding: 10px; display: flex; gap: 10px; flex-wrap: wrap; background: var(--surface); border-top: 1px solid var(--border); }}
  .legend-item {{ font-size: 10px; color: var(--text-secondary); display: flex; align-items: center; gap: 4px; }}
  .legend-dot {{ width: 8px; height: 8px; border-radius: 2px; }}
</style>

<div class="viz-header">
  <div style="color:var(--text-primary); font-size:12px; font-weight:bold;">{mode_label}</div>
  <div style="color:var(--text-secondary); font-size:11px;">{len(detections)} items • {inference_ms:.0f}ms</div>
</div>

<div class="viz-canvas">
  <div class="viz-img-wrap">
    <img src="{data_uri}" />
    {overlays_html}
  </div>
</div>
{legend_html}
</div>
"""

# ============================================================
# Processing Logic
# ============================================================

def process_analysis(image, mode, language, viz_theme):
    if image is None:
        return "<div style='padding:40px;color:#888;text-align:center;'>Upload an image to begin.</div>", []

    image_rgb = image
    raw_data = []
    html_output = ""

    if mode in ("Standard OCR", "Mobile OCR"):
        engine = (
            get_ocr_model_mobile(language)
            if mode == "Mobile OCR"
            else get_ocr_model(language)
        )

        start = time.perf_counter()
        results = list(engine.predict(image_rgb))
        elapsed = time.perf_counter() - start

        detections = []
        for result in results:
            if (
                "rec_polys" in result
                and "rec_texts" in result
                and "rec_scores" in result
            ):
                for poly, text, score in zip(
                    result["rec_polys"], result["rec_texts"], result["rec_scores"]
                ):
                    detections.append(
                        {
                            "text": text,
                            "confidence": round(float(score), 4),
                            "polygon": [[int(p[0]), int(p[1])] for p in poly],
                        }
                    )

        raw_data = detections
        html_output = build_interactive_html(
            image_rgb, detections, mode="ocr", inference_ms=elapsed * 1000, viz_theme=viz_theme
        )

    else:  # Layout Extraction
        engine = get_structure_model()

        start = time.perf_counter()
        results = list(engine.predict(image_rgb))
        elapsed = time.perf_counter() - start

        layout_items = []
        parsed_lookup = {}

        for result in results:
            layout_res = result.get("layout_det_res", {})
            boxes = (
                layout_res.get("boxes", [])
                if hasattr(layout_res, "get")
                else getattr(layout_res, "boxes", [])
            )

            for box in boxes:
                if hasattr(box, "get"):
                    label = box.get("label", "unknown")
                    score = round(float(box.get("score", 0)), 2)
                    coord = box.get("coordinate", [0, 0, 0, 0])
                else:
                    label = getattr(box, "label", "unknown")
                    score = round(float(getattr(box, "score", 0)), 2)
                    coord = getattr(box, "coordinate", [0, 0, 0, 0])

                x1, y1, x2, y2 = [int(float(c)) for c in coord]
                layout_items.append(
                    {
                        "label": label,
                        "score": score,
                        "bbox": [x1, y1, x2, y2],
                    }
                )

            parsing_list = (
                result.get("parsing_res_list", [])
                if hasattr(result, "get")
                else getattr(result, "parsing_res_list", [])
            )
            for block in parsing_list:
                content = (
                    block.get("content", "")
                    if hasattr(block, "get")
                    else getattr(block, "content", "")
                )
                blabel = (
                    block.get("label", "")
                    if hasattr(block, "get")
                    else getattr(block, "label", "")
                )
                bbox = (
                    block.get("bbox", None)
                    if hasattr(block, "get")
                    else getattr(block, "bbox", None)
                )
                if content and bbox:
                    # Try to match parsed content to layout boxes
                    key = f"{int(bbox[0])},{int(bbox[1])}"
                    parsed_lookup[key] = content.strip()

        # Attach parsed content to layout items for tooltips
        for item in layout_items:
            bx = item["bbox"]
            key = f"{bx[0]},{bx[1]}"
            if key in parsed_lookup:
                item["content"] = parsed_lookup[key]

        raw_data = layout_items
        html_output = build_interactive_html(
            image_rgb, layout_items, mode="layout", inference_ms=elapsed * 1000, viz_theme=viz_theme
        )

    return html_output, raw_data


# --- Gradio UI ---
with gr.Blocks(
    title="PaddleOCR Pro",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.neutral,
        neutral_hue=gr.themes.colors.zinc,
    ),
    css="""
    .gradio-container { max-width: 1600px !important; }
    #ocr-viz-root { min-height: 400px; }
    """,
) as ui:
    gr.Markdown("### 🔍 PaddleOCR & Layout Analysis — Interactive Viewer")

    with gr.Row():
        with gr.Column(scale=1, min_width=260):
            input_img = gr.Image(label="Input Image", type="numpy")
            with gr.Accordion("UI Settings", open=True):
                theme_toggle = gr.Radio(
                    choices=["Dark", "Light", "High Contrast"], 
                    value="Dark", 
                    label="Viewer Theme"
                )
            lang_dropdown = gr.Dropdown(
                choices=["german", "ar", "en", "fr", "ch"],
                value="german",
                label="Language",
            )
            mode_toggle = gr.Radio(
                ["Standard OCR", "Mobile OCR", "Layout Extraction"],
                value="Standard OCR",
                label="Mode",
            )
            run_btn = gr.Button("⚡ Analyze", variant="primary", size="lg")

        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.TabItem("Interactive View"):
                    output_html = gr.HTML(
                        value="<div style='padding:60px;color:#555;text-align:center;font-family:monospace;'>Upload an image and click Analyze</div>"
                    )
                with gr.TabItem("Raw JSON"):
                    output_json = gr.JSON(label="Structured Output")

    run_btn.click(
        fn=process_analysis,
        inputs=[input_img, mode_toggle, lang_dropdown, theme_toggle],
        outputs=[output_html, output_json],
    )

app = gr.mount_gradio_app(app, ui, path="/")

if __name__ == "__main__":
    import uvicorn

    print("Starting PaddleOCR Pro Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
