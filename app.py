import os
import time
import numpy as np
import gradio as gr

from visualizer import build_html

os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = "0.0"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

local_tmp_dir = os.path.join(os.getcwd(), "gradio_tmp")
os.makedirs(local_tmp_dir, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = local_tmp_dir

from paddleocr import PaddleOCR

_model_cache = {}

def get_model(language="german", mobile=False):
    key = f"{language}_{'mobile' if mobile else 'server'}"
    if key not in _model_cache:
        kwargs = dict(
            lang=language,
            device="gpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
        )
        if mobile:
            kwargs["text_detection_model_name"] = "PP-OCRv5_mobile_det"
        print(f"Loading OCR model: {key}...")
        _model_cache[key] = PaddleOCR(**kwargs)
        try:
            list(_model_cache[key].predict(np.zeros((100, 100, 3), dtype=np.uint8)))
        except Exception as e:
            print(f"Warmup warning: {e}")
        print(f"Model ready: {key}")
    return _model_cache[key]

def run_ocr(image, language, mobile, theme):
    if image is None:
        return []
    model = get_model(language, mobile)
    start = time.perf_counter()
    results = list(model.predict(image.copy()))
    elapsed = time.perf_counter() - start

    detections = []
    for result in results:
        for poly, text, score in zip(
            result.get("rec_polys", []),
            result.get("rec_texts", []),
            result.get("rec_scores", []),
        ):
            detections.append({
                "text": text,
                "confidence": round(float(score), 4),
                "polygon": [[int(p[0]), int(p[1])] for p in poly],
            })

    print(f"OCR done: {len(detections)} detections in {elapsed*1000:.0f}ms")
    html = build_html(image, detections, mode="ocr", inference_ms=elapsed * 1000, theme=theme)
    return html, detections

with gr.Blocks(title="PaddleOCR") as ui:
    gr.Markdown("## PaddleOCR")
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Input", type="numpy")
            lang = gr.Dropdown(["german", "en", "fr", "ar", "ch"], value="german", label="Language")
            mobile = gr.Checkbox(label="Mobile model (faster, less accurate)")
            theme = gr.Radio(["dark", "light"], value="dark", label="Theme")
            run_btn = gr.Button("Analyze", variant="primary")
        with gr.Column(scale=2):
            output_html = gr.HTML(label="Visualization")
            output_json = gr.JSON(label="Results")

    run_btn.click(
            fn=run_ocr,
            inputs=[input_img, lang, mobile, theme],
            outputs=[output_html, output_json],
            )

if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0", server_port=8000)
