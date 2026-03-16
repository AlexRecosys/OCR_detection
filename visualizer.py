import base64
import io
import cv2
import numpy as np
from PIL import Image

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


def _img_to_data_uri(image_rgb: np.ndarray, max_width=1400) -> tuple[str, int, int]:
    h, w = image_rgb.shape[:2]
    if w > max_width:
        scale = max_width / w
        image_rgb = cv2.resize(image_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = image_rgb.shape[:2]
    buf = io.BytesIO()
    Image.fromarray(image_rgb).save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}", w, h


def _conf_color(conf: float) -> str:
    return f"hsl({int(conf * 120)}, 90%, 45%)"


def _conf_color_fill(conf: float) -> str:
    return f"hsla({int(conf * 120)}, 90%, 50%, 0.12)"


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _build_overlays(detections: list, mode: str, orig_w: int, orig_h: int) -> str:
    divs = []
    for det in detections:
        if mode == "layout":
            label = det.get("label", "unknown")
            score = det.get("score", 0)
            bbox = det.get("bbox", [0, 0, 0, 0])
            content = det.get("content", "")
            color = LAYOUT_COLORS.get(label, LAYOUT_DEFAULT)
            x1, y1, x2, y2 = bbox
            l = (x1 / orig_w) * 100
            t = (y1 / orig_h) * 100
            w = ((x2 - x1) / orig_w) * 100
            h = ((y2 - y1) / orig_h) * 100
            v = "tt-down" if t < 15 else "tt-up"
            ha = "tt-right" if l > 75 else "tt-left"
            preview = _escape(content[:120] + "…" if len(content) > 120 else content)
            tooltip = (
                f'<div class="tt-label" style="color:{color}">{_escape(label.upper())}</div>'
                f'<div class="tt-conf">Conf: <b>{score:.0%}</b></div>'
                + (f'<div class="tt-text">{preview}</div>' if preview else "")
            )
            divs.append(
                f'<div class="det-box" style="left:{l:.3f}%;top:{t:.3f}%;width:{w:.3f}%;height:{h:.3f}%;'
                f'border-color:{color};background:{color}18;">'
                f'<div class="tt {v} {ha}">{tooltip}</div></div>'
            )
        else:
            text = det.get("text", "")
            conf = det.get("confidence", 0)
            poly = det.get("polygon", [])
            if not poly or len(poly) < 3:
                continue
            xs, ys = [p[0] for p in poly], [p[1] for p in poly]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            l = (x1 / orig_w) * 100
            t = (y1 / orig_h) * 100
            w = ((x2 - x1) / orig_w) * 100
            h = ((y2 - y1) / orig_h) * 100
            bw, bh = (x2 - x1) or 1, (y2 - y1) or 1
            stroke = _conf_color(conf)
            fill = _conf_color_fill(conf)
            v = "tt-down" if t < 15 else "tt-up"
            ha = "tt-right" if l > 75 else "tt-left"
            pts = " ".join(f"{(p[0]-x1):.1f},{(p[1]-y1):.1f}" for p in poly)
            preview = _escape(text[:120] + "…" if len(text) > 120 else text)
            tooltip = (
                f'<div class="tt-text">{preview}</div>'
                f'<div class="tt-conf" style="color:{stroke}">Conf: <b>{conf:.1%}</b></div>'
            )
            divs.append(
                f'<div class="det-box det-poly" style="left:{l:.3f}%;top:{t:.3f}%;width:{w:.3f}%;height:{h:.3f}%;border-color:transparent;">'
                f'<svg class="poly-svg" viewBox="0 0 {bw:.1f} {bh:.1f}" preserveAspectRatio="none">'
                f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="2" vector-effect="non-scaling-stroke"/>'
                f'</svg>'
                f'<div class="tt {v} {ha}">{tooltip}</div></div>'
            )
    return "\n".join(divs)


def _build_legend(detections: list) -> str:
    labels = set(d.get("label", "") for d in detections)
    items = "".join(
        f'<div class="legend-item"><div class="legend-dot" style="background:{LAYOUT_COLORS.get(l, LAYOUT_DEFAULT)}"></div>{l}</div>'
        for l in labels
    )
    return f'<div class="viz-legend">{items}</div>' if items else ""


CSS = """
.ocr-root {
  --bg:#0e0e11;--surface:#18181c;--border:#2a2a32;
  --tp:#e8e8ec;--ts:#8e8e9a;
  --tt-bg:rgba(24,24,28,0.95);--tt-bd:#3a3a48;--tt-sh:rgba(0,0,0,.5);
  font-family:'JetBrains Mono',monospace;
  background:var(--bg);border-radius:12px;overflow:hidden;
}
.ocr-root.light{--bg:#f4f4f7;--surface:#fff;--border:#d1d1db;--tp:#1a1a1e;--ts:#62626e;--tt-bg:rgba(255,255,255,.98);--tt-bd:#bcbccb;--tt-sh:rgba(0,0,0,.1);}
.ocr-root.light .tt{color:#1a1a1e!important;}
.ocr-root.light .tt-text{color:#1a1a1e!important;border-top:1px solid #d1d1db;}
.viz-header{display:flex;justify-content:space-between;padding:12px 18px;background:var(--surface);border-bottom:1px solid var(--border);}
.viz-canvas{position:relative;overflow:auto;max-height:84vh;background:var(--bg);}
.viz-img-wrap{position:relative;display:inline-block;width:100%;}
.viz-img-wrap img{display:block;width:100%;}
.det-box{position:absolute;border:1px solid;z-index:10;}
.det-box:hover{z-index:100;border-width:2px;cursor:crosshair;}
.poly-svg{position:absolute;width:100%;height:100%;pointer-events:none;}
.tt{display:none;position:absolute;z-index:9999;background:var(--tt-bg);backdrop-filter:blur(8px);border:1px solid var(--tt-bd);border-radius:6px;padding:8px 10px;width:max-content;max-width:320px;box-shadow:0 4px 12px var(--tt-sh);pointer-events:none;flex-direction:column;gap:4px;align-items:flex-start;}
.det-box:hover>.tt{display:flex;}
.tt.tt-up{bottom:100%;top:auto;margin-bottom:6px;}
.tt.tt-down{top:100%;bottom:auto;margin-top:6px;}
.tt.tt-left{left:0;right:auto;}
.tt.tt-right{right:0;left:auto;}
.tt-label{font-size:11px;font-weight:800;text-transform:uppercase;color:var(--tp);}
.tt-conf{font-size:10px;color:var(--ts);}
.tt-text{font-size:11px;color:var(--tp);line-height:1.4;padding-top:4px;margin-top:2px;border-top:1px solid var(--border);width:100%;}
.viz-legend{padding:10px;display:flex;gap:10px;flex-wrap:wrap;background:var(--surface);border-top:1px solid var(--border);}
.legend-item{font-size:10px;color:var(--ts);display:flex;align-items:center;gap:4px;}
.legend-dot{width:8px;height:8px;border-radius:2px;}
"""


def build_html(
    image_rgb: np.ndarray,
    detections: list,
    mode: str,  # "ocr" | "layout"
    inference_ms: float = 0,
    theme: str = "dark",
) -> str:
    orig_h, orig_w = image_rgb.shape[:2]
    data_uri, _, _ = _img_to_data_uri(image_rgb)
    overlays = _build_overlays(detections, mode, orig_w, orig_h)
    legend = _build_legend(detections) if mode == "layout" else ""
    label = mode.replace("_", " ").title()

    return (
        f'<div class="ocr-root {theme.lower()}">'
        f"<style>{CSS}</style>"
        f'<div class="viz-header">'
        f'<div style="color:var(--tp);font-size:12px;font-weight:bold;">{label}</div>'
        f'<div style="color:var(--ts);font-size:11px;">{len(detections)} items • {inference_ms:.0f}ms</div>'
        f"</div>"
        f'<div class="viz-canvas"><div class="viz-img-wrap">'
        f'<img src="{data_uri}"/>'
        f"{overlays}"
        f"</div></div>"
        f"{legend}"
        f"</div>"
    )
