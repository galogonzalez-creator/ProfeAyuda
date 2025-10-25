# scene_manim.py
from __future__ import annotations
import os
import time
from typing import Optional

from manim import (
    Scene, VGroup, Line, Polygon, Circle, MathTex, Text, Dot,
    config, tempconfig, WHITE, BLACK, GREY_B, BLUE, ORIGIN, RIGHT, UP, DOWN, LEFT
)
from PIL import Image
import numpy as np

# =========================
# Ajustes globales "safe"
# =========================
# Fuerza renderer sin GPU/ventana (headless)
os.environ.setdefault("MANIMCE_DISABLE_CUDA", "1")
config.renderer = "cairo"
config.disable_caching = True
config.write_to_movie = False
config.frame_rate = 30
config.background_color = WHITE

# -------------------------
# Utilidad para salvar PNG
# -------------------------
def _save_scene_png(scene: Scene, path_png: str) -> None:
    """
    Renderiza la escena y guarda el último frame como PNG (fondo blanco).
    Funciona con renderer 'cairo' (headless). Incluye fallback de frame.
    """
    scene.render()
    frame = None

    # Manim >=0.18: renderer.get_frame()
    try:
        frame = scene.renderer.get_frame()
    except Exception:
        frame = None

    # Fallback: tomar imagen de la cámara
    if frame is None:
        try:
            # En cairo la cámara expone el frame como array RGBA
            frame = scene.camera.get_image()
            if hasattr(frame, "shape"):
                frame = np.array(frame)
        except Exception:
            frame = None

    if frame is None:
        raise RuntimeError("No se pudo obtener el frame renderizado de Manim.")

    Image.fromarray(frame).save(path_png)

# -------------------------
# Detección simple por texto
# -------------------------
def _infer_kind_from_text(text: str) -> Optional[str]:
    t = (text or "").lower()
    if any(k in t for k in ["triangulo", "triángulo", "equilatero", "equilátero", "isosceles", "isósceles"]):
        return "triangle"
    if any(k in t for k in ["rectangulo", "rectángulo", "cuadrilatero", "cuadrilátero", "rectángulo"]):
        return "rectangle"
    if any(k in t for k in ["circulo", "círculo", "circle", "circunferencia", "radio"]):
        return "circle"
    return None

# =========================
# Escenas con geometría
# =========================
def _safe_mathtex(latex: str, color=BLUE, scale_val=1.0) -> VGroup:
    """
    Intenta crear MathTex; si no hay LaTeX instalado, usa Text como respaldo.
    Devuelve el mobject listo para agregar a la escena.
    """
    try:
        obj = MathTex(latex, color=color).scale(scale_val)
    except Exception:
        obj = Text(latex.replace("\\", ""), color=color).scale(scale_val)
    return obj

class TriangleArea(Scene):
    """Triángulo con base b y altura h; fórmula A = b·h / 2"""
    def construct(self):
        self.camera.background_color = WHITE

        base_len = 6.0
        height = 3.5

        A = UP * height/2
        B = LEFT * (base_len/2)
        C = RIGHT * (base_len/2)

        tri = Polygon(A, B, C, stroke_color=BLACK, stroke_width=6, fill_opacity=0)
        base = Line(B, C, stroke_color=BLACK, stroke_width=6)

        # Altura (desde A a la mitad de la base)
        foot = (B + C) / 2
        alt = Line(A, foot, stroke_color=GREY_B, stroke_width=5)

        label_base = Text("b", font_size=36, color=BLACK).next_to(base, DOWN, buff=0.2)
        label_h = Text("h", font_size=36, color=BLACK).next_to(alt, RIGHT, buff=0.2)

        formula = _safe_mathtex(r"A = \frac{b \cdot h}{2}", color=BLUE, scale_val=1.0)
        formula.to_edge(UP, buff=0.6)

        grp = VGroup(tri, base, alt, label_base, label_h, formula).move_to(ORIGIN)
        self.add(grp)

class RectangleArea(Scene):
    """Rectángulo con base b y altura h; fórmula A = b·h"""
    def construct(self):
        self.camera.background_color = WHITE

        w = 6.0  # base
        h = 3.5  # altura

        p1 = LEFT * (w/2) + DOWN * (h/2)
        p2 = RIGHT * (w/2) + DOWN * (h/2)
        p3 = RIGHT * (w/2) + UP * (h/2)
        p4 = LEFT * (w/2) + UP * (h/2)
        rect = Polygon(p1, p2, p3, p4, stroke_color=BLACK, stroke_width=6, fill_opacity=0)

        base_line = Line(p1, p2, stroke_color=BLACK, stroke_width=6)
        height_line = Line(p2, p3, stroke_color=BLACK, stroke_width=6)
        label_b = Text("b", font_size=36, color=BLACK).next_to(base_line, DOWN, buff=0.2)
        label_h = Text("h", font_size=36, color=BLACK).next_to(height_line, RIGHT, buff=0.2)

        formula = _safe_mathtex(r"A = b \cdot h", color=BLUE, scale_val=1.1)
        formula.to_edge(UP, buff=0.6)

        grp = VGroup(rect, base_line, height_line, label_b, label_h, formula).move_to(ORIGIN)
        self.add(grp)

class CircleArea(Scene):
    """Círculo con radio r; fórmula A = π·r²"""
    def construct(self):
        self.camera.background_color = WHITE

        r = 2.5
        circ = Circle(radius=r, stroke_color=BLACK, stroke_width=6)
        center = Dot(ORIGIN, color=BLACK, radius=0.06)

        radius_line = Line(ORIGIN, RIGHT * r, stroke_color=GREY_B, stroke_width=5)
        label_r = Text("r", font_size=36, color=BLACK).next_to(radius_line, UP, buff=0.2)

        formula = _safe_mathtex(r"A = \pi \cdot r^2", color=BLUE, scale_val=1.1)
        formula.to_edge(UP, buff=0.6)

        grp = VGroup(circ, center, radius_line, label_r, formula).move_to(ORIGIN)
        self.add(grp)

# =========================
# API pública
# =========================
def render_figure(
    prompt: str,
    out_dir: str = "./config",
    kind: Optional[str] = None,
) -> Optional[str]:
    """
    Renderiza una figura geométrica como PNG (fondo blanco) en `out_dir`.

    - `prompt`: texto del usuario (sirve para inferir la figura).
    - `kind`: si lo pasas, fuerza el tipo: "triangle" | "rectangle" | "circle".
    - Devuelve la ruta del PNG o None si no se pudo.
    """
    os.makedirs(out_dir, exist_ok=True)

    k = kind or _infer_kind_from_text(prompt)
    if k not in {"triangle", "rectangle", "circle"}:
        # Si no se reconoce, dejamos que SDXL se encargue en el backend
        return None

    # Nombre único para evitar caché del navegador
    out_path = os.path.join(out_dir, f"geometry_{k}_{int(time.time())}.png")

    # Parametrización local y aislada por render
    cfg = {
        "pixel_width": 768,
        "pixel_height": 768,
        "background_color": WHITE,
        "renderer": "cairo",
        "disable_caching": True,
        "write_to_movie": False,
        "media_dir": "media",   # evita ensuciar carpetas
        "frame_rate": 30,
    }

    if k == "triangle":
        scene = TriangleArea()
    elif k == "rectangle":
        scene = RectangleArea()
    else:  # "circle"
        scene = CircleArea()

    try:
        with tempconfig(cfg):
            _save_scene_png(scene, out_path)
        return out_path
    except Exception as e:
        print(f"[scene_manim] Error renderizando {k}: {e}")
        return None

# ---- Alias retrocompatibilidad ----
def render_triangle(out_dir: str = "./config") -> Optional[str]:
    return render_figure("triángulo", out_dir=out_dir, kind="triangle")

