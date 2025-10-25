# main.py
import os
import re
import math
from typing import List, Optional, Literal, Tuple, Dict

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

# ====== Módulos locales ======
from mistral_local import MistralLocal
from tfidf_store import RAGStore
from sdxl_controlnet import SDXLControlNet
from scene_manim import render_figure  # triángulo/rectángulo/círculo/rombo (si lo agregas)

# ==========================================
# CONFIG
# ==========================================
BASE_DIR    = os.path.abspath(os.path.dirname(__file__))
CFG_DIR     = os.path.join(BASE_DIR, "config")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
TEMPLATES   = os.path.join(BASE_DIR, "templates")

MODEL_MISTRAL_PATH = os.environ.get("MODEL_MISTRAL_PATH", "./Mistral")
MODEL_SDXL_PATH    = os.environ.get("MODEL_SDXL_PATH",    "./Mistral/sdxl")
CONTROLNET_PATH    = os.environ.get("CONTROLNET_LINEART_PATH", "./ControlNet")

USE_CONTROLNET = os.environ.get("USE_CONTROLNET", "0") == "1"
LOW_VRAM       = os.environ.get("LOW_VRAM", "1") == "1"

# Índices que consideraremos "válidos" por defecto (4to/5to EGB) si existen
PREFERRED_INDEX_HINTS = [
    "4EGB", "4to", "cuarto",
    "5EGB", "5to", "quinto",
    "profe_ayuda_5to",
    "quinto_EGB", "unidades", "lotes"
]

# ==========================================
# FASTAPI + CORS
# ==========================================
app = FastAPI(title="Profe Ayuda – Chatbot Matemáticas")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

os.makedirs(CFG_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Servimos imágenes generadas desde config/
app.mount("/generated", StaticFiles(directory=CFG_DIR), name="generated")
templates = Jinja2Templates(directory=TEMPLATES)

# ==========================================
# RAG + LLM Lazy
# ==========================================
rag = RAGStore(base_dir=CFG_DIR)
_llm = None
_img = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = MistralLocal(model_path=MODEL_MISTRAL_PATH)
    return _llm

def get_img():
    global _img
    if _img is None:
        _img = SDXLControlNet(
            sdxl_path=MODEL_SDXL_PATH,
            controlnet_path=CONTROLNET_PATH if USE_CONTROLNET else None,
            out_dir=CFG_DIR,
            low_vram=LOW_VRAM
        )
    return _img

# ==========================================
# AUTO-INDEX EN STARTUP
# ==========================================
AUTO_INDEX_FILE = os.environ.get("AUTO_INDEX_FILE", "")

@app.on_event("startup")
def _auto_index_startup():
    """
    Si no hay índices, intenta construir 1 desde uploads/AUTO_INDEX_FILE (opcional).
    Si sí hay índices, imprime lista (útil para ver 4EGB/5EGB).
    """
    try:
        existing = rag.list_indexes()
        if not existing and AUTO_INDEX_FILE:
            path = AUTO_INDEX_FILE
            if not os.path.isabs(path):
                path = os.path.join(UPLOADS_DIR, path)
            if os.path.exists(path):
                name = rag.build(path, text_cols=["instruction", "input", "output"], name="auto_index")
                print(f"[AUTO-INDEX] Índice creado: {name}")
            else:
                print(f"[AUTO-INDEX] Archivo no encontrado: {path}")
        else:
            print(f"[AUTO-INDEX] Índices ya disponibles: {existing}")
    except Exception as e:
        print(f"[AUTO-INDEX] Error al cargar índice: {e}")

# ==========================================
# SCHEMAS
# ==========================================
class IndexReq(BaseModel):
    jsonl_path: str
    text_cols: Optional[List[str]] = None
    name: Optional[str] = None

class ChatReq(BaseModel):
    query: str
    top_k: int = 5
    draw_example: bool = False   # si true => intenta imagen (solo una vez)

class DiagramReq(BaseModel):
    prompt: str
    width: int = 768
    height: int = 768
    guidance: float = 3.5
    steps: int = 16
    force: Optional[Literal["manim", "sdxl"]] = None

class CheckReq(BaseModel):
    question: str
    student_answer: str

# ==========================================
# UTILIDADES (detección y formato)
# ==========================================
GEOM_PATTERNS = {
    "triangle":  re.compile(r"\btri[aá]ngulo\b", re.I),
    "rectangle": re.compile(r"\brect[aá]ngulo\b", re.I),
    "circle":    re.compile(r"\bc[ií]rculo\b|\bradio\b", re.I),
    "rhombus":   re.compile(r"\brombo\b", re.I),
}

META_STOP = re.compile(r"(sigue estos pasos:$|formato:$)", re.I)
NUM_RE    = r"([0-9]+(?:[.,][0-9]+)?)"
BASE_RE   = re.compile(rf"(?:base|lado)\s*{NUM_RE}", re.I)
ALT_RE    = re.compile(rf"(?:altura|alto|h)\s*{NUM_RE}", re.I)
RAD_RE    = re.compile(rf"(?:radio|r)\s*{NUM_RE}", re.I)

def detect_geometry(query: str) -> Optional[str]:
    for label, pat in GEOM_PATTERNS.items():
        if pat.search(query or ""):
            return label
    return None

def _split_sentences(txt: str) -> List[str]:
    parts = re.split(r'(?<=[\.\?\!])\s+|\n+', txt or "")
    return [p.strip() for p in parts if p and p.strip()]

def format_as_numbered_steps(answer: str, max_steps: int = 5, min_steps: int = 3) -> str:
    sents = _split_sentences(answer)
    sents = [s for s in sents if not META_STOP.search(s)]
    bullet_re = re.compile(r'^\s*(?:\d+|[•\-–\*])\s*[.\)\-]?\s*')
    cleaned = []
    for s in sents:
        s2 = bullet_re.sub('', s).strip()
        if s2:
            cleaned.append(s2)
    if len(cleaned) < min_steps:
        fillers = [
            "Sustituye los valores numéricos en la operación.",
            "Escribe con claridad cada transformación." ,
            "Verifica las unidades si aplican."
        ]
        for f in fillers:
            if len(cleaned) >= min_steps:
                break
            cleaned.append(f)
    cleaned = cleaned[:max_steps]
    if not any(x.endswith("?") for x in cleaned[-2:]):
        cleaned.append("¿Cuánto te da tu resultado?. ✍️")
    cleaned = cleaned[:max_steps]
    return "\n".join(f"{i+1}. {s}" for i, s in enumerate(cleaned))

# ---- utilidades numéricas ----
def _to_float(s: str) -> float:
    return float(s.replace(",", "."))

def _safe_eval_arith(expr: str) -> Optional[float]:
    if not expr:
        return None
    e = expr.strip().replace(",", ".")
    if not re.fullmatch(r"[0-9\.\s\+\-\*\/\(\)]+", e):
        return None
    try:
        val = eval(e, {"__builtins__": {}}, {})
        if isinstance(val, (int, float)) and math.isfinite(val):
            return float(val)
        return None
    except Exception:
        return None

def compute_expected(query: str) -> Tuple[Optional[float], Optional[str]]:
    kind = detect_geometry(query or "")
    if kind:
        q = (query or "").lower()
        if kind in ("triangle", "rectangle"):
            b = None; h = None
            m = BASE_RE.search(q)
            if m: b = _to_float(m.group(1))
            m = ALT_RE.search(q)
            if m: h = _to_float(m.group(1))
            if (b is None or h is None):
                nums = [_to_float(x) for x in re.findall(NUM_RE, q)]
                if len(nums) >= 2:
                    b = nums[0] if b is None else b
                    h = nums[1] if h is None else h
            if b is not None and h is not None:
                area = b * h if kind == "rectangle" else (b * h) / 2.0
                return area, kind
            return None, kind
        if kind == "circle":
            r = None
            m = RAD_RE.search(q)
            if m: r = _to_float(m.group(1))
            else:
                nums = [_to_float(x) for x in re.findall(NUM_RE, q)]
                if nums: r = nums[0]
            if r is not None:
                return math.pi * r * r, kind
            return None, kind
        if kind == "rhombus":
            return None, kind
    # No geometría: intenta operación aritmética
    val = _safe_eval_arith(query)
    if val is not None:
        return val, None
    return None, None

def pick_preferred_indexes(all_indexes: List[str]) -> List[str]:
    if not all_indexes:
        return []
    # Prioriza índices que contengan pistas de 4to/5to EGB
    preferred = [ix for ix in all_indexes if any(h.lower() in ix.lower() for h in PREFERRED_INDEX_HINTS)]
    return preferred or all_indexes

def retrieve_from_all(query: str, top_k: int = 5) -> List[Dict]:
    """
    Recupera de múltiples índices si el store lo soporta (aquí llamamos al RAGStore
    sin nombre de índice; asumimos recuperación global interna). Si tu RAGStore
    requiere nombre, podrías iterar por índice y concatenar.
    """
    try:
        return rag.retrieve(query, top_k=top_k)
    except Exception:
        return []

# ==========================================
# RUTAS
# ==========================================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health():
    indices = rag.list_indexes()
    return {
        "ok": True,
        "indexes": indices,
        "preferred": pick_preferred_indexes(indices),
        "mistral_path": os.path.abspath(MODEL_MISTRAL_PATH),
        "sdxl_path": os.path.abspath(MODEL_SDXL_PATH),
        "controlnet_path": os.path.abspath(CONTROLNET_PATH),
        "use_controlnet": USE_CONTROLNET,
        "low_vram": LOW_VRAM,
    }

@app.post("/index")
def index_corpus(req: IndexReq):
    path = req.jsonl_path
    if not os.path.isabs(path):
        path = os.path.join(UPLOADS_DIR, path)
    name = rag.build(path, text_cols=req.text_cols or ["instruction", "input", "output"], name=req.name)
    return {"ok": True, "index": name, "available": rag.list_indexes()}

@app.post("/chat")
def chat(req: ChatReq):
    # 1) Recupera contexto desde índices (4to/5to EGB preferidos)
    all_ix = rag.list_indexes()
    _ = pick_preferred_indexes(all_ix)  # Nota: si tu RAGStore admite por índice, itera aquí.
    ctx = retrieve_from_all(req.query, top_k=req.top_k)  # combinado

    context_block = "\n".join([f"[DOC {i+1}] {d.get('text','')[:220]}" for i, d in enumerate(ctx)]) or "Sin contexto relevante."

    # 2) Prompt (ESPAÑOL, sin dar resultado final)
    system_prompt = (
    	"Eres Profe Ayuda, una maestra que enseña Matemáticas a niños de 7 a 8 años. "
    	"Explica en español, con frases muy simples y amigables. "
    	"Usa solo ejemplos del nivel de primaria baja (sumas, restas, multiplicaciones, divisiones, figuras). "
    	"Guía al niño paso a paso, con frases como 'primero', 'luego', 'ahora tú intenta'. "
    	"No des palabras difíciles como 'residuo' o 'cociente'. "
    	"Si se puede resolver con pocos pasos, hazlo con 3 o 4. "
    	"Termina siempre preguntando: '¿Cuánto te da tu resultado?'"
    )

    user_msg = f"Contexto (fragmentos de libros 4.º/5.º EGB):\n{context_block}\n\nPregunta del estudiante: {req.query}"

    try:
        llm = get_llm()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",    "content": user_msg}
        ]
        answer_raw = llm.chat(messages)
        answer = format_as_numbered_steps(answer_raw, max_steps=5, min_steps=3)
    except Exception as e:
        return {"ok": False, "answer": f"⚠️ Error al generar respuesta: {e}", "context": ctx, "image_url": None}

    # 3) Cálculo esperado (para validar en la UI hasta que acierte)
    expect, kind = compute_expected(req.query)

    # 4) Imagen (si y solo si el usuario pulsó 'Dibujar ejemplo'); una vez.
    image_url = None
    if req.draw_example:
        # Manim primero si reconoce figura
        if kind in {"triangle", "rectangle", "circle", "rhombus"}:
            try:
                path = render_figure(req.query, out_dir=CFG_DIR)
                if path:
                    image_url = f"/generated/{os.path.basename(path)}"
            except Exception:
                image_url = None

        # SDXL como respaldo
        if image_url is None:
            try:
                img = get_img()
                auto_prompt = (
                    f"Diagrama geométrico limpio para: {req.query}. "
                    "Fondo blanco, líneas negras, estilo libro escolar, sin texto."
                )
                path = img.generate_lineart(prompt=auto_prompt)
                if path:
                    image_url = f"/generated/{os.path.basename(path)}"
            except Exception:
                image_url = None

    return {
        "ok": True,
        "answer": answer,
        "context": ctx,
        "image_url": image_url,
        "expect": expect,   # float o null (para validar en frontend)
        "kind": kind        # tipo de figura detectada o null
    }

@app.post("/api/check")
def api_check(req: CheckReq):
    """
    Valida la respuesta del estudiante. Si el problema es aritmético o de área
    con datos suficientes, compara numéricamente con tolerancia. Si no, devuelve
    explicación genérica (para que Profe Ayuda continúe guiando).
    """
    try:
        gold, _ = compute_expected(req.question)
    except Exception:
        gold = None

    if gold is None:
        # No hay validación automática posible
        return {"ok": False, "explanation": "No puedo validar automáticamente este ejercicio. Comparémoslo paso a paso."}

    # Normaliza y evalúa la respuesta del estudiante
    pred = _safe_eval_arith(req.student_answer)
    if pred is None:
        return {"ok": False, "explanation": "Tu respuesta no parece numérica válida. Revisa formato y prioridad de operaciones."}

    tol = 1e-4
    if abs(float(pred) - float(gold)) <= tol:
        return {"ok": True, "explanation": "✅ ¡Correcto! Coincide con el resultado esperado."}
    else:
        return {"ok": False, "explanation": "❌ Aún no coincide. Revisa divisiones/multiplicaciones y el orden de operaciones."}

@app.post("/image/diagram")
def image_diagram(req: DiagramReq):
    # Forzar Manim si lo pide
    if req.force == "manim":
        try:
            path = render_figure(req.prompt, out_dir=CFG_DIR)
            if path:
                return {"ok": True, "image_url": f"/generated/{os.path.basename(path)}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    kind = detect_geometry(req.prompt)
    if kind in {"triangle", "rectangle", "circle", "rhombus"}:
        try:
            path = render_figure(req.prompt, out_dir=CFG_DIR)
            if path:
                return {"ok": True, "image_url": f"/generated/{os.path.basename(path)}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # SDXL fallback
    try:
        img = get_img()
        prompt = (
            f"{req.prompt}. Diagrama educativo de líneas, fondo blanco, limpio, sin texto."
        )
        path = img.generate_lineart(
            prompt=prompt,
            width=req.width,
            height=req.height,
            guidance=req.guidance,
            steps=req.steps,
        )
        if not path:
            return {"ok": False, "error": "no_image"}
        return {"ok": True, "image_url": f"/generated/{os.path.basename(path)}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
        
if __name__ == "__main__":
    import os

    # Estos flags desactivan SDXL o controlnet si los usas
    os.environ["USE_SDXL"] = "0"
    os.environ["USE_CONTROLNET"] = "0"
    os.environ["LOW_VRAM"] = "1"

    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

