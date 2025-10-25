# image_server.py
import os
import time
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# ======================
# Config (rutas y flags)
# ======================
SDXL_PATH = os.environ.get("MODEL_SDXL_PATH", "./MIstral/sdxl")
OUT_DIR   = os.environ.get("IMG_OUT_DIR", "./config")
USE_CONTROLNET = os.environ.get("USE_CONTROLNET", "0") == "1"  # lo dejamos por si luego quieres
LOW_VRAM  = os.environ.get("LOW_VRAM", "1") == "1"

os.makedirs(OUT_DIR, exist_ok=True)

app = FastAPI(title="Image Server (SDXL)")
app.mount("/generated", StaticFiles(directory=OUT_DIR), name="generated")

# Estado global
pipe = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

class GenReq(BaseModel):
    prompt: str
    width: int = 512
    height: int = 512
    steps: int = 12
    guidance: float = 3.5
    seed: Optional[int] = None

def _load_sdxl() -> StableDiffusionXLPipeline:
    """
    Carga un SDXL base ligero (sin ControlNet) con ajustes de VRAM para ROCm.
    """
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"[INFO] Cargando SDXL desde: {SDXL_PATH} | device={_device} | dtype={dtype}")

    p = StableDiffusionXLPipeline.from_pretrained(
        SDXL_PATH,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    p.scheduler = DPMSolverMultistepScheduler.from_config(p.scheduler.config)
    p.to(_device)

    if LOW_VRAM:
        # Ahorros de VRAM
        p.enable_attention_slicing()
        p.enable_vae_slicing()
        p.enable_sequential_cpu_offload()
        print("[INFO] LOW_VRAM=1 -> attention/vae slicing + cpu_offload")

    return p


@app.on_event("startup")
def _startup():
    global pipe
    # Evita doble carga al recargar con --reload
    if pipe is None:
        pipe = _load_sdxl()
    print("[OK] Image server listo.")


@app.get("/health")
def health():
    return {
        "ok": True,
        "sdxl_path": SDXL_PATH,
        "device": _device,
        "low_vram": LOW_VRAM
    }


@app.post("/generate")
def generate(req: GenReq):
    """
    Genera una imagen con SDXL base (estilo educativo/lineart).
    Devuelve URL estática para consumir desde el frontend del chatbot.
    """
    global pipe
    if pipe is None:
        return JSONResponse({"ok": False, "error": "pipeline_not_loaded"}, status_code=500)

    full_prompt = (
        f"{req.prompt}, fondo blanco, lineart limpio, estilo educativo, alta claridad, "
        "sin texto, sin sombras, tipo libro escolar"
    )

    generator = None
    if req.seed is not None:
        generator = torch.Generator(device=_device).manual_seed(req.seed)

    try:
        print(f"[INFO] Generando: '{full_prompt}' {req.width}x{req.height} steps={req.steps} g={req.guidance}")
        result = pipe(
            prompt=full_prompt,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance,
            width=req.width,
            height=req.height,
            generator=generator
        )
        image = result.images[0]
        out_path = os.path.join(OUT_DIR, f"diagram_{int(time.time())}.png")
        image.save(out_path)

        url = f"/generated/{os.path.basename(out_path)}"
        print(f"[OK] Imagen: {out_path}")
        return {"ok": True, "image_url": url, "path": out_path}
    except torch.cuda.OutOfMemoryError:
        return JSONResponse({"ok": False, "error": "hip_oom"}, status_code=500)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

