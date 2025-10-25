# sdxl_controlnet.py
import os
import time
from typing import Optional, Union, Tuple

import torch
from PIL import Image

from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
    DPMSolverMultistepScheduler,
)


def _use_cuda() -> bool:
    # En ROCm, torch reporta "cuda" también
    return torch.cuda.is_available()


def _nearest_multiple_of_8(x: int, lo: int = 256, hi: int = 1024) -> int:
    """SDXL exige múltiplos de 8; además limitamos para ahorrar VRAM."""
    x = max(lo, min(hi, int(x)))
    return (x // 8) * 8


class SDXLControlNet:
    """
    Generador de imágenes educativas con SDXL.
    - ControlNet es OPCIONAL. Si no hay imagen de control, genera con SDXL base.
    - Ajustado para ROCm (AMD) y bajo VRAM.
    - Sin 'variant=fp16' para evitar errores cuando ese branch no existe.

    ENV útiles:
      MODEL_SDXL_PATH              -> ruta al SDXL base (por defecto ./MIstral/sdxl)
      CONTROLNET_LINEART_PATH      -> ruta al modelo de ControlNet (opcional)
      USE_CONTROLNET=0/1           -> habilita ControlNet si existe (default 1)
      LOW_VRAM=0/1                 -> activa slicing/offload (default 1)
      PYTORCH_HIP_ALLOC_CONF       -> ej.: "garbage_collection_threshold:0.8,max_split_size_mb:128"
    """

    def __init__(
        self,
        sdxl_path: str = "./MIstral/sdxl",
        controlnet_path: Optional[str] = "./ControlNet",
        out_dir: str = "./config",
        low_vram: Optional[bool] = None,
        use_controlnet: Optional[bool] = None,
    ):
        self.sdxl_path = sdxl_path
        self.controlnet_path = controlnet_path or ""
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.device = "cuda" if _use_cuda() else "cpu"
        self.dtype = torch.float16 if _use_cuda() else torch.float32

        env_low_vram = os.environ.get("LOW_VRAM", "1") == "1"
        env_use_cn = os.environ.get("USE_CONTROLNET", "1") == "1"
        self.low_vram = env_low_vram if low_vram is None else bool(low_vram)
        self.want_controlnet = env_use_cn if use_controlnet is None else bool(use_controlnet)

        self.pipe_base: Optional[StableDiffusionXLPipeline] = None
        self.pipe_cn: Optional[StableDiffusionXLControlNetPipeline] = None
        self.controlnet: Optional[ControlNetModel] = None

        print(f"[INFO] Dispositivo: {self.device} | dtype={self.dtype}")
        print(f"[INFO] SDXL path: {os.path.abspath(self.sdxl_path)}")
        print(f"[INFO] ControlNet path: {os.path.abspath(self.controlnet_path) if self.controlnet_path else '(no configurado)'} | USE_CONTROLNET={self.want_controlnet}")
        print(f"[INFO] LOW_VRAM={self.low_vram}")

        self._load_pipelines()

    # --------------------------
    # Carga de modelos
    # --------------------------
    def _assert_path(self, path: str, label: str) -> None:
        if not path or not os.path.isdir(path):
            raise FileNotFoundError(f"{label} no encontrado: {path}")

    def _load_base(self):
        print("[INFO] Cargando SDXL base...")
        self._assert_path(self.sdxl_path, "SDXL")
        self.pipe_base = StableDiffusionXLPipeline.from_pretrained(
            self.sdxl_path,
            torch_dtype=self.dtype,
            use_safetensors=True,
        )
        self.pipe_base.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe_base.scheduler.config)

        # VAE local (opcional)
        try:
            vae_local = os.path.join(self.sdxl_path, "vae")
            if os.path.isdir(vae_local):
                self.pipe_base.vae = AutoencoderKL.from_pretrained(vae_local, torch_dtype=self.dtype)
        except Exception as e:
            print(f"[WARN] VAE opcional no cargó: {e}")

        self.pipe_base.to(self.device)
        if self.low_vram:
            try:
                self.pipe_base.enable_attention_slicing()
                self.pipe_base.enable_vae_slicing()
                # Si tienes accelerate: usa model offload (ahorra VRAM)
                self.pipe_base.enable_model_cpu_offload()
            except Exception as e:
                print(f"[WARN] Offload/slicing no disponible: {e}")

        print("[OK] SDXL base lista.")

    def _can_load_controlnet(self) -> bool:
        if not self.want_controlnet:
            return False
        if not self.controlnet_path:
            return False
        cfg = os.path.join(self.controlnet_path, "config.json")
        return os.path.isfile(cfg)

    def _load_controlnet(self):
        if not self.want_controlnet:
            print("[INFO] USE_CONTROLNET=0 -> no se cargará ControlNet.")
            return
        if not self._can_load_controlnet():
            print("[WARN] ControlNet no encontrado/completo. Continuando sin ControlNet.")
            return

        print("[INFO] Cargando ControlNet…")
        self.controlnet = ControlNetModel.from_pretrained(
            self.controlnet_path,
            torch_dtype=self.dtype,
            use_safetensors=True,
        )

        print("[INFO] Construyendo pipeline SDXL+ControlNet…")
        self.pipe_cn = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.sdxl_path,
            controlnet=self.controlnet,
            torch_dtype=self.dtype,
            use_safetensors=True,
        )
        self.pipe_cn.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe_cn.scheduler.config)

        try:
            vae_local = os.path.join(self.sdxl_path, "vae")
            if os.path.isdir(vae_local):
                self.pipe_cn.vae = AutoencoderKL.from_pretrained(vae_local, torch_dtype=self.dtype)
        except Exception as e:
            print(f"[WARN] VAE opcional (CN) no cargó: {e}")

        self.pipe_cn.to(self.device)
        if self.low_vram:
            try:
                self.pipe_cn.enable_attention_slicing()
                self.pipe_cn.enable_vae_slicing()
                self.pipe_cn.enable_model_cpu_offload()
            except Exception as e:
                print(f"[WARN] Offload/slicing (CN) no disponible: {e}")

        print("[OK] SDXL + ControlNet listos.")

    def _load_pipelines(self):
        try:
            self._load_base()
        except Exception as e:
            print(f"[ERROR] No se pudo cargar SDXL base: {e}")
            self.pipe_base = None

        try:
            self._load_controlnet()
        except Exception as e:
            print(f"[WARN] No se pudo inicializar ControlNet. Motivo: {e}")
            self.pipe_cn = None
            self.controlnet = None

    # --------------------------
    # Salud / estado
    # --------------------------
    def health(self) -> dict:
        return {
            "device": self.device,
            "dtype": str(self.dtype),
            "base_loaded": self.pipe_base is not None,
            "controlnet_loaded": self.pipe_cn is not None and self.controlnet is not None,
            "paths": {
                "sdxl_path": self.sdxl_path,
                "controlnet_path": self.controlnet_path,
                "out_dir": self.out_dir,
            },
        }

    # --------------------------
    # Generación
    # --------------------------
    def _compose_prompt(self, prompt: str) -> str:
        # Prompt guía para diagramas limpios estilo libro escolar
        extras = (
            ", estilo educativo, líneas limpias, fondo blanco, dibujo sencillo, "
            "sin texto extra, alto contraste, tipo libro escolar"
        )
        return (prompt or "").strip() + extras

    def _sanitize_wh(self, width: int, height: int) -> Tuple[int, int]:
        return _nearest_multiple_of_8(width), _nearest_multiple_of_8(height)

    def generate_lineart(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        guidance: float = 4.5,
        steps: int = 20,
        control_image: Optional[Union[Image.Image, "np.ndarray"]] = None,
        seed: Optional[int] = None,
    ) -> Optional[str]:
        """
        Genera una imagen.

        - Si se provee control_image y hay ControlNet cargado, usa SDXL+ControlNet.
        - Si NO se provee control_image, usa SDXL base (aunque ControlNet esté cargado).
        - Devuelve la ruta del PNG generado o None si falló.
        """
        if self.pipe_base is None:
            print("[ERROR] Pipeline base no disponible.")
            return None

        width, height = self._sanitize_wh(width, height)
        full_prompt = self._compose_prompt(prompt)

        generator = None
        if seed is not None:
            try:
                generator = torch.Generator(device=self.device).manual_seed(int(seed))
            except Exception:
                generator = None

        try:
            if control_image is not None and self.pipe_cn is not None:
                print("[INFO] Generando con SDXL + ControlNet…")
                result = self.pipe_cn(
                    prompt=full_prompt,
                    image=control_image,  # <-- CLAVE: no llamar CN sin imagen
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance),
                    width=int(width),
                    height=int(height),
                    generator=generator,
                )
            else:
                if control_image is not None and self.pipe_cn is None:
                    print("[WARN] Se pasó control_image pero ControlNet no está disponible. Usando SDXL base.")
                print(f"[INFO] Generando con SDXL base… {width}x{height}, steps={steps}, g={guidance}")
                result = self.pipe_base(
                    prompt=full_prompt,
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance),
                    width=int(width),
                    height=int(height),
                    generator=generator,
                )

            img = result.images[0] if hasattr(result, "images") and result.images else None
            if img is None or not isinstance(img, Image.Image):
                print("[ERROR] La pipeline no devolvió una PIL.Image válida.")
                return None

            out_path = os.path.join(self.out_dir, f"diagram_{int(time.time())}.png")
            img.save(out_path)
            print(f"[OK] Imagen guardada en: {out_path}")
            return out_path

        except torch.cuda.OutOfMemoryError:
            print("[ERROR] Sin VRAM suficiente. Reduce resolución/pasos o activa LOW_VRAM=1.")
            return None
        except Exception as e:
            print(f"[ERROR] Falló la generación: {e}")
            return None

