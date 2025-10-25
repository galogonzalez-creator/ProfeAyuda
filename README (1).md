
# Profe Ayuda v2 — Mistral + RAG + SDXL (ControlNet) + Manim

## Modelos locales requeridos
- `models/mistral_7b_instruct_v02/` → Mistral-7B-Instruct-v0.2 (safetensors + tokenizer + configs)
- `models/sdxl_base_1_0/` → sd_xl_base_1.0.safetensors + sd_xl_base_1.0_0.9vae.safetensors
- `models/controlnet_lineart_sdxl/` → ControlNet Lineart SDXL

## Instalación rápida (AMD ROCm)
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
pip install fastapi uvicorn[standard] pydantic[dotenv] python-multipart
pip install scikit-learn numpy scipy matplotlib
pip install transformers accelerate safetensors sentencepiece
pip install diffusers==0.30.3 controlnet-aux==0.0.7
pip install manim
```
## Ejecutar
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
