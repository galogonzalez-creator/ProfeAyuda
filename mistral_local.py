# mistral_local.py
import os
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


class MistralLocal:
    """
    Carga Mistral local/HF y genera continuación de chat.
    - Usa chat template si existe (apply_chat_template).
    - Decodifica SOLO la continuación (no el prompt).
    - Evita warnings: si do_sample=False, NO fija temperature/top_p.
    - Devuelve siempre texto no vacío (fallback).
    """

    def __init__(self, model_path: str = "./MIstral"):
        # --- Config por entorno ---
        self.model_id = os.environ.get("HF_MODEL_ID", model_path)
        self.max_input_tokens = int(os.environ.get("MISTRAL_MAX_INPUT_TOKENS", 512))
        self.max_new_tokens = int(os.environ.get("MISTRAL_MAX_NEW_TOKENS", 160))
        self.do_sample = os.environ.get("MISTRAL_DO_SAMPLE", "0") == "1"  # por defecto NO samplea
        self.temperature = float(os.environ.get("MISTRAL_TEMPERATURE", 0.2))
        self.top_p = float(os.environ.get("MISTRAL_TOP_P", 0.9))
        self.repetition_penalty = float(os.environ.get("MISTRAL_REP_PENALTY", 1.1))
        self.trim_words = int(os.environ.get("MISTRAL_TRIM_WORDS", 120))
        self.low_vram = os.environ.get("LOW_VRAM", "1") == "1"
        self.trust_remote_code = os.environ.get("TRUST_REMOTE_CODE", "1") == "1"

        print(f"[INFO] Inicializando Mistral: {self.model_id}")
        print(
            f"[INFO] max_input_tokens={self.max_input_tokens}, "
            f"max_new_tokens={self.max_new_tokens}, do_sample={self.do_sample}, "
            f"temp={self.temperature}, top_p={self.top_p}"
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            if torch.cuda.is_available():
                dtype = torch.float16
                device_map = "auto"
            else:
                dtype = "auto"
                device_map = "auto" if self.low_vram else None

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
                trust_remote_code=self.trust_remote_code,
            )
            self.model.eval()
            torch.set_grad_enabled(False)

            # Si do_sample=False, no seteamos temperature/top_p para evitar warnings
            if self.do_sample:
                self.gen_config = GenerationConfig(
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            else:
                self.gen_config = GenerationConfig(
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=self.repetition_penalty,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            print(
                f"[OK] Modelo cargado en "
                f"{'GPU (ROCm/CUDA)' if torch.cuda.is_available() else 'CPU'} "
                f"con dtype={dtype}."
            )
        except Exception as e:
            print(f"[ERROR] No se pudo cargar el modelo: {e}")
            self.model = None
            self.tokenizer = None

    # ---------- helpers ----------
    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return ""

    def _fallback_prompt(self, messages: List[Dict[str, str]]) -> str:
        lines = []
        for m in messages:
            role = m.get("role", "user")
            txt = (m.get("content") or "").strip()
            if role == "system":
                lines.append(f"Instrucciones: {txt}")
            elif role == "assistant":
                lines.append(f"Profesor: {txt}")
            else:
                lines.append(f"Estudiante: {txt}")
        # abrimos el turno del asistente
        lines.append("Profesor:")
        return "\n".join(lines)

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        prompt = self._apply_chat_template(messages)
        if not prompt:
            prompt = self._fallback_prompt(messages)
        return prompt

    # ---------- chat ----------
    def chat(self, messages: List[Dict[str, str]]) -> str:
        if self.model is None or self.tokenizer is None:
            return "⚠️ El modelo no está disponible. Revisa HF_MODEL_ID o la ruta local."

        prompt = self._build_prompt(messages)
        print("=== [DEBUG] PROMPT ===")
        print(prompt[:800])

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_input_tokens
        )

        # Mover tensores al dispositivo del modelo (si aplica)
        if getattr(self.model, "device", None) is not None and self.model.device.type != "meta":
            try:
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            except Exception:
                pass

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, generation_config=self.gen_config)

        # Decodificar SOLO la continuación (sin el prompt)
        gen_ids = outputs[0][input_len:]
        if gen_ids.numel() == 0:
            print("[WARN] Generación vacía (gen_ids=0).")
            return (
                "Puedo ayudarte paso a paso, pero necesito un poco más de contexto. "
                "¿Puedes repetir la pregunta?"
            )

        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        print("=== [DEBUG] GENERATION ===")
        print(text)

        if not text:
            text = (
                "Puedo ayudarte paso a paso, pero necesito un poco más de contexto. "
                "¿Puedes repetir la pregunta?"
            )

        if self.trim_words > 0:
            words = text.split()
            if len(words) > self.trim_words:
                text = " ".join(words[:self.trim_words]) + "…"

        return text

