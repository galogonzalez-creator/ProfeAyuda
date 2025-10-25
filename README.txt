PROFE AYUDA — CHATBOT EDUCATIVO CON INTELIGENCIA ARTIFICIAL
Autores: Galo Vladimir González - Jorge Andrés Orellana
Proyecto Final – Maestría en Inteligencia Artificial (UEES)
Versión 1.0 — Octubre 2025

------------------------------------------------------------
DESCRIPCIÓN GENERAL
------------------------------------------------------------
Profe Ayuda es un chatbot educativo diseñado para asistir a estudiantes de educación básica y bachillerato en el aprendizaje de matemáticas, combinando inteligencia artificial local, razonamiento paso a paso y generación de gráficos educativos.

El sistema utiliza un modelo de lenguaje local tipo Mistral, junto con un motor de recuperación semántica (RAG) basado en TF-IDF, para ofrecer explicaciones claras y contextualizadas sin depender de Internet.  
Además, genera figuras geométricas mediante Manim o SDXL-ControlNet, según los recursos del entorno.

------------------------------------------------------------
OBJETIVOS DEL PROYECTO
------------------------------------------------------------
- Desarrollar un asistente educativo inteligente centrado en matemáticas.
- Aplicar modelos de lenguaje locales (LLM) con razonamiento paso a paso.
- Implementar recuperación aumentada (RAG) con TF-IDF sobre bases JSONL.
- Incorporar gráficos automatizados y explicaciones visuales.
- Permitir el funcionamiento 100% local y offline.
- Promover un uso ético y pedagógico de la IA.

------------------------------------------------------------
TECNOLOGÍAS PRINCIPALES
------------------------------------------------------------
Backend: FastAPI + Uvicorn
IA Local: Transformers (modelo Mistral)
RAG: Scikit-learn (TF-IDF Vectorizer)
Gráficos: Manim y Diffusers (SDXL)
Optimización de VRAM: Accelerate + Torch Offload

------------------------------------------------------------
ESTRUCTURA DEL PROYECTO
------------------------------------------------------------
profe-ayuda/
├── main.py                  → API principal (FastAPI)
├── mistral_local.py          → Modelo local Mistral
├── tfidf_store.py            → Motor RAG (TF-IDF)
├── sdxl_controlnet.py        → Generador SDXL
├── scene_manim.py            → Escenas geométricas (Manim)
├── image_server.py           → Servidor opcional de imágenes
├── templates/index.html      → Interfaz del chatbot
├── uploads/figuras_geometricas.jsonl → Dataset base
├── config/                   → Índices e imágenes generadas
├── requirements.txt
└── README.txt

------------------------------------------------------------
INSTALACIÓN Y EJECUCIÓN
------------------------------------------------------------
1. Clonar o copiar el proyecto
   git clone https://github.com/usuario/profe-ayuda.git
   cd profe-ayuda

2. Instalar dependencias
   pip install -r requirements.txt

3. Instalar PyTorch según entorno:
   CPU: pip install torch==2.4.1
   CUDA: pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
   ROCm: pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/rocm6.0

4. Configurar variables de entorno:
   export MODEL_MISTRAL_PATH=./MIstral
   export MODEL_SDXL_PATH=./MIstral/sdxl
   export CONTROLNET_LINEART_PATH=./ControlNet
   export USE_CONTROLNET=0
   export LOW_VRAM=1
   export AUTO_INDEX_FILE=figuras_geometricas.jsonl

5. Ejecutar el servidor:
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload

6. Abrir en el navegador:
   http://localhost:8000

------------------------------------------------------------
ENDPOINTS PRINCIPALES
------------------------------------------------------------
GET /                → Interfaz del chatbot
GET /health          → Estado del modelo e índice
POST /chat           → Consulta con razonamiento paso a paso
POST /index          → Construcción de índice TF-IDF
POST /image/diagram  → Generación de figuras geométricas

Ejemplo de solicitud JSON:
{
  "query": "¿Cómo hallo el área de un triángulo con base 6 y altura 4?",
  "draw_example": true
}

------------------------------------------------------------
FLUJO INTERNO DEL SISTEMA
------------------------------------------------------------
1. El usuario realiza una pregunta desde la interfaz web.
2. El servidor consulta la base JSONL con TF-IDF.
3. El modelo Mistral genera los pasos explicativos.
4. Si el usuario pide un dibujo, se genera con Manim o SDXL.
5. El chatbot devuelve los pasos y la imagen.
6. Se guarda el historial y se puede corregir el resultado final.

------------------------------------------------------------
EJEMPLO DE RESPUESTA
------------------------------------------------------------
Pregunta:
¿Cómo hallo el área de un triángulo con base 8 y altura 6?

Respuesta del chatbot:
1. Identifica los datos: base = 8 y altura = 6.
2. Aplica la fórmula del área: (base × altura) ÷ 2.
3. Sustituye: (8 × 6) ÷ 2.
4. Realiza la multiplicación y división.
5. Escribe tú el resultado final. ✍️

------------------------------------------------------------
ARQUITECTURA TÉCNICA
------------------------------------------------------------
Frontend → Interfaz HTML y JavaScript.
FastAPI → Middleware que conecta los módulos.
RAGStore → Motor semántico TF-IDF (Scikit-learn).
MistralLocal → LLM local para razonamiento paso a paso.
SDXL / Manim → Módulos gráficos educativos.
Evaluador → Detección y validación numérica.

------------------------------------------------------------
CONSIDERACIONES ÉTICAS
------------------------------------------------------------
- La IA no reemplaza al docente; complementa su labor.
- Promueve el aprendizaje activo y razonado.
- Evita sesgos o entrega directa de resultados.
- Requiere supervisión docente para su uso educativo.

------------------------------------------------------------
ERRORES COMUNES Y SOLUCIONES
------------------------------------------------------------
No genera imagen → Instalar manim o revisar SDXL.
Error de VRAM → Activar LOW_VRAM=1.
LLM no responde → Revisar ruta MODEL_MISTRAL_PATH.
Índice vacío → Confirmar archivo en uploads/.
Respuestas repetitivas → Bajar temperatura del modelo.


------------------------------------------------------------
REFERENCIAS BIBLIOGRÁFICAS
------------------------------------------------------------
- Tiangolo, S. (2025). FastAPI Documentation. https://fastapi.tiangolo.com
- Hugging Face (2025). Transformers & Diffusers. https://huggingface.co/docs
- Mistral AI (2025). Mistral-7B Instruct Model Card. https://huggingface.co/mistralai
- Manim Community Developers (2024). Manim Docs. https://docs.manim.community
- Pedregosa et al. (2024). Scikit-Learn: Machine Learning in Python. JMLR.
- OpenAI (2024). RAG and Local LLM Principles in Education.
