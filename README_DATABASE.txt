==============================================
📘 GUÍA DE CARGA DE BASES DE DATOS - PROFE AYUDA
==============================================

1️⃣ Requisitos previos:
------------------------
- Tener el archivo .jsonl con los ejercicios o explicaciones.  
  Ejemplo: "quinto_EGB_unidades1-6_39temas_merged.jsonl"
- Cada línea debe tener campos como:
  {"instruction": "Pregunta", "input": "Contexto", "output": "Respuesta explicada"}

- Asegúrate de que el archivo esté dentro de la carpeta:
  ./uploads/

2️⃣ Entrenar un nuevo índice:
------------------------------
Desde la terminal (en la carpeta /chatbot):

python3 - <<'PY'
from tfidf_store import RAGStore
rag = RAGStore()
rag.build('./uploads/quinto_EGB_unidades1-6_39temas_merged.jsonl', name='profe_ayuda_5to')
PY

Esto creará un índice guardado en:
./config/profe_ayuda_5to_index.pkl

3️⃣ Cargar un índice existente:
-------------------------------
python3 - <<'PY'
from tfidf_store import RAGStore
rag = RAGStore()
rag.load_index('profe_ayuda_5to')
print(rag.retrieve("¿Cómo se suman fracciones?", top_k=3))
PY

4️⃣ Agregar una nueva base (por ejemplo, 4to EGB):
--------------------------------------------------
- Copia el archivo JSONL a ./uploads/
- Cambia el nombre en el comando build:

python3 - <<'PY'
from tfidf_store import RAGStore
rag = RAGStore()
rag.build('./uploads/cuarto_EGB_unidades1-6.jsonl', name='profe_ayuda_4to')
PY

5️⃣ Ver los índices disponibles:
-------------------------------
python3 - <<'PY'
from tfidf_store import RAGStore
rag = RAGStore()
print(rag.list_indexes())
PY

✅ Recomendación:
-----------------
Mantén una convención de nombres:
- profe_ayuda_3ro
- profe_ayuda_4to
- profe_ayuda_5to
- profe_ayuda_6to
etc.

Esto permitirá que el sistema cargue dinámicamente la base según el año o materia.


