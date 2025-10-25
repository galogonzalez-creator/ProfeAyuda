# tfidf_store.py
import os
import json
import pickle
import glob
from typing import List, Dict, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────
# Stopwords español (opcional, sin romper si no hay NLTK)
# ─────────────────────────────────────────────────────────
STOP_ES: Optional[List[str]] = None
try:
    from nltk.corpus import stopwords  # type: ignore
    try:
        STOP_ES = stopwords.words("spanish")
    except Exception:
        STOP_ES = None
except Exception:
    STOP_ES = None


def _normalize_text(s: str) -> str:
    """Limpieza ligera para mejorar el TF-IDF."""
    if not s:
        return ""
    s = str(s)
    # quita saltos, tabs y comprime espacios
    s = " ".join(s.split())
    return s.lower()


class RAGStore:
    """
    Almacén TF-IDF multi-índice para RAG sencillo.

    • build(jsonl)  -> crea y guarda índice {vectorizer, matrix, data}
    • load_index()  -> carga índice guardado
    • retrieve()    -> recupera top-k documentos por similitud coseno
    • list_indexes() / set_active() / delete_index()
    """

    def __init__(self, base_dir: str = "./config"):
        os.makedirs(base_dir, exist_ok=True)
        self.base_dir = base_dir
        self.vectorizers: Dict[str, TfidfVectorizer] = {}
        self.matrices = {}   # nombre -> scipy.sparse.csr_matrix
        self.datasets: Dict[str, List[str]] = {}
        self.active_dataset: Optional[str] = None
        print(f"[INFO] RAGStore inicializado. Carpeta base: {base_dir}")

        # Activa automáticamente un índice si ya existe alguno
        self.auto_activate()

    # ─────────────────────────────────────────────────────────
    # Utilidades de persistencia
    # ─────────────────────────────────────────────────────────
    def _index_path(self, name: str) -> str:
        return os.path.join(self.base_dir, f"{name}_index.pkl")

    def list_indexes(self) -> List[str]:
        files = glob.glob(os.path.join(self.base_dir, "*_index.pkl"))
        return sorted(os.path.basename(f).replace("_index.pkl", "") for f in files)

    def auto_activate(self) -> None:
        """Activa el primer índice disponible si no hay activo."""
        if self.active_dataset:
            return
        existing = self.list_indexes()
        if existing:
            try:
                self.load_index(existing[0])
            except Exception:
                pass

    def set_active(self, name: str) -> None:
        if name not in self.vectorizers and os.path.exists(self._index_path(name)):
            self.load_index(name)
        if name not in self.vectorizers:
            raise ValueError(f"Índice '{name}' no está cargado.")
        self.active_dataset = name
        print(f"[INFO] Dataset activo: {name}")

    def delete_index(self, name: str) -> bool:
        """Elimina el archivo del índice y lo descarga de memoria."""
        removed = False
        p = self._index_path(name)
        if os.path.exists(p):
            os.remove(p)
            removed = True
        self.vectorizers.pop(name, None)
        self.matrices.pop(name, None)
        self.datasets.pop(name, None)
        if self.active_dataset == name:
            self.active_dataset = None
            self.auto_activate()
        return removed

    # ─────────────────────────────────────────────────────────
    # Construcción / carga
    # ─────────────────────────────────────────────────────────
    def build(self, jsonl_path: str, text_cols: Optional[List[str]] = None, name: Optional[str] = None) -> str:
        """
        Crea un índice TF-IDF a partir de un JSONL.
        Cada línea del JSONL debe ser un dict con columnas en `text_cols`.
        """
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"No existe el archivo: {jsonl_path}")

        name = name or os.path.splitext(os.path.basename(jsonl_path))[0]
        cols = text_cols or ["instruction", "input", "output"]
        print(f"[INFO] Entrenando índice: {name}")

        data: List[str] = []
        bad_lines = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    bad_lines += 1
                    continue
                parts = [_normalize_text(item.get(c, "")) for c in cols]
                txt = " ".join(p for p in parts if p)
                if txt:
                    data.append(txt)

        if not data:
            raise ValueError(f"El dataset '{jsonl_path}' no tiene filas válidas para indexar.")
        if bad_lines:
            print(f"[WARN] {bad_lines} líneas con JSON inválido se omitieron.")

        vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            stop_words=STOP_ES,          # si STOP_ES es None, sklearn ignora
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
        )
        matrix = vectorizer.fit_transform(data)

        self.vectorizers[name] = vectorizer
        self.matrices[name] = matrix
        self.datasets[name] = data
        self.active_dataset = name

        with open(self._index_path(name), "wb") as f:
            pickle.dump({"vectorizer": vectorizer, "matrix": matrix, "data": data}, f)

        print(f"[OK] Índice '{name}' entrenado y guardado en {self.base_dir}/")
        return name

    def load_index(self, name: str) -> str:
        p = self._index_path(name)
        if not os.path.exists(p):
            raise FileNotFoundError(f"No se encontró el índice: {p}")

        with open(p, "rb") as f:
            obj = pickle.load(f)

        self.vectorizers[name] = obj["vectorizer"]
        self.matrices[name] = obj["matrix"]
        self.datasets[name] = obj["data"]
        self.active_dataset = name
        print(f"[INFO] Índice '{name}' cargado correctamente.")
        return name

    def load_or_build(self, jsonl_path: str, text_cols: Optional[List[str]] = None, name: Optional[str] = None) -> str:
        """Carga el índice si existe; de lo contrario, lo construye."""
        name = name or os.path.splitext(os.path.basename(jsonl_path))[0]
        if os.path.exists(self._index_path(name)):
            return self.load_index(name)
        return self.build(jsonl_path, text_cols=text_cols, name=name)

    # ─────────────────────────────────────────────────────────
    # Recuperación
    # ─────────────────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int = 5, dataset_name: Optional[str] = None) -> List[Dict]:
        """
        Devuelve los documentos más similares a `query`.
        """
        dataset_name = dataset_name or self.active_dataset
        if not dataset_name:
            # Sin dataset activo: no tumbar el backend
            return []

        vec = self.vectorizers.get(dataset_name)
        mat = self.matrices.get(dataset_name)
        docs = self.datasets.get(dataset_name)
        if vec is None or mat is None or docs is None:
            return []

        q = _normalize_text(query or "")
        if not q:
            return []

        query_vec = vec.transform([q])
        sims = cosine_similarity(query_vec, mat).flatten()
        if sims.size == 0:
            return []

        top_idx = sims.argsort()[-int(top_k):][::-1]
        results = [{"text": docs[i], "score": float(sims[i])} for i in top_idx]
        return results

