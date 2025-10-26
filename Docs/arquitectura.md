# Arquitectura

El sistema **Profe Ayuda** está compuesto por tres capas principales: **API (FastAPI)**, **modelo IA (Mistral)** e **interfaz (Streamlit)**.
El flujo modular garantiza escalabilidad, mantenimiento y facilidad de uso.

## Flujo del sistema
```
Usuario → Interfaz Streamlit → API FastAPI → Modelo Mistral → Respuesta generada
```

[Insertar diagrama de flujo aquí]

## Componentes
- **FastAPI:** Gestiona solicitudes y respuestas entre el usuario y el modelo.  
- **Mistral:** Procesa consultas en lenguaje natural y genera explicaciones paso a paso.  
- **Streamlit:** Permite la interacción visual amigable y accesible.  
- **Arquitectura basada en microservicios** con validación de entradas y salidas JSON.
