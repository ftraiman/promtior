# Promtior Chatbot (Ollama Version)

Esta es la implementación del chatbot RAG utilizando **Ollama** como motor de inteligencia artificial. Es una solución 100% local y gratuita.

## 📋 Requisitos Previos

Necesitas tener **Ollama** instalado y ejecutándose en tu máquina.

### 1. Instalar Ollama
- **macOS / Linux / Windows**: Descarga desde [ollama.com](https://ollama.com).

### 2. Descargar el Modelo
Abrir una terminal y ejecutar:
```bash
ollama pull llama3
```
*Este comando descargará el modelo Llama 3 (~4.7GB).*

## 🚀 Cómo Ejecutar

### 1. Instalar Dependencias del Proyecto
Navega a la carpeta del proyecto e instala los paquetes de Python:

```bash
pip install -r requirements.txt
```

### 2. Ingerir Datos
Este paso descarga el contenido de `promtior.ai`, lo procesa y genera la base de datos vectorial localmente usando Ollama embeddings.

```bash
python -m app.ingest
```

### 3. Iniciar el Servidor
Inicia la API REST y el Playground.

```bash
python -m app.server
```

### 4. Usar el Chatbot
Abre tu navegador en:
👉 [http://localhost:8000/promtior-bot/playground](http://localhost:8000/promtior-bot/playground)

## 🏗️ Arquitectura

- **LLM**: Meta Llama 3 (vía Ollama)
- **Embeddings**: Llama 3 (vía Ollama)
- **Vector Store**: ChromaDB (Local)
- **Framework**: LangChain & LangServe
