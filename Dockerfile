FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHROMA_DB_PATH=/data/chroma \
    AI_PROVIDER=openrouter \
    OPENROUTER_MODEL=meta-llama/llama-3-8b-instruct:free \
    OLLAMA_MODEL=gemma4:26b

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
