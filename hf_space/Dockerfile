FROM python:3.12-slim

WORKDIR /app

# Install dependencies (torch CPU-only for inference)
RUN pip install --no-cache-dir \
    openenv-core \
    fastapi \
    uvicorn \
    mcp \
    torch --index-url https://download.pytorch.org/whl/cpu \
    transformers \
    accelerate \
    sentencepiece

# Pre-download model weights at build time (faster cold start)
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct'); \
    AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')"

# Copy app code (paths relative to HF Space repo root)
COPY server/ server/
COPY models.py .
COPY serve.py .
COPY static/ static/

EXPOSE 7860

CMD ["python", "serve.py"]
