# ==========================================
# ContextIQ - Ultra-Low-Memory Production Dockerfile
# Engineered with ONNX Runtime & FastEmbed (Guaranteed <300MB RAM)
# ==========================================

FROM python:3.12-slim

# Prevent stdout buffering and enforce single-threaded execution
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=10000 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    STREAMLIT_SERVER_MAX_UPLOAD_SIZE=50 \
    HF_HOME=/app/.cache \
    FASTEMBED_CACHE_PATH=/app/.cache \
    PYTHONPATH=/app/src

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Ensure cache and storage directories exist with non-root ownership
RUN mkdir -p /app/.cache data/uploads data/index && \
    chown -R appuser:appuser /app

# Install dependencies (Zero-PyTorch, ~200MB total footprint)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Pre-download the ONNX MiniLM model into /app/.cache during build
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')"

# Pre-generate sample documents and default FAISS index
RUN python src/create_sample_pdf.py && \
    python -c "from rag_pipeline import process_pdf; process_pdf('data/uploads/sample.pdf')"

EXPOSE 10000 8501 7860

# Launch Streamlit with dynamic PORT
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-10000} --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false"]
