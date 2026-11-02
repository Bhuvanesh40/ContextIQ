# ==========================================
# ContextIQ - Memory-Optimized Production Dockerfile
# Specifically engineered for Render Free Tier (<500MB RAM)
# ==========================================

FROM python:3.12-slim

# Prevent Python from buffering stdout/stderr and writing .pyc files
# Enforce single-threaded execution for BLAS/OpenMP to avoid thread stack memory bloat
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=10000 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    TORCH_NUM_THREADS=1 \
    STREAMLIT_SERVER_MAX_UPLOAD_SIZE=50

# Install minimal build tools and curl for health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Step 1: Install CPU-only PyTorch (crucial: avoids 2GB CUDA bloat and saves ~300MB RAM)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install remaining lean production requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 3: Pre-download model weights into the container image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Step 4: Copy application source code
COPY . .

# Ensure storage directories exist and have proper permissions for appuser
RUN mkdir -p data/uploads data/index && \
    chown -R appuser:appuser /app

# Switch to non-root user for security
USER appuser

EXPOSE 10000 8501 7860

# Launch Streamlit with dynamic PORT and lean memory flags
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-10000} --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false"]
