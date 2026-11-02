# ==========================================
# ContextIQ - Production Dockerfile
# Optimized for Render Cloud, Hugging Face Spaces, and Local Deployments
# ==========================================

FROM python:3.12-slim

# Prevent Python from buffering stdout/stderr and writing .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=10000

# Install required system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security compliance
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Copy dependency requirements first (maximizes Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the SentenceTransformer embedding model into the image
# This eliminates runtime cold-starts and prevents network timeouts in Render
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application source code
COPY . .

# Ensure storage directories exist and have proper permissions for appuser
RUN mkdir -p data/uploads data/index && \
    chown -R appuser:appuser /app

# Switch to non-root user for runtime security
USER appuser

# Expose Render standard port (10000) as well as Streamlit default (8501)
EXPOSE 10000 8501 7860

# Launch Streamlit with dynamic PORT expansion for Render cloud routing
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-10000} --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false"]
