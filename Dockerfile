FROM python:3.10-slim

# Install system audio libraries and build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces requires a non-root user with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR $HOME/app

# Install Python dependencies
COPY --chown=user requirements.txt $HOME/app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the deepfake model during build for instant cold start
RUN python3 -c "from transformers import pipeline; pipeline('audio-classification', model='mo-thecreator/Deepfake-audio-detection')"

# Copy application source code
COPY --chown=user . $HOME/app

# Ensure outputs directory exists and is writable
RUN mkdir -p outputs

# Hugging Face Spaces maps port 7860 by default
EXPOSE 7860

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-7860}"]
