FROM python:3.11-slim

# Install system dependencies (including OpenCV + ImageMagick fixes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    imagemagick \
    libsndfile1 \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Fix ImageMagick policy for TextClip
RUN sed -i 's/<policy domain="coder" rights="none" pattern="PDF" \/>/<policy domain="coder" rights="read|write" pattern="PDF" \/>/g' /etc/ImageMagick-6/policy.xml || true

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application
COPY . .

# Create required folders
RUN mkdir -p jobs outputs

# Railway automatically injects $PORT
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=${PORT:-8501}", "--server.address=0.0.0.0"]