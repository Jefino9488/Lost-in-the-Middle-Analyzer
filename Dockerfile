# Lightweight Python image
FROM python:3.11-slim

WORKDIR /app

# system deps for matplotlib backend and general utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy app
COPY . .

ENV PORT=8080
EXPOSE 8080

# Run streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true"]
