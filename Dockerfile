# callcenter-intelligence
# Dockerfile

FROM python:3.11-slim

WORKDIR /app

# ffmpeg: non-WAV decode. libsndfile1: soundfile reads Gradio's numpy recordings.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
