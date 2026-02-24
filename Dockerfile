FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Python deps (cached layer — only rebuilds when requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt matplotlib

# Copy project (see .dockerignore for exclusions)
COPY . .

# Results directory (volume mount target)
RUN mkdir -p /app/results

# Default benchmark — override with BENCHMARK env var or docker compose command
ENV BENCHMARK=benchmarks/benchmark_v43.py
CMD python ${BENCHMARK}
