FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends git gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Python deps (cached layer — only rebuilds when requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt matplotlib

# Copy project (see .dockerignore for exclusions)
COPY . .

# Results directory (volume mount target)
RUN mkdir -p /app/results

# Benchmark config — override any of these at runtime
ENV BENCHMARK=benchmarks/benchmark_scaling.py
ENV SCALE=""
ENV DATASET=2
ENV MODEL=""
ENV CONFIGS=""

CMD ["sh", "-c", "python $BENCHMARK"]
