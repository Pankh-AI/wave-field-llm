#!/bin/bash
# Wave Field LLM — Single entry point for all benchmarks
#
# Usage:
#   ./run.sh s1       # Run S1 scaling (22M params, ~25 min)
#   ./run.sh s2       # Run S2 scaling (55M params, ~2.3 hrs)
#   ./run.sh s3       # Run S3 scaling (100M params, ~9 hrs)
#   ./run.sh v43      # Run V4.3 benchmark (5M tokens)
#   ./run.sh clean    # Remove all wave containers + images

set -e

SERVICE="${1:-s1}"

case "$SERVICE" in
  clean)
    echo "Cleaning all wave containers and images..."
    docker compose down --rmi all --volumes --remove-orphans 2>/dev/null || true
    docker container prune -f
    docker image prune -f
    echo "Done."
    ;;
  s1|s2|s3|v43)
    echo "=== Running $SERVICE ==="
    echo "Building image..."
    docker compose build "$SERVICE"
    echo "Starting $SERVICE (use Ctrl+C to stop)..."
    docker compose run --rm "$SERVICE"
    echo "=== $SERVICE complete. Results in ./results/ ==="
    ;;
  *)
    echo "Unknown service: $SERVICE"
    echo "Available: s1, s2, s3, v43, clean"
    exit 1
    ;;
esac
