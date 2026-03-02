#!/bin/bash
# Wave Field LLM — Single entry point for all benchmarks
#
# Usage:
#   ./run.sh lc-sweet          # 2K+4K long-context sweet spot (~25 min)
#   ./run.sh lc-2k             # 2K only (~15 min)
#   ./run.sh lc-4k             # 4K only (~20 min)
#   ./run.sh s1                # S1 scaling (22M params, ~25 min)
#   ./run.sh s2                # S2 scaling (55M params, ~2.3 hrs)
#   ./run.sh s3                # S3 scaling (100M params, ~9 hrs)
#   ./run.sh v43               # V4.3 benchmark (5M tokens)
#   ./run.sh exp-a             # Experiment A: PPL gap vs seq length (~1.5 hrs)
#   ./run.sh exp-a-512         # Exp A: seq=512 only (~25 min)
#   ./run.sh clean             # Remove all wave containers
#   ./run.sh nuke              # Remove containers + images + volumes
#
# Options:
#   --rebuild                  # Force full Docker rebuild (no cache)

set -e

SERVICE="${1:-lc-sweet}"
REBUILD=0
for arg in "$@"; do
    [ "$arg" = "--rebuild" ] && REBUILD=1
done

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Wave Field LLM — Docker Runner${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════${NC}"

# ── Always clean stale containers first ──
echo -e "\n${YELLOW}[cleanup]${NC} Stopping stale wave-* containers..."
STALE=$(docker ps -a --filter "name=wave-" --format "{{.Names}}" 2>/dev/null || true)
if [ -n "$STALE" ]; then
    echo -e "  ${RED}Removing:${NC} $STALE"
    docker stop $STALE 2>/dev/null || true
    docker rm $STALE 2>/dev/null || true
else
    echo -e "  ${GREEN}Clean.${NC}"
fi

case "$SERVICE" in
  clean)
    echo -e "\n${GREEN}Done. All wave-* containers removed.${NC}"
    exit 0
    ;;
  nuke)
    echo -e "\n${RED}Nuking everything (images + volumes)...${NC}"
    docker compose down --rmi all --volumes --remove-orphans 2>/dev/null || true
    docker container prune -f
    docker image prune -f
    echo -e "${GREEN}Done.${NC}"
    exit 0
    ;;
  s1|s2|s3|v43|lc-2k|lc-4k|lc-sweet|exp-a|exp-a-512|exp-a-2k|exp-a-4k)
    # Get version for results tagging
    VERSION=$(python -c "import sys; sys.path.insert(0,'.'); from src import __version__; print(__version__)" 2>/dev/null || echo "unknown")
    echo -e "\n${YELLOW}[version]${NC} v${VERSION}"

    # Create versioned results dirs
    VDIR="results/v${VERSION}"
    mkdir -p "${VDIR}/checkpoints" "${VDIR}/monitor" "${VDIR}/data" "${VDIR}/plots"
    echo -e "${YELLOW}[results]${NC} ${VDIR}/"

    # Build
    echo -e "\n${YELLOW}[build]${NC} Docker image..."
    if [ "$REBUILD" -eq 1 ]; then
        docker compose build --no-cache "$SERVICE"
    else
        docker compose build "$SERVICE"
    fi

    # Run
    echo -e "\n${YELLOW}[run]${NC} ${SERVICE}"
    echo -e "  Start: $(date '+%Y-%m-%d %H:%M:%S')\n"

    docker compose run --rm \
        --name "wave-${SERVICE}" \
        -e RESULTS_VERSION="v${VERSION}" \
        "$SERVICE"

    echo -e "\n  ${GREEN}Done: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo -e "  Results: ${VDIR}/"
    ;;
  *)
    echo -e "${RED}Unknown service: $SERVICE${NC}"
    echo ""
    echo "Available services:"
    echo "  lc-sweet   2K+4K long-context sweet spot (~25 min)"
    echo "  lc-2k      2K seq only (~15 min)"
    echo "  lc-4k      4K seq only (~20 min)"
    echo "  s1         S1 scaling 22M params (~25 min)"
    echo "  s2         S2 scaling 55M params (~2.3 hrs)"
    echo "  s3         S3 scaling 100M params (~9 hrs)"
    echo "  v43        V4.3 benchmark (5M tokens)"
    echo "  exp-a      Experiment A: PPL gap vs seq length (~1.5 hrs)"
    echo "  exp-a-512  Exp A: seq=512 pair only (~25 min)"
    echo "  exp-a-2k   Exp A: seq=2048 pair only (~35 min)"
    echo "  exp-a-4k   Exp A: seq=4096 pair only (~50 min)"
    echo "  clean      Remove wave-* containers"
    echo "  nuke       Remove containers + images + volumes"
    exit 1
    ;;
esac
