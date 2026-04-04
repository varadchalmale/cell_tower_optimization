#!/bin/bash
# ============================================================
# Cell Tower Optimization — One-Shot Setup & Run Script
# ============================================================
# Usage:
#   chmod +x setup_and_run.sh
#   ./setup_and_run.sh            # full pipeline (100 towers)
#   ./setup_and_run.sh --quick    # simplified pipeline (15 towers)
#   ./setup_and_run.sh --skip-data  # skip downloads (data already present)
# ============================================================

set -e  # exit on first error

QUICK=false
SKIP_DATA=false

for arg in "$@"; do
  case $arg in
    --quick)      QUICK=true ;;
    --skip-data)  SKIP_DATA=true ;;
  esac
done

echo "============================================================"
echo "  Cell Tower Optimization — Nagpur District, India"
echo "  AI-driven placement: 100 towers vs Airtel's 1,200"
echo "============================================================"

# ── Step 1: Install dependencies ─────────────────────────────
echo ""
echo "[1/5] Installing Python dependencies..."
pip install -r requirements_upgraded.txt
echo "      Done."

if [ "$SKIP_DATA" = false ]; then
  # ── Step 2: Download external data ───────────────────────────
  echo ""
  echo "[2/5] Downloading external geospatial data..."
  echo "      (WorldPop ~1.2 GB + SRTM DEM ~50 MB — this may take 30-60 min)"
  python scripts/download_worldpop.py
  python scripts/download_srtm_dem.py
  echo "      Done."

  # ── Step 3: Process raw data → feature rasters ───────────────
  echo ""
  echo "[3/5] Processing raw data and generating feature rasters..."
  python src/data_processing.py
  python src/feature_engineering.py
  echo "      Done."
else
  echo ""
  echo "[2/5] Skipping data download (--skip-data flag set)."
  echo "[3/5] Skipping feature generation (--skip-data flag set)."
fi

# ── Step 4: Run optimization ──────────────────────────────────
echo ""
if [ "$QUICK" = true ]; then
  echo "[4/5] Running simplified pipeline (15 towers, src/main_upgraded.py)..."
  python src/main_upgraded.py
  echo ""
  echo "      Outputs written to: results/upgraded/"
else
  echo "[4/5] Running full 9-stage pipeline (100 towers)..."
  cd "upgraded cell tower optimization"
  python main.py
  cd ..
  echo ""
  echo "      Outputs written to: upgraded cell tower optimization/outputs/"
fi
echo "      Done."

# ── Step 5: Show results ──────────────────────────────────────
echo ""
echo "[5/5] Results summary:"
if [ "$QUICK" = true ]; then
  ls -lh results/upgraded/ 2>/dev/null || echo "      (no files yet — check for errors above)"
else
  ls -lh "upgraded cell tower optimization/outputs/" 2>/dev/null || echo "      (no files yet — check for errors above)"
  echo ""
  echo "      Open the interactive map in your browser:"
  echo "      upgraded cell tower optimization/outputs/interactive_towers.html"
fi

echo ""
echo "============================================================"
echo "  Done! Project run complete."
echo "============================================================"
