#!/usr/bin/env bash
# Moga Otonom - kurulum ve demo (Linux / macOS)
#
#   bash scripts/setup_and_demo.sh              # MogaSim + otonomi yığını, testler, demo
#   bash scripts/setup_and_demo.sh --metadrive  # ek olarak MetaDrive ortamı ve 2 sürüş
#
# Ortam değişkenleri: PYTHON (varsayılan python3), PYTHON311 (varsayılan python3.11)
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
"$PY" - <<'EOF'
import sys
if sys.version_info < (3, 10):
    sys.exit(f"Python 3.10 veya üstü gerekli (bulunan: {sys.version.split()[0]})")
EOF

echo "==> 1/3 Sanal ortam (.venv) ve paketler"
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt

echo "==> 2/3 Hızlı testler (~30 s)"
.venv/bin/python -m pytest -q -n 4 -m "not scenario" -p no:cacheprovider

echo "==> 3/3 Demo: kırsal yolda karşı trafiği bekleyip arızalı aracı geçme (~1 dk)"
.venv/bin/python scripts/run_scenario.py m4_rural_bypass --gif out/demo.gif --overview out/demo.png
echo "Çıktılar: out/demo.gif (animasyon), out/demo.png (rota ve hız)"
echo "Canlı izlemek için: .venv/bin/python scripts/run_scenario.py m5_urban_mission --live"

if [ "${1:-}" = "--metadrive" ]; then
    PY311="${PYTHON311:-python3.11}"
    if ! command -v "$PY311" >/dev/null 2>&1; then
        echo "MetaDrive için Python 3.11 gerekli (ör. PYTHON311=/usr/local/bin/python3.11)" >&2
        exit 1
    fi
    echo "==> MetaDrive ortamı (.venv-metadrive)"
    [ -d .venv-metadrive ] || "$PY311" -m venv .venv-metadrive
    .venv-metadrive/bin/python -m pip install -q --upgrade pip
    .venv-metadrive/bin/python -m pip install -q -r requirements-metadrive.txt
    .venv-metadrive/bin/python -m pip install -q --no-deps metadrive-simulator==0.2.6.0
    .venv-metadrive/bin/python scripts/run_metadrive.py --seeds 0 1 --map 3 --plot --out out/metadrive
    echo "Çıktılar: out/metadrive/ (her sürüş için harita + hız grafiği, JSON)"
    echo "3B pencerede izlemek için: .venv-metadrive/bin/python scripts/run_metadrive.py --seeds 3 --render"
fi
