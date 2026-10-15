#!/bin/bash
# Double-click this file in Finder (macOS) to launch the GPCR Docking & Analytics Suite.
# It activates the conda environment from environment.yml and starts the Streamlit app.
#
# Optional overrides (export them before launching, or edit the defaults below):
#   GPCR_DOCKING_ENV  conda env name               (default: gpcr-docking)
#   CONDA_BASE        conda install prefix         (default: `conda info --base`)
#   AUTOGRID4_BIN     autogrid4 binary, if not on PATH (hydrated docking only)
#   PREPARE_RECEPTOR4 prepare_receptor4.py, if not on PATH (hydrated docking only)
#   AMBERHOME         AmberTools install, if not in the env (MM/PBSA rescoring only)

# Always run from this script's own folder, so the app's relative paths work.
cd "$(dirname "$0")" || exit 1

echo "🧬  GPCR Docking & Analytics Suite"
echo "-----------------------------------"

ENV_NAME="${GPCR_DOCKING_ENV:-gpcr-docking}"

# --- Activate the conda environment ---------------------------------------
if [ -z "$CONDA_BASE" ] && command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
fi
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    echo "❌ conda not found. Install miniforge, or set CONDA_BASE to your conda prefix."
    echo "   (Press any key to close.)"; read -n 1; exit 1
fi

conda activate "$ENV_NAME" || {
    echo "❌ Could not activate the '$ENV_NAME' environment."
    echo "   Create it with: conda env create -f environment.yml"
    echo "   (Press any key to close.)"; read -n 1; exit 1
}

# Put the env's tools ahead of any other copies on PATH, so the Streamlit
# subprocesses (python3 run_screen.py / preptarget.py) use this env's
# interpreter and not a shadowing system python3.
export PATH="$CONDA_PREFIX/bin:$PATH"

echo "✅ Environment ready: $(which python3)"
echo "🚀 Launching the app — a browser tab will open shortly."
echo "   Leave this window open while you work; close it or press Ctrl+C to stop."
echo ""

streamlit run src/app.py
