#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash download_models.sh [DATA_ROOT]

Download RoboDojo Pi_05 checkpoint 59999 and Valen-2B/Valen-4B.
DATA_ROOT defaults to the working directory where you launch this command,
not the directory containing this script. An explicit DATA_ROOT overrides it.
Models, the isolated downloader, caches and temporary files stay under DATA_ROOT.
Requires Python 3.10+ with venv/pip support. No inference environment is installed.

Examples:
  bash /path/to/download_models.sh
  HF_ENDPOINT=https://hf-mirror.com bash download_models.sh /data

Optional environment variables:
  PYTHON                 Python executable (default: python3)
  HF_ENDPOINT            Download endpoint (default: https://huggingface.co)
  HF_TOKEN               Access token if required; never stored in this script
  VALEN_2B_REVISION       Default: previously verified deployment revision
  VALEN_4B_REVISION       Default: main
  DOWNLOAD_ATTEMPTS      Retries per repository (default: 5)

Rerun the same command after interruption to reuse completed downloads.
HELP
}

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    usage
    exit 0
fi
if [[ $# -gt 1 || "${1:-}" == -* ]]; then
    usage >&2
    exit 2
fi

DATA_ROOT="${1:-$PWD}"
PYTHON="${PYTHON:-python3}"
ATTEMPTS="${DOWNLOAD_ATTEMPTS:-5}"
if [[ ! "$ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
    printf 'DOWNLOAD_ATTEMPTS must be a positive integer.\n' >&2
    exit 2
fi
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else "Python 3.10+ is required")'
mkdir -p "$DATA_ROOT"
DATA_ROOT="$(cd "$DATA_ROOT" && pwd -P)"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HOME="$DATA_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export XDG_CACHE_HOME="$DATA_ROOT/.cache"
export PIP_CACHE_DIR="$DATA_ROOT/pip-cache"
export TMPDIR="$DATA_ROOT/tmp"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$DATA_ROOT/models"

ENV_DIR="$DATA_ROOT/.model-download-venv"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    "$PYTHON" -m venv "$ENV_DIR"
fi
if [[ ! -x "$ENV_DIR/bin/hf" ]]; then
    "$ENV_DIR/bin/python" -m pip install 'huggingface_hub>=1.0,<3'
fi
HF="$ENV_DIR/bin/hf"

download() {
    local attempt
    for ((attempt = 1; attempt <= ATTEMPTS; attempt++)); do
        if "$HF" download "$@"; then
            return 0
        fi
        if ((attempt < ATTEMPTS)); then
            printf 'Download failed; retry %s/%s in 10 seconds.\n' "$((attempt + 1))" "$ATTEMPTS" >&2
            sleep 10
        fi
    done
    printf 'Download failed. Rerun this script to continue.\n' >&2
    return 1
}

printf 'Download destination: %s/models\n' "$DATA_ROOT"
df -h "$DATA_ROOT"

# This checkpoint is in a DATASET repo, not cjgogo's sim-10task/9000 model.
download RoboDojo-Benchmark/RoboDojo \
    --repo-type dataset \
    --revision 35efbc7dedfdbeeb6e95fb749bd885d73d483e41 \
    --include 'ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999/*' \
    --exclude '*/train_state/*' \
    --local-dir "$DATA_ROOT/models/RoboDojo"

# Keep tokenizer, processor, configuration and custom loading code with weights.
download Valen-Team/Valen-2B \
    --repo-type model \
    --revision "${VALEN_2B_REVISION:-8106f4c18f91a96a23a7d6dc420ac6a6ff66e8a6}" \
    --local-dir "$DATA_ROOT/models/Valen-2B"

download Valen-Team/Valen-4B \
    --repo-type model \
    --revision "${VALEN_4B_REVISION:-main}" \
    --local-dir "$DATA_ROOT/models/Valen-4B"

printf '\nAll three downloads completed:\n%s\n%s\n%s\n' \
    "$DATA_ROOT/models/RoboDojo/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999" \
    "$DATA_ROOT/models/Valen-2B" \
    "$DATA_ROOT/models/Valen-4B"
