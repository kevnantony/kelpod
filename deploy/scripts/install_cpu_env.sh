#!/usr/bin/env bash
# Install the locked Python dependencies into a venv with CPU-only PyTorch (Linux CI / CPU image).
#
# Why: uv.lock resolves torch from PyPI, whose Linux wheels depend on ~3 GB of nvidia-* CUDA
# wheels. Until pyproject.toml routes torch to the PyTorch CPU index (see docs/runbooks/
# mac-dev.md, "CPU-only torch"), we export the lockfile pins, drop the CUDA-only packages and
# let uv fetch the same torch/torchvision versions as "+cpu" builds (--torch-backend cpu).
# Every other package keeps its exact locked version.
#
# usage: install_cpu_env.sh [--dev] [--venv PATH] [--project DIR] [--no-project-install]
#   --dev                 also install the `dev` dependency group (ruff, mypy, pytest, ...)
#   --venv PATH           target venv (default: $UV_PROJECT_ENVIRONMENT or <project>/.venv)
#   --project DIR         project root containing pyproject.toml + uv.lock (default: cwd)
#   --no-project-install  only third-party deps (Docker layer caching); install zllf later with
#                         `uv pip install --python <venv> --no-deps -e <project>`
set -euo pipefail

dev=0
project="$PWD"
venv=""
install_project=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev) dev=1 ;;
    --venv) venv="$2"; shift ;;
    --project) project="$2"; shift ;;
    --no-project-install) install_project=0 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
venv="${venv:-${UV_PROJECT_ENVIRONMENT:-$project/.venv}}"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

export_args=(--frozen --no-hashes --no-emit-project --format requirements.txt)
[[ $dev == 1 ]] || export_args+=(--no-dev)
# -o, not a stdout redirect: with FORCE_COLOR=1 (set in CI) uv colours stdout, which puts an
# ANSI escape before the first line and makes `uv pip install -r` reject the file.
uv export --project "$project" "${export_args[@]}" -o "$tmp/req-all.txt" >/dev/null

# Drop CUDA runtime wheels and triton (GPU kernels); CPU torch does not need them.
grep -viE '^(nvidia-|triton==|cuda-|cupy)' "$tmp/req-all.txt" > "$tmp/req-cpu.txt"
echo "install_cpu_env: $(grep -cE '^[a-zA-Z0-9]' "$tmp/req-cpu.txt") pinned packages" \
     "($(grep -ciE '^(nvidia-|triton==|cuda-|cupy)' "$tmp/req-all.txt" || true) CUDA-only dropped)"

[[ -x "$venv/bin/python" ]] || uv venv --python 3.12 "$venv"
uv pip install --python "$venv/bin/python" --torch-backend cpu -r "$tmp/req-cpu.txt"
if [[ $install_project == 1 ]]; then
  uv pip install --python "$venv/bin/python" --no-deps -e "$project"
fi
"$venv/bin/python" - <<'PY'
import importlib.util
spec = importlib.util.find_spec("torch")
if spec is None:
    print("install_cpu_env: torch not installed")
else:
    import torch
    print(f"install_cpu_env: torch {torch.__version__} cuda={torch.version.cuda}")
PY
