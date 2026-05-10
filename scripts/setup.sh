#!/usr/bin/env bash
# One-shot local dev setup script

set -euo pipefail

echo "==> Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required="3.11"
if [[ "$(printf '%s\n' "$required" "$python_version" | sort -V | head -n1)" != "$required" ]]; then
  echo "ERROR: Python $required+ required (found $python_version)" >&2
  exit 1
fi

echo "==> Creating virtual environment..."
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing dependencies..."
pip install --upgrade pip
pip install -r requirements-dev.txt

echo "==> Setting up pre-commit hooks..."
pre-commit install

echo "==> Copying .env.example to .env..."
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "  Created .env -- fill in your secrets before running the app."
else
  echo "  .env already exists, skipping."
fi

echo "==> Creating data directories..."
mkdir -p data/chroma credentials

echo ""
echo "Setup complete."
echo "  Activate env : source .venv/bin/activate"
echo "  Run server   : uvicorn app.main:app --reload"
echo "  Run tests    : pytest"
