#!/usr/bin/env bash
# Bundles nibabel (+ its lightweight deps) into packages/, required because
# network is disabled during ingestion — see SUBMISSION_GUIDE.md. packages/ is
# gitignored (don't commit binary wheels); run this locally before zipping.
#
# Usage:
#   ./build.sh
#   zip -r submission.zip run.py metadata packages/

set -e
cd "$(dirname "$0")"

pip install --target=packages --no-deps -q nibabel packaging importlib-resources typing-extensions

echo "packages/ built."
echo "Zip with: zip -r submission.zip run.py metadata packages/"
