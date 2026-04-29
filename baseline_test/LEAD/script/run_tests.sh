#!/bin/bash
# ============================================================
# 运行全部单元测试
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

python -m pytest tests/ -v --tb=short "$@"
