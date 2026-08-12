#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v cygpath >/dev/null 2>&1; then
  echo "This launcher is intended for Git Bash on Windows." >&2
  exit 1
fi

powershell_script="$(cygpath -w "$script_dir/start-localhost.ps1")"
exec powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$powershell_script" "$@"
