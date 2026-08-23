#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    if ! command -v powershell.exe >/dev/null 2>&1; then
      echo "Windows PowerShell is required for Git Bash setup." >&2
      exit 2
    fi
    script_path="$(cygpath -w "$project_root/scripts/setup-pcloud.ps1")"
    exec powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$script_path"
    ;;
esac

version="1.75.0"
platform=""
expected_sha256=""
binary_name="rclone"

case "$(uname -s):$(uname -m)" in
  Linux:x86_64)
    platform="linux-amd64"
    expected_sha256="aa2804e08f48250e71009c727124b6341cd0288465804a9a09d14663cabafbaa"
    ;;
  *)
    echo "Unsupported setup platform: $(uname -s) $(uname -m)" >&2
    exit 2
    ;;
esac

archive_name="rclone-v${version}-${platform}.zip"
tool_root="$project_root/.tools/rclone/v${version}"
archive_path="$tool_root/$archive_name"
rclone_path="$tool_root/rclone-v${version}-${platform}/$binary_name"
config_directory="$project_root/secrets/rclone"
config_path="$config_directory/rclone.conf"

mkdir -p "$tool_root" "$config_directory"
if [ ! -f "$archive_path" ]; then
  echo "Downloading rclone v${version} from the official release site..."
  curl --fail --location --output "$archive_path" \
    "https://downloads.rclone.org/v${version}/${archive_name}"
fi
printf '%s  %s\n' "$expected_sha256" "$archive_path" | sha256sum --check -

if [ ! -f "$rclone_path" ]; then
  if command -v unzip >/dev/null 2>&1; then
    unzip -q -o "$archive_path" -d "$tool_root"
  elif command -v python3 >/dev/null 2>&1; then
    python3 -m zipfile -e "$archive_path" "$tool_root"
  else
    echo "A ZIP extractor is required (install unzip or Python 3)." >&2
    exit 2
  fi
fi
if [ -f "$rclone_path" ]; then
  chmod 755 "$rclone_path"
fi
if [ ! -x "$rclone_path" ]; then
  echo "rclone executable was not found after extraction: $rclone_path" >&2
  exit 1
fi

echo
echo "A browser window will open for one-time pCloud authorization."
echo "The OAuth credential will be saved without being printed to the terminal."
echo
"$rclone_path" --config "$config_path" config create highlightcraft-pcloud pcloud --no-output
chmod 600 "$config_path"

remotes="$("$rclone_path" --config "$config_path" listremotes)"
case $'\n'"$remotes"$'\n' in
  *$'\nhighlightcraft-pcloud:\n'*) ;;
  *)
    echo "The required remote highlightcraft-pcloud: was not created" >&2
    exit 1
    ;;
esac

echo
echo "OAuth config saved outside Git at: $config_path"
echo "Next (Linux):"
printf 'export PINGPONG_UID=%s PINGPONG_GID=%s\n' "$(id -u)" "$(id -g)"
echo "docker compose run --rm --no-deps pcloud-admin doctor"
