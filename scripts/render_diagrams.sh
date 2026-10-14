#!/usr/bin/env bash
# Re-render docs/diagrams/*.mmd to SVG next to the sources.
#
# Needs node >= 18 and a Chromium. Defaults to `npx @mermaid-js/mermaid-cli`; override with
#   MMDC=/path/to/mmdc                      an installed mermaid-cli binary
#   PUPPETEER_CONFIG=/path/puppeteer.json   puppeteer settings (executablePath, --no-sandbox)
# If PUPPETEER_CONFIG is unset a minimal config with --no-sandbox is generated, and
# PUPPETEER_EXECUTABLE_PATH (if set) is used as the browser.
set -euo pipefail
cd "$(dirname "$0")/.."

cfg="${PUPPETEER_CONFIG:-}"
tmp=""
if [ -z "$cfg" ]; then
  tmp=$(mktemp)
  printf '{"args":["--no-sandbox","--disable-gpu"]%s}\n' \
    "${PUPPETEER_EXECUTABLE_PATH:+,\"executablePath\":\"$PUPPETEER_EXECUTABLE_PATH\"}" > "$tmp"
  cfg="$tmp"
fi

for f in docs/diagrams/*.mmd; do
  if [ -n "${MMDC:-}" ]; then
    "$MMDC" -p "$cfg" -c docs/diagrams/mermaid.json -b transparent -i "$f" -o "${f%.mmd}.svg"
  else
    npx --yes @mermaid-js/mermaid-cli -p "$cfg" -c docs/diagrams/mermaid.json -b transparent -i "$f" -o "${f%.mmd}.svg"
  fi
done
[ -z "$tmp" ] || rm -f "$tmp"
