#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <input.typ> <output.tex> [pandoc extra args...]" >&2
  exit 1
fi

input_typst="$1"
output_tex="$2"
shift 2

if [[ ! -f "$input_typst" ]]; then
  echo "Error: input file not found: $input_typst" >&2
  exit 1
fi

tmp_typst="$(mktemp --suffix=.typ)"
trap 'rm -f "$tmp_typst"' EXIT

cp "$input_typst" "$tmp_typst"

# Pandoc's Typst reader currently misses some math identifiers/symbol aliases.
# Keep the replacements intentionally minimal and targeted.
perl -i -pe 's/\bmapsto\b/->/g; s/\barrow\.r\.bar\b/->/g; s/partial_/diff_/g; s/\bpartial\b/diff/g' "$tmp_typst"

pandoc "$tmp_typst" -f typst -t latex -o "$output_tex" "$@"

echo "Wrote $output_tex"
