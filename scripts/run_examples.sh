#!/usr/bin/env bash
# Run the hand-written example files through the CLI and keep the output.
set -u
cd "$(dirname "$0")/.."
for file in examples/*.txt; do
    echo "== $file"
    uv run --frozen regexsmith solve -f "$file" --budget 5 --no-color
    echo
done | tee results/hand_written_examples.txt
