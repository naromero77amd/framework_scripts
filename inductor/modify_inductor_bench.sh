#!/usr/bin/sh
# Recursively update benchmark_gpu calls in *.py files
# Usage: ./update_benchmarks.sh [directory]
# Default directory: current working directory

set -eu

DIR="${1:-.}"

if [ ! -d "$DIR" ]; then
  echo "Error: '$DIR' is not a directory." >&2
  exit 1
fi

find "$DIR" -type f -name "*.py" | while IFS= read -r file; do
  tmpfile="$(mktemp)"
  sed \
    -e '/ms = benchmarker\.benchmark_gpu(lambda: call(args),/{
      s/rep=1000/rep=1000/g;
      s/warmup=100/warmup=100/g;
      s/, *return_mode="median"//g;
      s/return_mode="median"//g;
    }' "$file" > "$tmpfile"

  if ! cmp -s "$file" "$tmpfile"; then
    mv "$tmpfile" "$file"
    echo "Updated: $file"
  else
    rm -f "$tmpfile"
  fi
done

echo "Done."
