#!/usr/bin/bash
# Usage: ./tsv_to_csv.sh input.tsv output.csv

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 input.tsv output.csv"
  exit 1
fi

input="$1"
output="$2"

# Convert tabs to commas
tr '\t' ',' < "$input" > "$output"