#!/usr/bin/bash

# environment variables
export AMDGCN_USE_BUFFER_OPS=1
export TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=1
export CUDA_VISIBLE_DEVICES=7

# Usage: ./run_triton.sh <kernels directory>
BASE_DIR="$1"

if [[ -z "$BASE_DIR" ]]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Ensure base directory exists
if [[ ! -d "$BASE_DIR" ]]; then
    echo "Directory $BASE_DIR does not exist."
    exit 1
fi

# --- Cleanup once at the start ---
echo "Initial cleanup..."

# delete old inductor cache
if [[ -d "/tmp/torchinductor_root" ]]; then
    rm -rf /tmp/torchinductor_root
fi

# Delete all *.best_config files
find "$BASE_DIR" -type f -name '*.best_config' -exec rm -f {} +

# Delete all *.py.launch_params files
find "$BASE_DIR" -type f -name '*.py.launch_params' -exec rm -f {} +

# --- Prepare CSV inside BASE_DIR ---
CSV_FILE="$BASE_DIR/triton_results_$(date +%Y%m%d_%H%M%S).csv"
echo "relative_dir,kernel_name,ms_per_call,gb_per_s" > "$CSV_FILE"

# --- Find and run triton_*_<integer>.py files ---
find "$BASE_DIR" -type f -regextype posix-extended -regex '.*/triton_.*_[0-9]+\.py' | while read -r file; do
    echo "Running $file ..."

    # Relative directory from BASE_DIR
    rel_dir=$(dirname "${file#$BASE_DIR/}")

    # Kernel name (basename without .py)
    kernel_name=$(basename "$file" .py)

    # Run the Python file and capture output
    output=$(python "$file")

    # Parse numeric values from output
    ms=$(echo "$output" | awk '{gsub("ms","",$1); print $1}')
    gb_per_s=$(echo "$output" | awk '{gsub("GB/s","",$3); print $3}')

    # Write CSV line
    echo "$rel_dir,$kernel_name,$ms,$gb_per_s" >> "$CSV_FILE"
done

echo "Results written to $CSV_FILE"
