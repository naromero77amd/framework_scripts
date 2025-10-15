#!/usr/bin/bash

# environment variables
export CUDA_VISIBLE_DEVICES=7
export AMDGCN_USE_BUFFER_OPS=1
export TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=1

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
echo "relative_dir,filename,kernel_name,ms_per_call,gb_per_s" > "$CSV_FILE"

# --- Find and run triton_*_<integer>.py files ---
# Change regex to include all hashed files
find "$BASE_DIR" -type f -regextype posix-extended -regex '.*/triton_per_.*_[0-9]+(_[A-Za-z0-9]+)?.py' | while read -r file; do
    echo "Running $file ..."

    # Relative directory from BASE_DIR
    rel_dir=$(dirname "${file#$BASE_DIR/}")

    # --- Extract xnumel and ro_numel from file contents ---
    xnumel=$(grep -Eo '^[[:space:]]*xnumel *= *[0-9]+' "$file" | head -n1 | grep -Eo '[0-9]+')
    ro_numel=$(grep -Po '^[[:space:]]*r0_numel\s*=\s*\K[0-9]+' "$file" | head -n1)
    # r0_numel=$(grep -Eo '^[[:space:]]*r0_numel *= *[0-9]+' "$file" | head -n1 | grep -Eo '[0-9]+')

    # Fallback if not found
    [[ -z "${xnumel:-}" ]] && xnumel="0"
    [[ -z "${ro_numel:-}" ]] && r0_numel="0"

    # Kernel name: basename + xnumel/ro_numel suffix
    base_kernel=$(basename "$file" .py)
    if [[ $base_kernel =~ ^(.*_[0-9]+)_[A-Za-z0-9]+$ ]]; then
        base_kernel="${BASH_REMATCH[1]}"
    fi
    kernel_name="${base_kernel}_x${xnumel}_r${ro_numel}"

    # Kernel name (basename without .py)
    # kernel_name=$(basename "$file" .py)

    # Run the Python file and capture output
    output=$(python "$file")

    # Parse numeric values from output
    ms=$(echo "$output" | awk '{gsub("ms","",$1); print $1}')
    gb_per_s=$(echo "$output" | awk '{gsub("GB/s","",$3); print $3}')

    # Write CSV line
    echo "$rel_dir,$file,$kernel_name,$ms,$gb_per_s" >> "$CSV_FILE"
done

echo "Results written to $CSV_FILE"
