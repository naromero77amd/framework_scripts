#!/usr/bin/env python3
"""
Compute 10th percentile, mean, 90th percentile, and count of `ms_per_call`
for each kernel base (including the plain entry and its hash variants).

CSV columns expected:
    relative_dir,kernel_name,ms_per_call,gb_per_s
"""

import argparse
import os
import pandas as pd
import numpy as np
import re

# --- Regex: capture kernel up to the trailing integer (before any _hash) ----
# Example:
#   triton_per_fused_mul_native_layer_norm_relu_sigmoid_8
#   triton_per_fused_mul_native_layer_norm_relu_sigmoid_8_c2pedc7...
# -> base = triton_per_fused_mul_native_layer_norm_relu_sigmoid_8
BASE_RE = re.compile(r"^(.*?_\d+)(?:_[0-9a-z]+)?$", re.IGNORECASE)


def get_base_name(name: str) -> str:
    """
    Return the part of kernel_name that ends with its integer index,
    stripping any optional '_<hash>' suffix.
    """
    m = BASE_RE.match(name)
    return m.group(1) if m else name


def main():
    parser = argparse.ArgumentParser(
        description="Compute 10th/mean/90th percentiles and counts for kernels (base + hash variants)."
    )
    parser.add_argument("csvfile", help="Input CSV file")
    parser.add_argument("-o", "--output", help="Write results to CSV instead of stdout")
    args = parser.parse_args()

    # ---- Load CSV ---------------------------------------------------------
    df = pd.read_csv(args.csvfile)
    df["kernel_name"] = df["kernel_name"].astype(str)

    # Extract base name (with integer, without hash)
    df["base_name"] = df["kernel_name"].apply(get_base_name)

    # ---- Group by base_name (includes plain + hashed) ---------------------
    stats = (
        df.groupby("base_name")["ms_per_call"]
        .agg(
            p10=lambda x: np.percentile(x, 10),
            mean="mean",
            p90=lambda x: np.percentile(x, 90),
            count="count",      # <- counts all rows: base + hashes
        )
        .reset_index()
        .sort_values("base_name")
    )

    # ---- Add p90/p10 ratio ------------------------------------------------
    stats["p90_over_p10"] = stats["p90"] / stats["p10"]

    # ---- Output -----------------------------------------------------------
    if args.output:
        outdir = os.path.dirname(args.output)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        stats.to_csv(args.output, index=False)
    else:
        print(stats.to_string(index=False))


if __name__ == "__main__":
    main()