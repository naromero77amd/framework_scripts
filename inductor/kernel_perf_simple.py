import pandas as pd
import numpy as np
import sys

def main(ref_csv, opt_csv, output_csv):
    threshold = 0.008 # microsecond

    # Read both CSVs
    ref_df = pd.read_csv(ref_csv)
    opt_df = pd.read_csv(opt_csv)

    # Drop rows missing any of the key columns
    required_cols = ['kernel_name', 'ms_per_call']
    ref_df = ref_df.dropna(subset=required_cols)
    opt_df = opt_df.dropna(subset=required_cols)

    # Filter: kernel_name starts with "triton_red_"
    ref_df = ref_df[ref_df['kernel_name'].str.match(r'^triton_poi_')]
    opt_df = opt_df[opt_df['kernel_name'].str.match(r'^triton_poi_')]

    # Keep only non-zero values in reference CSV
    ref_df = ref_df[
        (ref_df['ms_per_call'] > 0)
    ]

    # Filter out really small kernels as they could have a large variation
    ref_df = ref_df[
        (ref_df['ms_per_call'] > threshold)
    ]

    # Merge only on valid reference names (inner join)
    merged = pd.merge(ref_df, opt_df, on='kernel_name', suffixes=('_ref', '_opt'))

    # Compute ratios
    merged['latency_ratio'] = merged['ms_per_call_ref'] / merged['ms_per_call_opt']
    merged['delta_latency'] = merged['ms_per_call_ref'] - merged['ms_per_call_opt']

    # Filter: absolute difference must exceed 0.002 ms
    # merged = merged[merged['delta_latency'].abs() > threshold]

    # Select only required columns
    result = merged[['kernel_name', 'ms_per_call_ref', 'ms_per_call_opt', 'latency_ratio']]

    # Save to output CSV
    result.to_csv(output_csv, index=False)
    print(f"Saved output to {output_csv}")

    geo_mean = np.exp(np.log(result['latency_ratio']).mean())
    print(geo_mean)

    # from scipy.stats import gmean
    # print(gmean(result['latency_ratio']))

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python compare_latencies.py <reference.csv> <optimized.csv> <output.csv>")
        sys.exit(1)

    ref_csv = sys.argv[1]
    opt_csv = sys.argv[2]
    output_csv = sys.argv[3]
    main(ref_csv, opt_csv, output_csv)