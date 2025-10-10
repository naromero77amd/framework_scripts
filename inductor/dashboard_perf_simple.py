import pandas as pd
import sys

def main(ref_csv, opt_csv, output_csv):
    # Read both CSVs
    ref_df = pd.read_csv(ref_csv)
    opt_df = pd.read_csv(opt_csv)

    # Keep only non-zero values in reference CSV
    ref_df = ref_df[
        (ref_df['abs_latency'] != 0) &
        (ref_df['compilation_latency'] != 0)
    ]

    # Merge only on valid reference names (inner join)
    merged = pd.merge(ref_df, opt_df, on='Name', suffixes=('_ref', '_opt'))

    # Compute ratios
    merged['abs_latency_ratio'] = merged['abs_latency_ref'] / merged['abs_latency_opt']
    merged['compilation_latency_ratio'] = (
        merged['compilation_latency_ref'] / merged['compilation_latency_opt']
    )

    # Select only required columns
    result = merged[['Name', 'abs_latency_ratio', 'compilation_latency_ratio']]

    # Save to output CSV
    result.to_csv(output_csv, index=False)
    print(f"Saved output to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python compare_latencies.py <reference.csv> <optimized.csv> <output.csv>")
        sys.exit(1)

    ref_csv = sys.argv[1]
    opt_csv = sys.argv[2]
    output_csv = sys.argv[3]
    main(ref_csv, opt_csv, output_csv)