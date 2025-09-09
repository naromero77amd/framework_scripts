import pandas as pd
import re

def strip_suffix(name: str) -> str:
    """Remove trailing _<digits> from kernel name."""
    return re.sub(r'_\d+$', '', str(name))

def assign_pattern(kernel_name):
    """Assign kernel_name to one of the four groups based on prefix."""
    if kernel_name.startswith("triton_poi_"):
        return "triton_poi"
    elif kernel_name.startswith("triton_red_"):
        return "triton_red"
    elif kernel_name.startswith("triton_per_"):
        return "triton_per"
    elif kernel_name.startswith("triton_for_"):
        return "triton_for"
    else:
        return "other"

def process_csv(csv1, csv2, threshold=100.0, output="comparison_0828_poiautotune_internal_triton_meta.csv"):
    # Load CSVs
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    # Filter rows in first CSV above threshold
    filtered = df1[df1["sum_latency_mi350x"] > threshold].copy()

    # Convert avg_latency_mi350x to new units by dividing by 1000
    filtered["avg_latency_mi350x"] = filtered["avg_latency_mi350x"] / 1000

    # ---------- Step 1: Exact match ----------
    exact_merged = pd.merge(
        filtered, df2,
        left_on="s_name", right_on="kernel_name",
        how="inner"
    )

    matched_exact_snames = set(exact_merged["s_name"])
    used_kernel_names = set(exact_merged["kernel_name"])

    # ---------- Step 2: Fallback to base name ----------
    filtered["kernel_base"] = filtered["s_name"].apply(strip_suffix)
    df2["kernel_base"] = df2["kernel_name"].apply(strip_suffix)

    remaining = filtered[~filtered["s_name"].isin(matched_exact_snames)]

    base_merged = pd.merge(
        remaining, df2,
        on="kernel_base",
        how="inner"
    )


    # Keep only first fallback per s_name and avoid reusing kernel_name
    base_merged = base_merged.sort_values("s_name")
    base_unique = []
    used_kernels = set(used_kernel_names)
    used_snames = set(matched_exact_snames)

    for _, row in base_merged.iterrows():
        if row["s_name"] not in used_snames and row["kernel_name"] not in used_kernels:
            base_unique.append(row)
            used_snames.add(row["s_name"])
            used_kernels.add(row["kernel_name"])

    base_merged = pd.DataFrame(base_unique)

    # ---------- Combine results ----------
    merged = pd.concat([exact_merged, base_merged], ignore_index=True)

    # ---------- Count unmatched s_name ----------
    total_filtered = len(filtered)
    matched_count = len(merged)
    unmatched_count = total_filtered - matched_count
    print(f"Number of s_name entries above threshold: {total_filtered}")
    print(f"Number of matched s_name entries: {matched_count}")
    print(f"Number of unmatched s_name entries: {unmatched_count}")

    # Compute total time based on avg_latency_mi350x
    merged["total_latency"] = merged["count_mi350x"] * merged["avg_latency_mi350x"]
    total_sum_latency = merged["total_latency"].sum()

    # Compute total time based on ms_per_call
    merged["total_ms"] = merged["count_mi350x"] * merged["ms_per_call"]
    # merged["total_ms"] = merged["count_mi350x"] * merged["autotune_total_ms"]

    total_sum_ms = merged["total_ms"].sum()

    # ---------- Assign patterns ----------
    merged["pattern_group"] = merged["kernel_name"].apply(assign_pattern)

    # ---------- Group by pattern and sum total_ms ----------
    pattern_sums = merged.groupby("pattern_group")["total_ms"].sum().reset_index()
    print("Total ms per pattern group:")
    print(pattern_sums)


    # ---------- Sort by sum_latency_mi350x from CSV1 ----------
    merged = merged.sort_values("sum_latency_mi350x", ascending=False)

    # Select final columns for CSV (exclude computed columns)
    result = merged[[
        "s_name",
        "avg_latency_mi350x",
        "count_mi350x",
        "kernel_name",
        "ms_per_call"
    ]]
    # result = merged[[
    #     "s_name",
    #     "avg_latency_mi350x",
    #     "count_mi350x",
    #     "kernel_name",
    #     "autotune_total_ms"
    # ]]    

    # Save CSV
    result.to_csv(output, index=False)
    print(f"Saved results to {output}")
    print(f"Sum of all count * avg_latency_mi350x (data from spreadsheet): {total_sum_latency}")
    print(f"Sum of all count * ms_per_call: {total_sum_ms}")

    return result, total_sum_latency, total_sum_ms

# Example usage
if __name__ == "__main__":
    # autotune_perf.csv
    result_df = process_csv("model2-b200-mi350x-poiautotune-0828.csv", "kernels-cache-0828_autotune_poi/triton_results_20250908_162207.csv", threshold=1.0)

