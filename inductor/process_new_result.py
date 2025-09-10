import pandas as pd
import re

def strip_suffix(name: str) -> str:
    """Remove trailing _<digits> from kernel name."""
    return re.sub(r'_\d+$', '', str(name))

def assign_pattern(kernel_name):
    """Assign kernel_name to one of the four groups based on prefix."""
    kname = str(kernel_name)  # force to string
    if kname.startswith("triton_poi_"):
        return "triton_poi"
    elif kname.startswith("triton_red_"):
        return "triton_red"
    elif kname.startswith("triton_per_"):
        return "triton_per"
    elif kname.startswith("triton_for_"):
        return "triton_for"
    else:
        return "other"

def process_csv(csv1, csv2, threshold=100, output="test.csv"):
    # Load CSVs
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    # Filter rows above threshold based on sum_latency_mi350x
    filtered = df1[df1["sum_latency_mi350x"] > threshold].copy()
    filtered["avg_latency_mi350x"] = filtered["avg_latency_mi350x"] / 1000

    # Create base name for fallback matching
    filtered["kernel_base"] = filtered["s_name"].apply(strip_suffix)
    df2["kernel_base"] = df2["kernel_name"].apply(strip_suffix)

    results = []
    matched_snames = set()
    used_kernels = set()

    for _, row in filtered.iterrows():
        s_name = row["s_name"]

        # ---------- Try exact match ----------
        exact = df2[df2["kernel_name"] == s_name]
        if not exact.empty:
            match = exact.iloc[0]
            results.append({**row.to_dict(), **match.to_dict()})
            matched_snames.add(s_name)
            used_kernels.add(match["kernel_name"])
            continue

        # ---------- Try fallback match ----------
        base = row["kernel_base"]
        fallback = df2[(df2["kernel_base"] == base) & (~df2["kernel_name"].isin(used_kernels))]
        if not fallback.empty:
            match = fallback.iloc[0]
            results.append({**row.to_dict(), **match.to_dict()})
            matched_snames.add(s_name)
            used_kernels.add(match["kernel_name"])

    merged = pd.DataFrame(results)

    # ---------- Count unmatched ----------
    total_filtered = len(filtered)
    matched_count = len(merged)
    unmatched_count = total_filtered - matched_count
    print(f"Number of s_name entries above threshold: {total_filtered}")
    print(f"Number of matched s_name entries: {matched_count}")
    print(f"Number of unmatched s_name entries: {unmatched_count}")

    # ---------- Compute totals ----------
    if not merged.empty:
        merged["total_latency"] = merged["count_mi350x"] * merged["avg_latency_mi350x"]
        total_sum_latency = merged["total_latency"].sum()

        merged["total_ms"] = merged["count_mi350x"] * merged["ms_per_call"]
        total_sum_ms = merged["total_ms"].sum()

        # Assign patterns and compute sums
        merged["pattern_group"] = merged["kernel_name"].apply(assign_pattern)
        pattern_sums = merged.groupby("pattern_group")["total_ms"].sum().reset_index()

        # Sort results by sum_latency_mi350x
        merged = merged.sort_values("sum_latency_mi350x", ascending=False)

        result = merged[[
            "s_name",
            "avg_latency_mi350x",
            "count_mi350x",
            "kernel_name",
            "ms_per_call"
        ]]
        result.to_csv(output, index=False)

        print(f"Saved results to {output}")
        print(f"Sum of all count * avg_latency_mi350x: {total_sum_latency}")
        print(f"Sum of all count * ms_per_call: {total_sum_ms}")
        print("Total ms per pattern group:")
        print(pattern_sums)
    else:
        result = pd.DataFrame()
        total_sum_latency = 0
        total_sum_ms = 0
        pattern_sums = pd.DataFrame()

    return result, total_sum_latency, total_sum_ms, pattern_sums, unmatched_count

# Example usage
if __name__ == "__main__":
    result_df, total_latency, total_ms, pattern_sums, unmatched_count = process_csv("model2-b200-mi350x-poiautotune-0828.csv", "kernels-cache-0828_autotune_poi/triton_results_20250909_rocm7rc1_autotune_pointwise.csv", threshold=1.0)