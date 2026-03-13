import pandas as pd

def analyze_file(path):
    df = pd.read_csv(path)

    # Normalize kernel names
    df["Kernel Name"] = df["Kernel Name"].str.strip()

    # Total time across all kernels
    total_time = df["Total Device Time (us)"].sum()

    # Triton vs other
    triton_df = df[df["Kernel Name"].str.startswith("triton")]
    other_df  = df[~df["Kernel Name"].str.startswith("triton")]

    triton_total = triton_df["Total Device Time (us)"].sum()
    other_total  = other_df["Total Device Time (us)"].sum()

    # Triton subcategories
    categories = {
        "poi": triton_df[triton_df["Kernel Name"].str.startswith("triton_poi")]["Total Device Time (us)"].sum(),
        "tem": triton_df[triton_df["Kernel Name"].str.startswith("triton_tem")]["Total Device Time (us)"].sum(),
        "per": triton_df[triton_df["Kernel Name"].str.startswith("triton_per")]["Total Device Time (us)"].sum(),
        "red": triton_df[triton_df["Kernel Name"].str.startswith("triton_red")]["Total Device Time (us)"].sum(),
    }

    # Any triton kernel that is *not* one of the above
    categories["other_triton"] = (
        triton_total - sum(categories.values())
    )

    # Top 10 triton kernels by Total Device Time
    top10_triton = (
        triton_df.sort_values("Total Device Time (us)", ascending=False)
                 .head(10)
                 .reset_index(drop=True)
    )

    totals = {
        "total_time": total_time,
        "triton_total": triton_total,
        "other_total": other_total,
    }

    return totals, categories, top10_triton


def print_results(label, totals, cats, top10):
    print(f"\n===== {label} RESULTS =====")
    print(f"Total Time:     {totals['total_time']:.2f} us")
    print(f"  Triton Total: {totals['triton_total']:.2f} us")
    print(f"  Other Total:  {totals['other_total']:.2f} us\n")

    print("  Triton Breakdown:")
    for k, v in cats.items():
        print(f"    {k}: {v:.2f} us")
    print()

    print("\n  Top 10 Triton Kernels (by Total Device Time):")
    print(top10[["Kernel Name", "Total Device Time (us)"]].to_string(index=False))
    print()


if __name__ == "__main__":
    # nv_file = "H20_breakdown_decoder_10times_pyt29.csv"
    amd_file = "MI308_breakdown_decoder_10times_cuda_graph_1229-1.csv"
    # amd_file = "MI308_breakdown_decoder_10times_cuda_graph_restore_warp_mod.csv"
    # amd_file = "/root/xrec_decoder/MI308_breakdown_decoder_10times.csv"
    # nv_totals, nv_cats, nv_top10 = analyze_file(nv_file)
    amd_totals, amd_cats, amd_top10 = analyze_file(amd_file)

    #print_results("NVIDIA", nv_totals, nv_cats, nv_top10)
    print_results("AMD", amd_totals, amd_cats, amd_top10)
