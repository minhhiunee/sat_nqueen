import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nqueens_core import theoretical_binomial_counts, solve_nqueens_pysat

RESULTS_DIR = "results"
ENCODE_LIMIT_SEC = 15.0
SOLVE_LIMIT_SEC = 15.0

def clean_and_interpolate_data():
    df1_path = os.path.join(RESULTS_DIR, "raw_exp1_amo_sym_matrix.csv")
    if not os.path.exists(df1_path):
        print(f"File not found: {df1_path}")
        return None
        
    df1 = pd.read_csv(df1_path)
    for idx, row in df1.iterrows():
        if row["status"] == "ENCODE_TIMEOUT" and row["method"] == "binomial":
            n = int(row["n"])
            sym = bool(row["symmetry"])
            v_est, c_est = theoretical_binomial_counts(n, sym)
            df1.at[idx, "vars"] = v_est
            df1.at[idx, "clauses"] = c_est
            
    df1.to_csv(df1_path, index=False)
    return df1

def targeted_rerun_n300_with_persistence():
    print(f"\nRunning 5 repetitions for Product N=300 (limit={SOLVE_LIMIT_SEC}s)...")
    records = []
    
    for sym in [True, False]:
        sat_times = []
        for i in range(5):
            gc.collect()
            res = solve_nqueens_pysat(n=300, amo_method="product", solver_name="cadical195", 
                                      use_symmetry=sym, encode_limit=ENCODE_LIMIT_SEC, solve_limit=SOLVE_LIMIT_SEC)
            
            res.pop("queens", None)
            res.pop("stats", None)
            res["run_index"] = i + 1
            records.append(res)
            
            if res["status"] == "SAT":
                sat_times.append(res['pure_solve_time'])
                print(f"    Sym {'ON ' if sym else 'OFF'} | Run {i+1}/5 | Solve: {res['pure_solve_time']:.4f}s | SAT")
            else:
                print(f"    Sym {'ON ' if sym else 'OFF'} | Run {i+1}/5 | Status: {res['status']}")
        
        if sat_times:
            med = np.median(sat_times)
            iqr = np.percentile(sat_times, 75) - np.percentile(sat_times, 25)
            std_sample = np.std(sat_times, ddof=1) if len(sat_times) > 1 else 0.0
            print(f"  Summary: Sym {'ON ' if sym else 'OFF'} | Solved: {len(sat_times)}/5 | Median: {med:.4f}s | IQR: {iqr:.4f}s | StdDev: {std_sample:.4f}s\n")

    rerun_df = pd.DataFrame(records)
    rerun_csv_path = os.path.join(RESULTS_DIR, "targeted_reruns_product_n300.csv")
    rerun_df.to_csv(rerun_csv_path, index=False)
    print(f"Saved rerun data to {rerun_csv_path}")
    
    return rerun_df

def generate_transparent_plots(df1):
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})
    markers = {"binomial": "o", "binary": "s", "seqcounter": "^", "commander": "D", "product": "*"}
    colors = {"binomial": "#1f77b4", "binary": "#ff7f0e", "seqcounter": "#2ca02c", "commander": "#d62728", "product": "#9467bd"}
    
    # 1. Clauses
    plt.figure(figsize=(9, 6))
    df_sym = df1[df1["symmetry"] == True].copy()
    for m in markers:
        sub = df_sym[df_sym["method"] == m].sort_values("n")
        actual = sub[sub["status"] != "ENCODE_TIMEOUT"]
        
        if not actual.empty:
            plt.plot(actual["n"], actual["clauses"], marker=markers[m], color=colors[m], 
                     linewidth=2, label=f"{m.capitalize()} (Actual)")
            
        timeout_pts = sub[sub["status"] == "ENCODE_TIMEOUT"]
        if m == "binomial" and not timeout_pts.empty:
            theo_n = timeout_pts["n"].tolist()
            theo_c = [theoretical_binomial_counts(n, True)[1] for n in theo_n]
            if not actual.empty:
                theo_n = [actual["n"].iloc[-1]] + theo_n
                theo_c = [actual["clauses"].iloc[-1]] + theo_c
            plt.plot(theo_n, theo_c, marker=markers[m], color=colors[m], linestyle="--", 
                     markerfacecolor='none', linewidth=2, label=f"{m.capitalize()} (Theoretical)")

    plt.yscale("log")
    plt.xlabel("Board Size (N)")
    plt.ylabel("Number of Clauses (log scale)")
    plt.title("Formula Size Growth: Actual vs. Theoretical Bounds")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=9, ncol=2)
    plt.savefig(os.path.join(RESULTS_DIR, "fig1_clauses_transparent.pdf"))
    plt.close()

    # 2. AMO x Symmetry
    plt.figure(figsize=(11, 7))
    solve_to_labeled = False
    
    for m in ["seqcounter", "commander", "product"]: 
        for sym, ls in [(True, "-"), (False, "--")]:
            sub = df1[(df1["method"] == m) & (df1["symmetry"] == sym)].sort_values("n")
            
            sat_sub = sub[sub["status"] == "SAT"]
            plt.plot(sat_sub["n"], sat_sub["pure_solve_time"], marker=markers[m], color=colors[m], 
                     linestyle=ls, linewidth=2, label=f"{m.capitalize()} (Sym {'ON' if sym else 'OFF'})")
            
            solve_to = sub[sub["status"] == "SOLVE_TIMEOUT"]
            if not solve_to.empty:
                plt.scatter(solve_to["n"], [SOLVE_LIMIT_SEC]*len(solve_to), marker="x", s=100, color="red", zorder=5,
                            label="SOLVE_TIMEOUT (Search Failed)" if not solve_to_labeled else None)
                solve_to_labeled = True

    plt.axhline(SOLVE_LIMIT_SEC, color="black", linestyle=":", label="15.0s Search Cutoff")
    plt.yscale("log")
    plt.xlabel("Board Size (N)")
    plt.ylabel("Pure CDCL Solve Time (s, log scale)")
    plt.title("AMO Encodings vs Symmetry (Excluding ENCODE_TIMEOUTs)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(ncol=3, fontsize=9, loc="upper left", bbox_to_anchor=(0.0, -0.15))
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig3_interaction_transparent.pdf"))
    plt.close()
    print("Saved plots to results/")

def build_comprehensive_stability_summary(rerun_df):
    df2_path = os.path.join(RESULTS_DIR, "raw_exp2_solvers_stability.csv")
    summary = []
    
    if os.path.exists(df2_path):
        df2 = pd.read_csv(df2_path)
        for (n, config), grp in df2.groupby(["n", "config"]):
            sat_runs = grp[grp["status"] == "SAT"]
            med_solve = sat_runs["pure_solve_time"].median() if not sat_runs.empty else np.nan
            iqr_solve = np.percentile(sat_runs["pure_solve_time"], 75) - np.percentile(sat_runs["pure_solve_time"], 25) if not sat_runs.empty else np.nan
            std_solve = sat_runs["pure_solve_time"].std(ddof=1) if len(sat_runs) > 1 else 0.0
            
            summary.append({
                "Data_Source": "Exp2_Stability",
                "N": n,
                "Config": config,
                "Total_Runs": len(grp),
                "Success_Rate": f"{len(sat_runs)}/{len(grp)}",
                "Median_Solve_s": med_solve,
                "IQR_Solve_s": iqr_solve,
                "Sample_StdDev_s": std_solve
            })
            
    if rerun_df is not None:
        for sym in [True, False]:
            grp = rerun_df[rerun_df["symmetry"] == sym]
            sat_runs = grp[grp["status"] == "SAT"]
            med_solve = sat_runs["pure_solve_time"].median() if not sat_runs.empty else np.nan
            iqr_solve = np.percentile(sat_runs["pure_solve_time"], 75) - np.percentile(sat_runs["pure_solve_time"], 25) if not sat_runs.empty else np.nan
            std_solve = sat_runs["pure_solve_time"].std(ddof=1) if len(sat_runs) > 1 else 0.0
            
            summary.append({
                "Data_Source": "Targeted_Rerun_N300",
                "N": 300,
                "Config": f"cadical195_product_Sym{'ON' if sym else 'OFF'}",
                "Total_Runs": len(grp),
                "Success_Rate": f"{len(sat_runs)}/{len(grp)}",
                "Median_Solve_s": med_solve,
                "IQR_Solve_s": iqr_solve,
                "Sample_StdDev_s": std_solve
            })

    stab_df = pd.DataFrame(summary)
    out_path = os.path.join(RESULTS_DIR, "summary_stability_comprehensive.csv")
    stab_df.to_csv(out_path, index=False)
    print(f"Saved stability summary to {out_path}")

if __name__ == "__main__":
    df1 = clean_and_interpolate_data()
    if df1 is not None:
        generate_transparent_plots(df1)
    rerun_df = targeted_rerun_n300_with_persistence()
    build_comprehensive_stability_summary(rerun_df)