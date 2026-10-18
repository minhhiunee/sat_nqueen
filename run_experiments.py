import os
import sys
import json
import math
import time
import platform
import tracemalloc
import multiprocessing as mp
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pulp
import pysat

from nqueens_core import (
    get_peak_memory_mb,
    solve_nqueens_pysat,
    solve_nqueens_cpsat,
    verify_nqueens_solution,
    run_unit_tests,
)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

ENCODE_LIMIT_SEC = 15.0
SOLVE_LIMIT_SEC = 15.0
PAR2_PENALTY = 2.0 * SOLVE_LIMIT_SEC


def compute_par2(time_val: float, status: str, limit: float = SOLVE_LIMIT_SEC) -> float:
    if status in ("SAT", "UNSAT") and time_val <= limit:
        return float(time_val)
    return float(2.0 * limit)


def save_environment_metadata():
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "pysat_version": getattr(pysat, "__version__", "unknown"),
        "pulp_version": getattr(pulp, "__version__", "unknown"),
        "encode_limit_sec": ENCODE_LIMIT_SEC,
        "solve_limit_sec": SOLVE_LIMIT_SEC,
        "par2_penalty_sec": PAR2_PENALTY,
    }
    with open(os.path.join(RESULTS_DIR, "environment_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def solve_nqueens_ilp(n: int, time_limit: float = SOLVE_LIMIT_SEC, use_symmetry: bool = True):
    tracemalloc.start()
    t0 = time.perf_counter()
    prob = pulp.LpProblem(f"NQueens_{n}", pulp.LpMinimize)

    x = [[pulp.LpVariable(f"x_{r}_{c}", cat=pulp.LpBinary) for c in range(n)] for r in range(n)]
    prob += 0

    for r in range(n):
        prob += pulp.lpSum(x[r][c] for c in range(n)) == 1

    for c in range(n):
        prob += pulp.lpSum(x[r][c] for r in range(n)) == 1

    for diff in range(-(n - 2), n - 1):
        diag1 = [x[r][r - diff] for r in range(n) if 0 <= r - diff < n]
        if len(diag1) >= 2:
            prob += pulp.lpSum(diag1) <= 1

    for s in range(1, 2 * n - 2):
        diag2 = [x[r][s - r] for r in range(n) if 0 <= s - r < n]
        if len(diag2) >= 2:
            prob += pulp.lpSum(diag2) <= 1

    if use_symmetry and n >= 4:
        half = math.ceil(n / 2)
        for c in range(half, n):
            prob += x[0][c] == 0

    encode_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    load_time = time.perf_counter() - t1

    t2 = time.perf_counter()
    prob.solve(solver)
    pure_solve_time = time.perf_counter() - t2
    peak_mem = get_peak_memory_mb()
    tracemalloc.stop()

    queens = []
    for r in range(n):
        for c in range(n):
            val = pulp.value(x[r][c])
            if val is not None and val > 0.5:
                queens.append((r, c))

    checker_passed, _ = verify_nqueens_solution(n, queens)

    if checker_passed and (load_time + pure_solve_time) <= time_limit:
        status_str = "SAT"
        solve_time = load_time + pure_solve_time
    elif prob.status == pulp.LpStatusInfeasible:
        status_str = "UNSAT"
        solve_time = load_time + pure_solve_time
    else:
        status_str = "SOLVE_TIMEOUT"
        solve_time = float(time_limit)
        pure_solve_time = max(0.0, solve_time - load_time)
        checker_passed = False

    return {
        "n": n,
        "paradigm": "ILP (MIP-CBC)",
        "symmetry": use_symmetry,
        "encode_time": encode_time,
        "load_time": load_time,
        "pure_solve_time": pure_solve_time,
        "solve_time": solve_time,
        "total_time": encode_time + solve_time,
        "peak_memory_mb": peak_mem,
        "status": status_str,
        "checker_passed": checker_passed,
        "cnf_builds": "N/A",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_exp1_amo_symmetry_matrix():
    raw_path = os.path.join(RESULTS_DIR, "raw_exp1_amo_sym_matrix.csv")
    existing_df = pd.read_csv(raw_path) if os.path.exists(raw_path) else pd.DataFrame()

    n_list = [8, 16, 32, 50, 64, 80, 100, 128, 160, 200, 250, 300, 400]
    amo_methods = ["binomial", "binary", "seqcounter", "commander", "product"]
    records = existing_df.to_dict("records") if not existing_df.empty else []

    print("\n--- Running Exp 1: AMO Encodings and Symmetry Modes ---")
    for n in n_list:
        for sym in [True, False]:
            for m in amo_methods:
                already_done = any(
                    int(r["n"]) == n and str(r["method"]) == m and bool(r["symmetry"]) == sym
                    for r in records
                )
                if already_done:
                    continue

                res = solve_nqueens_pysat(
                    n=n,
                    amo_method=m,
                    solver_name="cadical195",
                    use_symmetry=sym,
                    encode_limit=ENCODE_LIMIT_SEC,
                    solve_limit=SOLVE_LIMIT_SEC,
                )
                res.pop("queens", None)
                res.pop("stats", None)
                res["solve_par2"] = compute_par2(res["solve_time"], res["status"], SOLVE_LIMIT_SEC)
                res["total_par2"] = compute_par2(res["total_time"], res["status"], ENCODE_LIMIT_SEC + SOLVE_LIMIT_SEC)
                records.append(res)

                pd.DataFrame(records).to_csv(raw_path, index=False)
                sym_tag = "SymON " if sym else "SymOFF"
                print(
                    f"  N={n:3d} | {m:10s} | {sym_tag} | Enc={res['encode_time']:6.3f}s | "
                    f"Load={res['load_time']:5.3f}s | Solve={res['pure_solve_time']:6.3f}s | "
                    f"Mem={res['peak_memory_mb']:6.1f}MB | {res['status']} | Check={res['checker_passed']}"
                )

    return pd.DataFrame(records)


def run_exp2_solvers_and_stability():
    raw_path = os.path.join(RESULTS_DIR, "raw_exp2_solvers_stability.csv")
    existing_df = pd.read_csv(raw_path) if os.path.exists(raw_path) else pd.DataFrame()
    records = existing_df.to_dict("records") if not existing_df.empty else []

    n_list = [16, 32, 64, 100, 128, 160, 200, 250, 300]
    repeat_n_set = {32, 64, 100, 128}

    print("\n--- Running Exp 2: Solver Comparison & Stability ---")
    for n in n_list:
        num_reps = 3 if n in repeat_n_set else 1
        for solver_name in ["cadical195", "glucose3"]:
            for sym in [True, False]:
                for rep in range(1, num_reps + 1):
                    already_done = any(
                        int(r["n"]) == n
                        and str(r["solver"]) == solver_name
                        and bool(r["symmetry"]) == sym
                        and int(r.get("repeat", 1)) == rep
                        for r in records
                    )
                    if already_done:
                        continue

                    res = solve_nqueens_pysat(
                        n=n,
                        amo_method="product",
                        solver_name=solver_name,
                        use_symmetry=sym,
                        encode_limit=ENCODE_LIMIT_SEC,
                        solve_limit=SOLVE_LIMIT_SEC,
                    )
                    res.pop("queens", None)
                    res.pop("stats", None)
                    res["repeat"] = rep
                    res["config"] = f"{solver_name}_{'SymON' if sym else 'SymOFF'}"
                    res["solve_par2"] = compute_par2(res["solve_time"], res["status"], SOLVE_LIMIT_SEC)
                    records.append(res)
                    pd.DataFrame(records).to_csv(raw_path, index=False)
                    print(
                        f"  N={n:3d} | {res['config']:17s} | Rep={rep} | "
                        f"Solve={res['solve_time']:6.3f}s | Total={res['total_time']:6.3f}s | "
                        f"{res['status']} | Check={res['checker_passed']}"
                    )

    return pd.DataFrame(records)


def run_exp3_cross_paradigm(df_exp1: pd.DataFrame):
    raw_path = os.path.join(RESULTS_DIR, "raw_exp3_paradigms.csv")
    existing_df = pd.read_csv(raw_path) if os.path.exists(raw_path) else pd.DataFrame()
    records = existing_df.to_dict("records") if not existing_df.empty else []

    n_list = [8, 16, 32, 50, 64, 80, 100, 128, 160, 200]

    print("\n--- Running Exp 3: SAT vs CP-SAT vs ILP ---")
    for n in n_list:
        for sym in [True, False]:
            for sat_m in ["commander", "product"]:
                p_name = f"SAT-{sat_m.capitalize()} (CaDiCaL)"
                if not any(int(r["n"]) == n and str(r["paradigm"]) == p_name and bool(r["symmetry"]) == sym for r in records):
                    sub = df_exp1[
                        (df_exp1["n"] == n) & (df_exp1["method"] == sat_m) & (df_exp1["symmetry"] == sym)
                    ]
                    if not sub.empty:
                        row = sub.iloc[0]
                        records.append({
                            "n": n,
                            "paradigm": p_name,
                            "symmetry": sym,
                            "encode_time": float(row["encode_time"]),
                            "load_time": float(row["load_time"]),
                            "pure_solve_time": float(row["pure_solve_time"]),
                            "solve_time": float(row["solve_time"]),
                            "total_time": float(row["total_time"]),
                            "peak_memory_mb": float(row["peak_memory_mb"]),
                            "solve_par2": float(row["solve_par2"]),
                            "status": str(row["status"]),
                            "checker_passed": bool(row["checker_passed"]),
                            "cnf_builds": "1",
                            "timestamp_utc": str(row["timestamp_utc"]),
                        })

            for p_name, fn in [("CP-SAT (OR-Tools)", solve_nqueens_cpsat), ("ILP (MIP-CBC)", solve_nqueens_ilp)]:
                if any(int(r["n"]) == n and str(r["paradigm"]) == p_name and bool(r["symmetry"]) == sym for r in records):
                    continue

                res = fn(n=n, time_limit=SOLVE_LIMIT_SEC, use_symmetry=sym)
                res.pop("queens", None)
                res["solve_par2"] = compute_par2(res["solve_time"], res["status"], SOLVE_LIMIT_SEC)
                records.append(res)
                pd.DataFrame(records).to_csv(raw_path, index=False)
                sym_tag = "SymON " if sym else "SymOFF"
                print(
                    f"  N={n:3d} | {p_name:20s} | {sym_tag} | Enc={res['encode_time']:6.3f}s | "
                    f"Solve={res['solve_time']:6.3f}s | Mem={res['peak_memory_mb']:6.1f}MB | "
                    f"{res['status']} | Check={res['checker_passed']}"
                )

    df3 = pd.DataFrame(records)
    df3.to_csv(raw_path, index=False)
    return df3


def generate_plots(df_exp1: pd.DataFrame, df_exp2: pd.DataFrame, df_exp3: pd.DataFrame):
    plt.rcParams.update({"font.size": 10.5, "figure.autolayout": True})
    markers = {"binomial": "o", "binary": "s", "seqcounter": "^", "commander": "D", "product": "*"}
    colors = {
        "binomial": "#1f77b4",
        "binary": "#ff7f0e",
        "seqcounter": "#2ca02c",
        "commander": "#d62728",
        "product": "#9467bd",
    }
    labels = {
        "binomial": "Binomial",
        "binary": "Binary",
        "seqcounter": "SeqCounter",
        "commander": "Commander",
        "product": "Product",
    }

    # Plot 1: Clauses, variables, and peak memory
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.8))
    sub_sym_on = df_exp1[df_exp1["symmetry"] == True]
    for m in labels:
        sub = sub_sym_on[sub_sym_on["method"] == m].sort_values("n")
        ax1.plot(sub["n"], sub["clauses"], marker=markers[m], color=colors[m], linewidth=2, label=labels[m])
        ax2.plot(sub["n"], sub["vars"], marker=markers[m], color=colors[m], linewidth=2, label=labels[m])
        ax3.plot(sub["n"], sub["peak_memory_mb"], marker=markers[m], color=colors[m], linewidth=2, label=labels[m])

    ax1.set_yscale("log")
    ax1.set_xlabel("Board Size (N)")
    ax1.set_ylabel("Number of Clauses (log scale)")
    ax1.set_title("(a) CNF Clause Growth")
    ax1.grid(True, which="both", linestyle="--", alpha=0.6)
    ax1.legend(fontsize=8.5)

    ax2.set_yscale("log")
    ax2.set_xlabel("Board Size (N)")
    ax2.set_ylabel("Number of Variables (log scale)")
    ax2.set_title("(b) CNF Variable Growth")
    ax2.grid(True, which="both", linestyle="--", alpha=0.6)
    ax2.legend(fontsize=8.5)

    ax3.set_xlabel("Board Size (N)")
    ax3.set_ylabel("Peak Memory (MB)")
    ax3.set_title("(c) Peak Memory Usage")
    ax3.grid(True, which="both", linestyle="--", alpha=0.6)
    ax3.legend(fontsize=8.5)

    plt.savefig(os.path.join(RESULTS_DIR, "fig1_clauses.pdf"))
    plt.savefig(os.path.join(RESULTS_DIR, "fig1_clauses.png"), dpi=300)
    plt.close()

    # Plot 2: Encodings and symmetry breaking
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    for m in labels:
        for sym, ls, alpha_val in [(True, "-", 1.0), (False, "--", 0.75)]:
            sub = df_exp1[(df_exp1["method"] == m) & (df_exp1["symmetry"] == sym)].sort_values("n")
            sat_sub = sub[(sub["status"] == "SAT") & (sub["checker_passed"] == True)]
            tag = f"{labels[m]} ({'Sym ON' if sym else 'Sym OFF'})"

            ax1.plot(
                sat_sub["n"],
                sat_sub["solve_time"],
                marker=markers[m],
                color=colors[m],
                linestyle=ls,
                alpha=alpha_val,
                linewidth=1.8,
                label=tag,
            )

            solved_times = np.sort(sat_sub["solve_time"].values)
            if len(solved_times) > 0:
                ax2.plot(
                    range(1, len(solved_times) + 1),
                    solved_times,
                    marker=markers[m],
                    color=colors[m],
                    linestyle=ls,
                    alpha=alpha_val,
                    linewidth=1.8,
                    label=f"{tag} [{len(solved_times)}/13]",
                )

    ax1.axhline(SOLVE_LIMIT_SEC, color="black", linestyle=":", linewidth=1.2, label="15s Solve Cutoff")
    ax1.set_yscale("log")
    ax1.set_xlabel("Board Size (N)")
    ax1.set_ylabel("Solver Time (s, log scale)")
    ax1.set_title("(a) AMO Encoding x Symmetry Breaking Runtime")
    ax1.grid(True, which="both", linestyle="--", alpha=0.6)
    ax1.legend(fontsize=7.5, ncol=2)

    ax2.axhline(SOLVE_LIMIT_SEC, color="black", linestyle=":", linewidth=1.2, label="15s Solve Cutoff")
    ax2.set_yscale("log")
    ax2.set_xlim(0.5, 13.5)
    ax2.set_xticks(range(1, 14))
    ax2.set_xlabel("Number of Verified Solved Instances (out of 13)")
    ax2.set_ylabel("Solver Time (s, log scale)")
    ax2.set_title("(b) Cactus Plot: 5 AMO Encodings x 2 Symmetry Modes")
    ax2.grid(True, which="both", linestyle="--", alpha=0.6)
    ax2.legend(fontsize=7.5, ncol=2, loc="upper left")

    plt.savefig(os.path.join(RESULTS_DIR, "fig3_runtime_encodings.pdf"))
    plt.savefig(os.path.join(RESULTS_DIR, "fig3_runtime_encodings.png"), dpi=300)
    plt.close()

    # Plot 3: Time breakdown
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    for ax, target_n in [(ax1, 160), (ax2, 300)]:
        sub_n = df_exp1[(df_exp1["n"] == target_n) & (df_exp1["symmetry"] == True)]
        methods_order = ["binomial", "binary", "seqcounter", "commander", "product"]
        enc_vals, load_vals, solve_vals = [], [], []
        for m in methods_order:
            r = sub_n[sub_n["method"] == m].iloc[0]
            enc_vals.append(float(r["encode_time"]))
            load_vals.append(float(r["load_time"]))
            solve_vals.append(float(r["pure_solve_time"]))

        x_pos = np.arange(len(methods_order))
        ax.bar(x_pos, enc_vals, label="CNF Encode Time", color="#4c72b0")
        ax.bar(x_pos, load_vals, bottom=enc_vals, label="Solver Load Time", color="#dd8452")
        bottom_solve = np.array(enc_vals) + np.array(load_vals)
        ax.bar(x_pos, solve_vals, bottom=bottom_solve, label="Pure CDCL Solve Time", color="#55a868")

        ax.set_xticks(x_pos)
        ax.set_xticklabels([labels[m] for m in methods_order])
        ax.set_ylabel("Time (seconds)")
        ax.set_title(f"Pipeline Cost Breakdown at N = {target_n} (Sym ON)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.6)
        ax.legend(fontsize=9)

    plt.savefig(os.path.join(RESULTS_DIR, "fig2_variables.pdf"))
    plt.savefig(os.path.join(RESULTS_DIR, "fig2_variables.png"), dpi=300)
    plt.close()

    # Plot 4: Paradigm comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    paradigm_styles = {
        "SAT-Product (CaDiCaL)": ("*", "#9467bd"),
        "SAT-Commander (CaDiCaL)": ("D", "#d62728"),
        "CP-SAT (OR-Tools)": ("s", "#2ca02c"),
        "ILP (MIP-CBC)": ("^", "#1f77b4"),
    }
    to_labeled_a = False
    to_labeled_b = False
    sub_exp3_sym = df_exp3[df_exp3["symmetry"] == True]
    for p_name, (mk, col) in paradigm_styles.items():
        sub = sub_exp3_sym[sub_exp3_sym["paradigm"] == p_name].sort_values("n")
        sat_sub = sub[(sub["status"] == "SAT") & (sub["checker_passed"] == True)]
        to_sub = sub[sub["status"].isin(["SOLVE_TIMEOUT", "ENCODE_TIMEOUT"])]

        ax1.plot(sat_sub["n"], sat_sub["solve_time"], marker=mk, color=col, linewidth=2, label=p_name)
        ax2.plot(sat_sub["n"], sat_sub["total_time"], marker=mk, color=col, linewidth=2, label=p_name)

        if not to_sub.empty:
            ax1.scatter(
                to_sub["n"],
                [SOLVE_LIMIT_SEC] * len(to_sub),
                marker="x",
                s=80,
                color=col,
                zorder=5,
                label="Solver Cutoff (15s)" if not to_labeled_a else None,
            )
            to_labeled_a = True
            ax2.scatter(
                to_sub["n"],
                to_sub["total_time"],
                marker="x",
                s=80,
                color=col,
                zorder=5,
                label="Run with Timeout (encode + 15s)" if not to_labeled_b else None,
            )
            to_labeled_b = True

    ax1.axhline(SOLVE_LIMIT_SEC, color="black", linestyle="--", linewidth=1.2, label=f"Solver Cutoff ({SOLVE_LIMIT_SEC}s)")
    ax1.set_yscale("log")
    ax1.set_xlabel("Board Size (N)")
    ax1.set_ylabel("Solver Time (s, log scale)")
    ax1.set_title("(a) Pure Solver Time (Matched Symmetry ON)")
    ax1.grid(True, which="both", linestyle="--", alpha=0.6)
    ax1.legend(fontsize=8.5)

    ax2.set_yscale("log")
    ax2.set_xlabel("Board Size (N)")
    ax2.set_ylabel("End-to-End Total Time: Encode + Solve (s, log scale)")
    ax2.set_title("(b) End-to-End Total Time (Includes Encoding Cost)")
    ax2.grid(True, which="both", linestyle="--", alpha=0.6)
    ax2.legend(fontsize=8.5)

    plt.savefig(os.path.join(RESULTS_DIR, "fig4_sat_vs_ilp_cp.pdf"))
    plt.savefig(os.path.join(RESULTS_DIR, "fig4_sat_vs_ilp_cp.png"), dpi=300)
    plt.close()

    print("\nResults and figures saved in 'results/'.")


def print_summary_tables(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame):
    print("\n=== SUMMARY 1: AMO ENCODINGS & SYMMETRY ===")
    for m in ["binomial", "binary", "seqcounter", "commander", "product"]:
        for sym in [True, False]:
            sub = df1[(df1["method"] == m) & (df1["symmetry"] == sym)]
            solved = int(((sub["status"] == "SAT") & (sub["checker_passed"] == True)).sum())
            encto = int((sub["status"] == "ENCODE_TIMEOUT").sum())
            solveto = int((sub["status"] == "SOLVE_TIMEOUT").sum())
            par2 = float(sub["solve_par2"].mean())
            max_mem = float(sub["peak_memory_mb"].max())
            print(
                f"  {m:10s} | Sym={'ON ' if sym else 'OFF'} | Solved={solved:2d}/13 | "
                f"EncTO={encto} | SolveTO={solveto} | Solve_PAR2={par2:6.2f}s | MaxMem={max_mem:6.1f}MB"
            )

    print("\n=== SUMMARY 2: REPEATED RUNS STABILITY ===")
    rep_df = df2[df2["n"].isin([32, 64, 100, 128])]
    for (n_val, cfg), grp in rep_df.groupby(["n", "config"]):
        med = float(grp["solve_time"].median())
        iqr = float(grp["solve_time"].quantile(0.75) - grp["solve_time"].quantile(0.25))
        print(f"  N={n_val:3d} | {cfg:17s} | Median Solve={med:7.4f}s | IQR={iqr:7.4f}s")

    print("\n=== SUMMARY 3: PARADIGM COMPARISON ===")
    for p_name in ["SAT-Commander (CaDiCaL)", "SAT-Product (CaDiCaL)", "CP-SAT (OR-Tools)", "ILP (MIP-CBC)"]:
        for sym in [True, False]:
            sub = df3[(df3["paradigm"] == p_name) & (df3["symmetry"] == sym)]
            solved = int(((sub["status"] == "SAT") & (sub["checker_passed"] == True)).sum())
            par2 = float(sub["solve_par2"].mean())
            
            if "SAT-" in p_name:
                max_mem = f"{float(sub['peak_memory_mb'].max()):6.1f}MB"
            else:
                max_mem = "   N/A (Out-of-process)"

            print(
                f"  {p_name:24s} | Sym={'ON ' if sym else 'OFF'} | "
                f"Solved={solved:2d}/10 | Solve_PAR2={par2:6.2f}s | MaxMem={max_mem}"
            )


if __name__ == "__main__":
    mp.freeze_support()
    save_environment_metadata()
    run_unit_tests()
    df_exp1 = run_exp1_amo_symmetry_matrix()
    df_exp2 = run_exp2_solvers_and_stability()
    df_exp3 = run_exp3_cross_paradigm(df_exp1)
    generate_plots(df_exp1, df_exp2, df_exp3)
    print_summary_tables(df_exp1, df_exp2, df_exp3)