import os
import math
import time
import ctypes
from ctypes import wintypes
import tracemalloc
import multiprocessing as mp
from datetime import datetime, timezone
from pysat.formula import CNF, IDPool
from pysat.card import CardEnc, EncType
from pysat.solvers import Solver


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def get_peak_memory_mb() -> float:
    if os.name == "nt":
        try:
            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                return round(pmc.PeakWorkingSetSize / (1024.0 * 1024.0), 2)
        except Exception:
            pass
    if tracemalloc.is_tracing():
        _, peak = tracemalloc.get_traced_memory()
        return round(peak / (1024.0 * 1024.0), 2)
    return 0.0


def verify_nqueens_solution(n: int, queens: list) -> tuple[bool, str]:
    if not isinstance(queens, list) or len(queens) != n:
        return False, f"Expected {n} queens, got {len(queens) if isinstance(queens, list) else 0}"

    rows, cols, main_diags, anti_diags = set(), set(), set(), set()

    for r, c in queens:
        if not (0 <= r < n and 0 <= c < n):
            return False, f"Coordinate out of range: ({r}, {c})"
        if r in rows:
            return False, f"Row conflict at r={r}"
        if c in cols:
            return False, f"Column conflict at c={c}"
        if (r - c) in main_diags:
            return False, f"Main diagonal conflict at ({r}, {c})"
        if (r + c) in anti_diags:
            return False, f"Anti-diagonal conflict at ({r}, {c})"

        rows.add(r)
        cols.add(c)
        main_diags.add(r - c)
        anti_diags.add(r + c)

    return True, "VALID"


def encode_amo_binomial(lits: list, cnf: CNF):
    m = len(lits)
    for i in range(m):
        for j in range(i + 1, m):
            cnf.append([-lits[i], -lits[j]])


def encode_amo_binary(lits: list, cnf: CNF, vpool: IDPool):
    m = len(lits)
    if m <= 1:
        return
    k = math.ceil(math.log2(m))
    aux_vars = [vpool.id() for _ in range(k)]
    for idx, x in enumerate(lits):
        for bit_pos in range(k):
            if (idx >> bit_pos) & 1:
                cnf.append([-x, aux_vars[bit_pos]])
            else:
                cnf.append([-x, -aux_vars[bit_pos]])


def encode_amo_seqcounter(lits: list, cnf: CNF, vpool: IDPool):
    m = len(lits)
    if m <= 1:
        return
    if m == 2:
        cnf.append([-lits[0], -lits[1]])
        return
    sub_cnf = CardEnc.atmost(lits=lits, bound=1, vpool=vpool, encoding=EncType.seqcounter)
    cnf.extend(sub_cnf.clauses)


def encode_amo_commander(lits: list, cnf: CNF, vpool: IDPool):
    m = len(lits)
    if m <= 1:
        return
    if m <= 3:
        encode_amo_binomial(lits, cnf)
        return

    group_size = max(2, int(math.ceil(math.sqrt(m))))
    groups = [lits[i:i + group_size] for i in range(0, m, group_size)]
    commanders = [vpool.id() for _ in range(len(groups))]

    encode_amo_binomial(commanders, cnf)
    for g_idx, group in enumerate(groups):
        c_var = commanders[g_idx]
        encode_amo_binomial(group, cnf)
        for x in group:
            cnf.append([-x, c_var])
        cnf.append([-c_var] + group)


def encode_amo_product(lits: list, cnf: CNF, vpool: IDPool):
    m = len(lits)
    if m <= 1:
        return
    if m <= 3:
        encode_amo_binomial(lits, cnf)
        return

    p = math.ceil(math.sqrt(m))
    q = math.ceil(m / p)
    row_vars = [vpool.id() for _ in range(p)]
    col_vars = [vpool.id() for _ in range(q)]

    encode_amo_binomial(row_vars, cnf)
    encode_amo_binomial(col_vars, cnf)

    for idx, x in enumerate(lits):
        u = idx // q
        v = idx % q
        cnf.append([-x, row_vars[u]])
        cnf.append([-x, col_vars[v]])


def add_amo_constraint(lits: list, cnf: CNF, vpool: IDPool, method: str):
    if len(lits) <= 1:
        return
    method = method.lower().replace("sat-", "")
    if method == "binomial":
        encode_amo_binomial(lits, cnf)
    elif method == "binary":
        encode_amo_binary(lits, cnf, vpool)
    elif method == "seqcounter":
        encode_amo_seqcounter(lits, cnf, vpool)
    elif method == "commander":
        encode_amo_commander(lits, cnf, vpool)
    elif method == "product":
        encode_amo_product(lits, cnf, vpool)
    else:
        raise ValueError(f"Unsupported AMO encoding: {method}")


def theoretical_binomial_counts(n: int, use_symmetry: bool) -> tuple[int, int]:
    row_col_clauses = 2 * n + 2 * n * (n * (n - 1) // 2)
    diag_clauses = 0
    for d in range(-(n - 2), n - 1):
        m = n - abs(d)
        if m >= 2:
            diag_clauses += m * (m - 1) // 2
    diag_clauses *= 2
    sym_clauses = (n - math.ceil(n / 2)) if (use_symmetry and n >= 4) else 0
    return n * n, row_col_clauses + diag_clauses + sym_clauses


def build_nqueens_cnf(n: int, amo_method: str = "product", use_symmetry: bool = True) -> tuple[CNF, int]:
    cnf = CNF()
    vpool = IDPool(start_from=n * n + 1)

    def var_id(r: int, c: int) -> int:
        return r * n + c + 1

    for r in range(n):
        row_lits = [var_id(r, c) for c in range(n)]
        cnf.append(row_lits)
        add_amo_constraint(row_lits, cnf, vpool, amo_method)

    for c in range(n):
        col_lits = [var_id(r, c) for r in range(n)]
        cnf.append(col_lits)
        add_amo_constraint(col_lits, cnf, vpool, amo_method)

    for diff in range(-(n - 2), n - 1):
        diag1_lits = [var_id(r, r - diff) for r in range(n) if 0 <= r - diff < n]
        if len(diag1_lits) >= 2:
            add_amo_constraint(diag1_lits, cnf, vpool, amo_method)

    for s in range(1, 2 * n - 2):
        diag2_lits = [var_id(r, s - r) for r in range(n) if 0 <= s - r < n]
        if len(diag2_lits) >= 2:
            add_amo_constraint(diag2_lits, cnf, vpool, amo_method)

    if use_symmetry and n >= 4:
        half = math.ceil(n / 2)
        for c in range(half, n):
            cnf.append([-var_id(0, c)])

    total_vars = max(cnf.nv, n * n)
    return cnf, total_vars


def _pysat_worker(n: int, clean_method: str, solver_name: str, use_symmetry: bool, shared_meta, result_queue):
    try:
        tracemalloc.start()
        shared_meta["phase"] = "encoding"
        shared_meta["encode_started_at"] = time.perf_counter()

        t0 = time.perf_counter()
        cnf, total_vars = build_nqueens_cnf(n, amo_method=clean_method, use_symmetry=use_symmetry)
        encode_time = time.perf_counter() - t0
        num_clauses = len(cnf.clauses)

        shared_meta["vars"] = total_vars
        shared_meta["clauses"] = num_clauses
        shared_meta["encode_time"] = encode_time
        shared_meta["peak_memory_mb"] = get_peak_memory_mb()

        shared_meta["phase"] = "solving"
        shared_meta["solve_started_at"] = time.perf_counter()

        t1 = time.perf_counter()
        solver = Solver(name=solver_name, bootstrap_with=cnf.clauses, use_timer=True)
        load_time = time.perf_counter() - t1
        shared_meta["load_time"] = load_time
        shared_meta["peak_memory_mb"] = max(float(shared_meta["peak_memory_mb"]), get_peak_memory_mb())
        del cnf

        t2 = time.perf_counter()
        is_sat = solver.solve()
        pure_solve_time = time.perf_counter() - t2

        peak_mem = max(float(shared_meta["peak_memory_mb"]), get_peak_memory_mb())
        shared_meta["peak_memory_mb"] = peak_mem

        stats = solver.accum_stats()
        model = solver.get_model() if is_sat else None
        solver.delete()
        tracemalloc.stop()

        queens = []
        if is_sat and model is not None:
            pos_vars = {lit for lit in model[:n * n] if 0 < lit <= n * n}
            for lit in pos_vars:
                idx = lit - 1
                queens.append((idx // n, idx % n))
            queens.sort()

        checker_ok, checker_msg = verify_nqueens_solution(n, queens) if is_sat else (False, "UNSAT")
        shared_meta["phase"] = "done"

        result_queue.put({
            "ok": True,
            "is_sat": is_sat,
            "checker_passed": checker_ok,
            "checker_msg": checker_msg,
            "load_time": load_time,
            "pure_solve_time": pure_solve_time,
            "peak_memory_mb": peak_mem,
            "queens": queens,
            "stats": stats,
        })
    except Exception as exc:
        shared_meta["phase"] = "error"
        result_queue.put({"ok": False, "error": str(exc)})


def solve_nqueens_pysat(
    n: int,
    amo_method: str = "product",
    solver_name: str = "cadical195",
    use_symmetry: bool = True,
    encode_limit: float = 15.0,
    solve_limit: float = 15.0,
):
    clean_method = amo_method.lower().replace("sat-", "")
    manager = mp.Manager()
    shared_meta = manager.dict({
        "phase": "init",
        "vars": 0,
        "clauses": 0,
        "encode_time": 0.0,
        "load_time": 0.0,
        "peak_memory_mb": 0.0,
        "encode_started_at": time.perf_counter(),
        "solve_started_at": 0.0,
    })
    result_queue = manager.Queue()

    p = mp.Process(
        target=_pysat_worker,
        args=(n, clean_method, solver_name, use_symmetry, shared_meta, result_queue),
    )
    t_launch = time.perf_counter()
    p.start()

    # Wait for encoding
    while p.is_alive() and shared_meta["phase"] in ("init", "encoding"):
        if (time.perf_counter() - t_launch) > encode_limit:
            p.terminate()
            p.join()
            v_est, c_est = (
                theoretical_binomial_counts(n, use_symmetry)
                if clean_method == "binomial"
                else (int(shared_meta["vars"]), int(shared_meta["clauses"]))
            )
            return {
                "n": n,
                "method": clean_method,
                "solver": solver_name,
                "symmetry": use_symmetry,
                "vars": v_est,
                "clauses": c_est,
                "encode_time": float(encode_limit),
                "load_time": 0.0,
                "pure_solve_time": 0.0,
                "solve_time": 0.0,
                "total_time": float(encode_limit),
                "peak_memory_mb": float(shared_meta["peak_memory_mb"]),
                "status": "ENCODE_TIMEOUT",
                "checker_passed": False,
                "cnf_builds": 1,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "queens": [],
                "stats": {},
            }
        time.sleep(0.01)

    # Wait for solver
    if p.is_alive():
        solve_start = float(shared_meta["solve_started_at"]) or time.perf_counter()
        elapsed_solve = max(0.0, time.perf_counter() - solve_start)
        remaining = max(0.01, solve_limit - elapsed_solve)
        p.join(timeout=remaining)

    if p.is_alive():
        p.terminate()
        p.join()
        encode_time = float(shared_meta["encode_time"])
        load_time = float(shared_meta["load_time"])
        solve_time = float(solve_limit)
        pure_solve_time = max(0.0, solve_time - load_time)
        return {
            "n": n,
            "method": clean_method,
            "solver": solver_name,
            "symmetry": use_symmetry,
            "vars": int(shared_meta["vars"]),
            "clauses": int(shared_meta["clauses"]),
            "encode_time": encode_time,
            "load_time": load_time,
            "pure_solve_time": pure_solve_time,
            "solve_time": solve_time,
            "total_time": encode_time + solve_time,
            "peak_memory_mb": float(shared_meta["peak_memory_mb"]),
            "status": "SOLVE_TIMEOUT",
            "checker_passed": False,
            "cnf_builds": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "queens": [],
            "stats": {},
        }

    p.join()
    encode_time = float(shared_meta["encode_time"])
    total_vars = int(shared_meta["vars"])
    num_clauses = int(shared_meta["clauses"])

    payload = None
    try:
        payload = result_queue.get(timeout=2.0)
    except Exception:
        payload = None

    if p.exitcode != 0 or payload is None or not payload.get("ok", False):
        return {
            "n": n,
            "method": clean_method,
            "solver": solver_name,
            "symmetry": use_symmetry,
            "vars": total_vars,
            "clauses": num_clauses,
            "encode_time": encode_time,
            "load_time": float(shared_meta["load_time"]),
            "pure_solve_time": 0.0,
            "solve_time": float(shared_meta["load_time"]),
            "total_time": encode_time + float(shared_meta["load_time"]),
            "peak_memory_mb": float(shared_meta["peak_memory_mb"]),
            "status": "EXEC_ERROR",
            "checker_passed": False,
            "cnf_builds": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "queens": [],
            "stats": {},
        }

    load_time = float(payload["load_time"])
    pure_solve_time = float(payload["pure_solve_time"])
    solve_time = load_time + pure_solve_time
    checker_passed = bool(payload["checker_passed"])

    if solve_time > solve_limit:
        status_str = "SOLVE_TIMEOUT"
        solve_time = float(solve_limit)
        pure_solve_time = max(0.0, solve_time - load_time)
        checker_passed = False
    elif payload["is_sat"]:
        status_str = "SAT" if checker_passed else "EXEC_ERROR"
    else:
        status_str = "UNSAT"

    return {
        "n": n,
        "method": clean_method,
        "solver": solver_name,
        "symmetry": use_symmetry,
        "vars": total_vars,
        "clauses": num_clauses,
        "encode_time": encode_time,
        "load_time": load_time,
        "pure_solve_time": pure_solve_time,
        "solve_time": solve_time,
        "total_time": encode_time + solve_time,
        "peak_memory_mb": float(payload["peak_memory_mb"]),
        "status": status_str,
        "checker_passed": checker_passed,
        "cnf_builds": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "queens": payload["queens"] if checker_passed else [],
        "stats": payload["stats"],
    }


def solve_nqueens_cpsat(n: int, time_limit: float = 15.0, use_symmetry: bool = True):
    from ortools.sat.python import cp_model

    tracemalloc.start()
    t0 = time.perf_counter()
    model = cp_model.CpModel()
    q = [model.NewIntVar(0, n - 1, f"q_{r}") for r in range(n)]

    model.AddAllDifferent(q)
    model.AddAllDifferent([q[r] + r for r in range(n)])
    model.AddAllDifferent([q[r] - r for r in range(n)])

    if use_symmetry and n >= 4:
        half = math.ceil(n / 2)
        model.Add(q[0] < half)

    encode_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    load_time = time.perf_counter() - t1

    t2 = time.perf_counter()
    status = solver.Solve(model)
    pure_solve_time = time.perf_counter() - t2
    peak_mem = get_peak_memory_mb()
    tracemalloc.stop()

    queens = []
    checker_passed = False
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) and (load_time + pure_solve_time) <= time_limit:
        queens = [(r, solver.Value(q[r])) for r in range(n)]
        checker_passed, _ = verify_nqueens_solution(n, queens)
        status_str = "SAT" if checker_passed else "EXEC_ERROR"
        solve_time = load_time + pure_solve_time
    elif status == cp_model.INFEASIBLE:
        status_str = "UNSAT"
        solve_time = load_time + pure_solve_time
    else:
        status_str = "SOLVE_TIMEOUT"
        solve_time = float(time_limit)
        pure_solve_time = max(0.0, solve_time - load_time)

    return {
        "n": n,
        "paradigm": "CP-SAT (OR-Tools)",
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
        "queens": queens,
    }


def run_unit_tests():
    amo_methods = ["binomial", "binary", "seqcounter", "commander", "product"]
    for m in amo_methods:
        for n_unsat in [2, 3]:
            res = solve_nqueens_pysat(n_unsat, amo_method=m, use_symmetry=False, encode_limit=5.0, solve_limit=5.0)
            assert res["status"] == "UNSAT" and not res["checker_passed"], f"Failed UNSAT test: {m}, N={n_unsat}"
        for n_sat in [4, 5, 6, 7, 8]:
            for sym in [False, True]:
                res = solve_nqueens_pysat(n_sat, amo_method=m, use_symmetry=sym, encode_limit=5.0, solve_limit=5.0)
                assert res["status"] == "SAT" and res["checker_passed"], f"Failed SAT test: {m}, N={n_sat}, sym={sym}"
    print("All unit tests passed.")


if __name__ == "__main__":
    mp.freeze_support()
    run_unit_tests()