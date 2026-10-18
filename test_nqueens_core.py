import sys
import traceback

from nqueens_core import (
    solve_nqueens_pysat,
    solve_nqueens_cpsat,
    verify_nqueens_solution,
)

AMO_METHODS = [
    "binomial",
    "binary",
    "seqcounter",
    "commander",
    "product",
]

SOLVER_NAME = "cadical195"

TEST_CASES = [
    (1, "SAT"),
    (2, "UNSAT"),
    (3, "UNSAT"),
    (4, "SAT"),
    (5, "SAT"),
    (8, "SAT"),
]

ENCODE_LIMIT_SEC = 5.0
SOLVE_LIMIT_SEC = 5.0


def check_result(n, result, expected_status):
    if not isinstance(result, dict):
        return False, "Result is not a dictionary"

    required_keys = {"status", "queens", "checker_passed"}
    missing = required_keys - result.keys()

    if missing:
        return False, f"Missing fields: {sorted(missing)}"

    status = str(result["status"])

    if status != expected_status:
        return False, (
            f"Expected {expected_status}, got {status}"
        )

    if expected_status == "UNSAT":
        return True, "Correctly returned UNSAT"

    queens = result["queens"]

    if queens is None:
        return False, "SAT result contains no solution"

    valid, message = verify_nqueens_solution(n, queens)

    if not valid:
        return False, f"Invalid solution: {message}"

    if len(queens) != n:
        return False, (
            f"Expected {n} queens, got {len(queens)}"
        )

    if result["checker_passed"] is False:
        return False, "Core checker rejected the solution"

    return True, "SAT solution independently verified"


def run_test(name, test_function):
    try:
        passed, message = test_function()
        tag = "PASS" if passed else "FAIL"
        print(f"[{tag}] {name}: {message}")
        return passed

    except Exception as exc:
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False


def test_all_amo_encodings():
    print("\n=== TEST 1: AMO ENCODINGS ===")

    passed = 0
    total = 0

    for n, expected in TEST_CASES:
        for method in AMO_METHODS:
            total += 1

            def check(n=n, expected=expected, method=method):
                result = solve_nqueens_pysat(
                    n=n,
                    amo_method=method,
                    solver_name=SOLVER_NAME,
                    use_symmetry=True,
                    encode_limit=ENCODE_LIMIT_SEC,
                    solve_limit=SOLVE_LIMIT_SEC,
                )

                ok, message = check_result(
                    n, result, expected
                )
                return ok, f"AMO={method}; {message}"

            if run_test(f"N={n}, AMO={method}", check):
                passed += 1

    print(f"AMO results: {passed}/{total} passed")
    return passed, total


def test_symmetry_on_off():
    print("\n=== TEST 2: SYMMETRY ON/OFF ===")

    passed = 0
    total = 0

    for n in [4, 8]:
        for symmetry in [True, False]:
            total += 1
            tag = "ON" if symmetry else "OFF"

            def check(n=n, symmetry=symmetry):
                result = solve_nqueens_pysat(
                    n=n,
                    amo_method="product",
                    solver_name=SOLVER_NAME,
                    use_symmetry=symmetry,
                    encode_limit=ENCODE_LIMIT_SEC,
                    solve_limit=SOLVE_LIMIT_SEC,
                )

                return check_result(n, result, "SAT")

            if run_test(f"N={n}, Symmetry={tag}", check):
                passed += 1

    print(f"Symmetry results: {passed}/{total} passed")
    return passed, total


def test_cpsat():
    print("\n=== TEST 3: CP-SAT ===")

    passed = 0
    total = 0

    for n, expected in TEST_CASES:
        total += 1

        def check(n=n, expected=expected):
            result = solve_nqueens_cpsat(
                n=n,
                time_limit=SOLVE_LIMIT_SEC,
                use_symmetry=True,
            )

            return check_result(n, result, expected)

        if run_test(f"CP-SAT, N={n}", check):
            passed += 1

    print(f"CP-SAT results: {passed}/{total} passed")
    return passed, total


def main():
    print("N-Queens Core Integration Tests")
    print(f"Python: {sys.version.split()[0]}")
    print(f"SAT solver: {SOLVER_NAME}")
    print(f"Encode timeout: {ENCODE_LIMIT_SEC}s")
    print(f"Solve timeout: {SOLVE_LIMIT_SEC}s")

    summaries = [
        test_all_amo_encodings(),
        test_symmetry_on_off(),
        test_cpsat(),
    ]

    total_passed = sum(item[0] for item in summaries)
    total_tests = sum(item[1] for item in summaries)
    failed = total_tests - total_passed

    print("\n=== FINAL SUMMARY ===")
    print(f"Passed: {total_passed}/{total_tests}")
    print(f"Failed: {failed}/{total_tests}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
