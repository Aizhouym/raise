"""
Solver Wrapper for the Cedar Synthesis Engine.

Uses Cedar CLI v4.10+ with `symcc` subcommand for formal verification.
The solver uses CVC5 under the hood — no @domain annotations needed.
"""
import json
import os
import shutil
import signal
import subprocess
from dataclasses import dataclass


def _resolve_binary(env_var: str, fallback_path: str, command_name: str) -> str:
    """Resolve a tool path from env var, then PATH, then a legacy fallback."""
    return (
        os.environ.get(env_var)
        or shutil.which(command_name)
        or os.path.expanduser(fallback_path)
    )


CVC5_PATH = _resolve_binary("CVC5", "~/.local/bin/cvc5", "cvc5")
CEDAR_PATH = _resolve_binary("CEDAR", "~/.cargo/bin/cedar", "cedar")


def _run_capture(cmd, timeout):
    """Like subprocess.run(capture_output=True, text=True, timeout=...) but kills
    the WHOLE process group on timeout. `cedar symcc` spawns cvc5 as a grandchild
    that inherits the stdout/stderr pipes; the stdlib timeout only SIGKILLs the
    direct child (cedar), then blocks in communicate() waiting for cvc5 to close
    the pipe — so a hard cvc5 query hangs the reward call FOREVER (deadlocked a
    5h GRPO run at step 17). start_new_session=True puts cedar+cvc5 in a new
    process group; on timeout we os.killpg the whole group so the pipes close.
    Returns a CompletedProcess; re-raises TimeoutExpired for existing handlers."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Result of a single verification check."""
    check_name: str
    check_type: str          # "implies", "always-denies", "never-errors"
    description: str
    passed: bool
    counterexample: str      # Empty if passed


@dataclass
class VerificationResult:
    """Aggregated result of all checks."""
    loss: int                        # Number of failed checks
    results: list[CheckResult]       # Individual check results
    solver_time_s: float = 0.0       # Wall-clock time for all solver calls

    @property
    def passed(self) -> bool:
        return self.loss == 0


# ---------------------------------------------------------------------------
# Gate 1: Syntax check (unchanged)
# ---------------------------------------------------------------------------

def run_syntax_check(schema_path: str, policy_path: str) -> tuple[bool, str, str]:
    """Run `cedar validate` on the given policy against the schema.

    Returns (is_valid, error_msg, error_kind) where error_kind is one of:
      - ""           : passed (is_valid=True)
      - "parse"      : returncode 1 — Cedar grammar / parse error
      - "validation" : returncode 3 — Cedar type-checker / validator rejected
                       (e.g. unguarded optional attribute, type mismatch)
      - "other"      : any other non-zero returncode or runtime failure
    """
    try:
        result = _run_capture(
            [CEDAR_PATH, "validate", "--schema", schema_path, "--policies", policy_path],
            timeout=10,
        )
        is_valid = result.returncode == 0
        error_msg = ""
        error_kind = ""
        if not is_valid:
            error_msg = (result.stderr.strip() or result.stdout.strip())
            if result.returncode == 1:
                error_kind = "parse"
            elif result.returncode == 3:
                error_kind = "validation"
            else:
                error_kind = "other"

        if os.environ.get("CEDAR_DEBUG"):
            import hashlib

            try:
                with open(policy_path, "rb") as policy_file:
                    content = policy_file.read()
                sha = hashlib.sha256(content).hexdigest()[:12]
                size = len(content)
            except Exception as exc:
                sha = f"read-err:{exc}"
                size = -1

            dbg_path = os.path.join(os.path.dirname(policy_path), "_syntax_debug.log")
            with open(dbg_path, "a") as dbg:
                dbg.write(
                    f"--- run_syntax_check ---\n"
                    f"CEDAR_PATH={CEDAR_PATH}\n"
                    f"policy_path={policy_path}\n"
                    f"policy_size={size} sha256={sha}\n"
                    f"returncode={result.returncode}\n"
                    f"error_kind={error_kind!r}\n"
                    f"stdout={result.stdout[:400]!r}\n"
                    f"stderr={result.stderr[:400]!r}\n"
                    f"is_valid={is_valid}\n\n"
                )

        return is_valid, error_msg, error_kind
    except subprocess.TimeoutExpired:
        return False, "Cedar validate timed out.", "other"
    except FileNotFoundError:
        return (
            False,
            "Cedar CLI not found. Install with: cargo install cedar-policy-cli",
            "other",
        )


# ---------------------------------------------------------------------------
# Gate 2: Symcc verification
# ---------------------------------------------------------------------------

def _run_symcc(
    schema_path: str,
    principal_type: str,
    action: str,
    resource_type: str,
    subcommand: str,
    extra_args: list[str],
) -> tuple[bool, str]:
    """
    Run a single `cedar symcc` check.
    Returns (passed, output_text).
    """
    cmd = [
        CEDAR_PATH, "symcc",
        "--cvc5-path", CVC5_PATH,
        "--principal-type", principal_type,
        "--action", action,
        "--resource-type", resource_type,
        "--schema", schema_path,
        "--counterexample",
        subcommand,
    ] + extra_args

    try:
        result = _run_capture(cmd, timeout=30)
        output = result.stdout.strip() or result.stderr.strip()
        passed = "VERIFIED" in output
        return passed, output
    except subprocess.TimeoutExpired:
        return False, "CVC5 solver timed out (30s limit)."
    except FileNotFoundError:
        return False, f"CVC5 not found at {CVC5_PATH}. Set CVC5 env var."


def run_implies_check(
    schema_path: str,
    candidate_path: str,
    reference_path: str,
    principal_type: str,
    action: str,
    resource_type: str,
    check_name: str,
    description: str,
) -> CheckResult:
    """
    Check: candidate ≤ reference (candidate is no more permissive).
    Uses `cedar symcc implies --policies1 <candidate> --policies2 <reference>`.
    """
    passed, output = _run_symcc(
        schema_path, principal_type, action, resource_type,
        "implies",
        ["--policies1", candidate_path, "--policies2", reference_path],
    )
    return CheckResult(
        check_name=check_name,
        check_type="implies",
        description=description,
        passed=passed,
        counterexample="" if passed else output,
    )


def run_always_denies_check(
    schema_path: str,
    candidate_path: str,
    principal_type: str,
    action: str,
    resource_type: str,
    check_name: str,
    description: str,
    expect_denies: bool = True,
) -> CheckResult:
    """
    Check: does the policy always deny for this request environment?
    For liveness: we want always-denies to be FALSE (expect_denies=False).
    """
    passed_raw, output = _run_symcc(
        schema_path, principal_type, action, resource_type,
        "always-denies",
        ["--policies", candidate_path],
    )
    # If expect_denies=False (liveness), we want the check to FAIL (not always deny)
    if expect_denies:
        passed = passed_raw
    else:
        passed = not passed_raw  # We WANT it to not always deny

    return CheckResult(
        check_name=check_name,
        check_type="liveness" if not expect_denies else "always-denies",
        description=description,
        passed=passed,
        counterexample="" if passed else (
            "LIVENESS VIOLATION: Policy always denies this action. "
            "It must allow at least one scenario."
            if not expect_denies else output
        ),
    )


def run_liveness_overlap_check(
    schema_path: str,
    candidate_path: str,
    probe_path: str,
    principal_type: str,
    action: str,
    resource_type: str,
    check_name: str,
    description: str,
) -> CheckResult:
    """Check candidate n liveness_probe is non-empty.

    Ported from autocedar/harness/solver_wrapper.py as of commit e739358
    ("Enrich SymCC signal layer"), which is the version that generated the
    dataset's verification plans. This vendored copy predates that commit, so
    every "liveness-overlap" entry fell through orchestrator's else-branch and
    was silently skipped -- 3,131 of the corpus's 44,388 checks (7.1%).

    `cedar symcc disjoint` verifies there is NO overlap. For liveness that
    VERIFIED result is a failure: a formal counterexample means at least one
    request is allowed by both candidate and probe, which is the witness we want.
    """
    disjoint, output = _run_symcc(
        schema_path, principal_type, action, resource_type,
        "disjoint",
        ["--policies1", candidate_path, "--policies2", probe_path],
    )
    if disjoint:
        return CheckResult(
            check_name=check_name,
            check_type="liveness",
            description=description,
            passed=False,
            counterexample=(
                "LIVENESS VIOLATION: candidate policy is disjoint from the required "
                "liveness probe, so no approved request slice remains possible."
            ),
        )
    return CheckResult(
        check_name=check_name,
        check_type="liveness",
        description=description,
        passed=True,
        counterexample="",
    )


def run_never_errors_check(
    schema_path: str,
    candidate_path: str,
    principal_type: str,
    action: str,
    resource_type: str,
) -> CheckResult:
    """Check: policy never produces runtime errors."""
    passed, output = _run_symcc(
        schema_path, principal_type, action, resource_type,
        "never-errors",
        ["--policies", candidate_path],
    )
    return CheckResult(
        check_name="no_runtime_errors",
        check_type="never-errors",
        description="Policy must not produce runtime errors for any input",
        passed=passed,
        counterexample="" if passed else output,
    )
