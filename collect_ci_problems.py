#!/usr/bin/env python3
"""Script to collect CI problems and format them for Claude 3.7.
This runs the same checks as your GitHub CI workflow and aggregates the results.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import toml


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Collect CI problems for fixing.")
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running pytest (useful for quick syntax/format checks)",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="cohortbalancer3",
        help="Source directory to check (default: cohortbalancer3)",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="tests",
        help="Test directory to check (default: tests)",
    )
    parser.add_argument(
        "--examples-dir",
        type=str,
        default="examples",
        help="Examples directory to check (default: examples)",
    )
    parser.add_argument(
        "--style-output-file",
        type=str,
        default="ci_style_issues.md",
        help="Output file for style/doc/annotation issues",
    )
    parser.add_argument(
        "--logic-output-file",
        type=str,
        default="ci_logic_issues.md",
        help="Output file for potential logic/code change issues",
    )
    return parser.parse_args()


def get_project_root() -> Path:
    """Get the project root directory."""
    # Start from the current working directory and walk up until
    # we find a directory that looks like a project root
    cwd = Path.cwd()

    # Look for common project markers
    for parent in [cwd] + list(cwd.parents):
        # Check for common project root indicators
        if any(
            (parent / marker).exists()
            for marker in [
                "pyproject.toml",
                "setup.py",
                "setup.cfg",
                ".git",
                ".github",
                "requirements.txt",
            ]
        ):
            return parent

    # If we can't find a good marker, just use the current directory
    return cwd


def load_pyproject_config() -> dict[str, Any]:
    """Load configuration from pyproject.toml."""
    project_root = get_project_root()
    pyproject_path = project_root / "pyproject.toml"

    if not pyproject_path.exists():
        raise FileNotFoundError(f"pyproject.toml not found in project root: {project_root}")

    try:
        return toml.load(pyproject_path)
    except Exception as e:
        raise OSError(f"Failed to load or parse pyproject.toml: {e}")


def make_path_relative(filepath: str) -> str:
    """Convert an absolute path to a path relative to the project root."""
    if not filepath:
        return filepath

    try:
        # Get the absolute path
        abs_path = Path(filepath).resolve()

        # Get the project root
        root = get_project_root()

        # Make the path relative to the root
        rel_path = abs_path.relative_to(root)

        return str(rel_path)
    except (ValueError, TypeError):
        # If we can't make it relative, return the original
        return filepath


def run_command(cmd: list[str], capture_output: bool = True) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, and stderr."""
    try:
        result = subprocess.run(
            cmd, capture_output=capture_output, text=True, check=False
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", f"Error executing command: {e}"


def get_ruff_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract ruff configuration from pyproject.toml."""
    if "tool" not in config or "ruff" not in config["tool"]:
        raise ValueError("Ruff configuration ([tool.ruff]) not found in pyproject.toml")

    ruff_config = config["tool"]["ruff"]

    # Ensure essential keys are present if needed, or rely on Ruff's defaults
    # For example, you might want to ensure 'lint' exists if you parse sub-keys later
    if "lint" not in ruff_config:
         print("Warning: [tool.ruff.lint] section not found in pyproject.toml. Ruff defaults will be used.")
         # Or raise ValueError("Ruff lint configuration ([tool.ruff.lint]) not found")

    return ruff_config


def run_ruff_format(directories: list[str]) -> dict[str, Any]:
    """Run ruff formatter to fix formatting."""
    print("Running ruff formatter...")

    # Format files with ruff
    format_cmd = ["ruff", "format"] + directories
    format_exit_code, format_stdout, format_stderr = run_command(format_cmd)

    return {
        "success": format_exit_code == 0,
        "errors": [{"message": format_stderr}] if format_exit_code != 0 else [],
    }


def run_ruff_check(directories: list[str], ruff_config: dict[str, Any]) -> dict[str, Any]:
    """Run ruff linter, apply fixes, and collect remaining issues using pyproject.toml config."""
    print("Running ruff linter and applying fixes...")

    # Get config path
    project_root = get_project_root()
    pyproject_path = project_root / "pyproject.toml"
    config_arg = ["--config", str(pyproject_path)]

    # --- Step 1: Apply fixes ---
    fix_cmd = ["ruff", "check", "--fix", "--exit-zero"] + config_arg + directories
    # Use --exit-zero so the script doesn't stop even if fixes were applied or errors remain
    fix_exit_code, fix_stdout, fix_stderr = run_command(fix_cmd, capture_output=True)

    if fix_exit_code != 0:
        print(f"Warning: Ruff fix command encountered issues.", file=sys.stderr)
        if fix_stderr:
            print(f"Ruff fix stderr:\n{fix_stderr}", file=sys.stderr)
    elif "fixed" in fix_stdout.lower() or "fixed" in fix_stderr.lower():
         print("Ruff applied automatic fixes.") # Inform user fixes were made

    # --- Step 2: Check for remaining issues and report ---
    print("Checking for remaining linting issues...")
    check_cmd = ["ruff", "check", "--output-format", "json"] + config_arg + directories

    exit_code, stdout, stderr = run_command(check_cmd)

    result = {"success": exit_code == 0, "errors": []}

    # Log stderr even on success, as it might contain warnings
    if stderr.strip():
         # Avoid printing the JSON parsing error message from stderr if stdout is valid json
         if not (exit_code != 0 and stdout.strip() and "invalid json" in stderr.lower()):
              print(f"Ruff check stderr:\n{stderr}", file=sys.stderr)

    if exit_code != 0:
        try:
            if stdout.strip():
                issues = json.loads(stdout)
                for issue in issues:
                    file_path = issue.get("filename", "")
                    fix_info = issue.get("fix")
                    # Report only non-fixable issues or issues where fix is 'never applicable'
                    if not fix_info or fix_info.get("applicability") == "never":
                        fixable_status = "" # No longer relevant as we only report unfixable
                    # Sometimes Ruff reports fixable even after --fix, mark them
                    elif fix_info and fix_info.get("applicability") != "never":
                         fixable_status = " (fix reported but not applied)"
                    else:
                         fixable_status = ""

                    result["errors"].append(
                        {
                            "file": make_path_relative(file_path),
                            "line": issue.get("location", {}).get("row"),
                            "code": issue.get("code"),
                            # Report message without fixable status unless fix failed
                            "message": f"{issue.get('message')}{fixable_status}",
                        }
                    )
            elif stderr.strip():
                 result["errors"].append({"message": f"Ruff check execution failed:\n{stderr}"})
            else:
                 result["errors"].append({"message": "Ruff check execution failed with no specific error output."})

        except json.JSONDecodeError:
            error_message = f"Failed to parse Ruff JSON output.\nStdout:\n{stdout}\nStderr:\n{stderr}"
            result["errors"].append({"message": error_message})
        except Exception as e:
            error_message = f"Error processing Ruff output: {e}\nStdout:\n{stdout}\nStderr:\n{stderr}"
            result["errors"].append({"message": error_message})

    # If after fixing, the exit code is 0, override success to True
    if not result["errors"] and exit_code == 0:
        result["success"] = True
    # If errors remain, ensure success is False
    elif result["errors"]:
         result["success"] = False


    return result


def run_mypy(source_dir: str, config: dict[str, Any]) -> dict[str, Any]:
    """Run mypy type checking and collect results."""
    print("Running mypy type checking...")

    # Use any mypy config options from pyproject.toml
    mypy_args = []
    if "tool" in config and "mypy" in config["tool"]:
        mypy_config = config["tool"]["mypy"]
        # You could extract specific options here if needed

    cmd = ["mypy", source_dir] + mypy_args
    exit_code, stdout, stderr = run_command(cmd)

    result = {"success": exit_code == 0, "errors": []}

    if exit_code != 0:
        # Parse mypy output which shows file:line: error format
        type_errors = []
        for line in stdout.splitlines():
            if ": error:" in line:
                parts = line.split(": error:", 1)
                file_line = parts[0]
                message = parts[1].strip()

                file_parts = file_line.split(":", 1)
                file_path = file_parts[0]
                line_num = file_parts[1] if len(file_parts) > 1 else "unknown"

                type_errors.append(
                    {
                        "file": make_path_relative(file_path),
                        "line": line_num,
                        "message": message,
                    }
                )

        result["errors"] = type_errors

    return result


def run_pytest(config: dict[str, Any]) -> dict[str, Any]:
    """Run pytest and collect results."""
    print("Running pytest...")

    # Extract pytest args from pyproject.toml if available
    pytest_args = []
    if "tool" in config and "pytest" in config["tool"]:
        pytest_config = config["tool"]["pytest"]
        # You could extract specific options here if needed

    cmd = ["pytest", "-v"] + pytest_args
    exit_code, stdout, stderr = run_command(cmd)

    result = {"success": exit_code == 0, "errors": []}

    if exit_code != 0:
        # Extract failed tests
        failed_tests = []

        # Look for the FAILURES section in pytest output
        failures_section = False
        current_test = None
        error_message = []

        for line in stdout.splitlines():
            if "FAILURES" in line and "=" in line:
                failures_section = True
                continue

            if failures_section:
                if line.startswith("_"):
                    # New test failure
                    if current_test:
                        failed_tests.append(
                            {"test": current_test, "message": "\n".join(error_message)}
                        )
                        error_message = []

                    current_test = line.strip("_")
                    # Make test path relative if possible
                    if " " in current_test:
                        parts = current_test.split(" ", 1)
                        relative_path = make_path_relative(parts[0])
                        current_test = f"{relative_path} {parts[1]}"
                elif current_test:
                    # Check for file paths in the error message and make them relative
                    if line.strip().startswith(("E   File ", "E    File ")):
                        parts = line.split('"', 2)
                        if len(parts) >= 3:
                            file_path = parts[1]
                            relative_path = make_path_relative(file_path)
                            line = parts[0] + '"' + relative_path + '"' + parts[2]
                    error_message.append(line)

        # Add the last test if there is one
        if current_test:
            failed_tests.append(
                {"test": current_test, "message": "\n".join(error_message)}
            )

        result["errors"] = failed_tests

    return result


def format_ai_guidelines(config: dict[str, Any]) -> str:
    """Format concise guidelines based on project configuration."""
    guidelines = []

    # Extract relevant configuration values for Ruff
    ruff_config = get_ruff_config(config)
    line_length = ruff_config.get("line-length", 88)

    # Create simple, direct guidelines
    guidelines.append("# Guidelines for Code Corrections")
    guidelines.append("\nFollow the project's Ruff configuration and style settings, with these important notes:")

    # Key settings
    guidelines.append("\n## Key Settings")
    guidelines.append(f"- Line length: {line_length} characters")
    guidelines.append("- Style: Follow Python PEP 8 conventions with Ruff enforcement")

    # Mathematical notation exception
    guidelines.append("\n## Important Exceptions")
    guidelines.append("- Mathematical notation: Use standard scientific notation (X, Y, Σ, etc.)")
    guidelines.append("  * Capital variable names for matrices and standard mathematical symbols are allowed")
    guidelines.append("  * This exception to Python naming conventions is intentional for scientific code")

    # Type annotations are critical
    guidelines.append("\n## Type Annotations")
    guidelines.append("- All functions must have complete type annotations")
    guidelines.append("- Use appropriate container types from typing module")

    # Brief example
    guidelines.append("\n## Example")
    guidelines.append("```python")
    guidelines.append("from typing import List")
    guidelines.append("import numpy as np")
    guidelines.append("from numpy.typing import NDArray")
    guidelines.append("")
    guidelines.append("def calculate_ols(X: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:")
    guidelines.append('    """Calculate ordinary least squares solution using matrix operations."""')
    guidelines.append("    # Mathematical notation with uppercase is permitted")
    guidelines.append("    beta = np.linalg.inv(X.T @ X) @ X.T @ y")
    guidelines.append("    return beta")
    guidelines.append("```")

    return "\n".join(guidelines)


def classify_error(error: Dict[str, Any], source: str) -> str:
    """Classify an error as 'style' (low-risk for AI) or 'logic' (high-risk for AI)."""
    
    # --- High-Risk Sources ---
    if source == "pytest":
        # Test failures always indicate potential logic issues.
        return "logic" 
        
    if source == "mypy":
        # Type *errors* (not just missing annotations) often signal logic problems
        # or misunderstandings of data types that require careful fixes.
        # Treating all Mypy errors as high-risk is safer for the AI.
        return "logic"

    # --- Ruff Classification (More Granular) ---
    if source == "ruff":
        code = error.get("code", "")
        if not code:
            return "logic" # Unknown Ruff error, assume high-risk

        # Low-Risk Categories (Style, Docs, Simple Annotations, Formatting)
        # E: PEP8 style errors (mostly whitespace, line length)
        # W: PEP8 style warnings (similar to E)
        # D: Docstring errors
        # I: Import sorting errors
        # COM: Comment issues (like missing noqa reasons)
        # ANN0xx, ANN1xx, ANN2xx: Basic missing annotations (argument, *, return) - Relatively low risk to add.
        # N: Naming conventions (excluding those ignored in pyproject.toml)
        # RUF100: Unused noqa directives (safe removal)
        # PTH: Pathlib suggestions (often stylistic improvements)
        # EM: Exception message format
        # F401: Unused import (safe removal)
        low_risk_prefixes = {"E", "W", "D", "I", "COM", "N", "RUF1", "PTH", "EM"}
        low_risk_exact = {"F401"}
        
        # ANN codes need careful consideration:
        # ANN001, ANN002, ANN003, ANN101, ANN102, ANN201, ANN202, ANN204, ANN205, ANN206 are generally safe additions.
        # ANN401 (Any type) might hide logic issues, could be borderline. Let's keep it low-risk for now to encourage annotation.
        is_simple_ann = code.startswith("ANN") # Consider most ANN low-risk for now

        if code.startswith(tuple(low_risk_prefixes)) or code in low_risk_exact or is_simple_ann:
             # Exclude specific complex/logic-related E/W codes if necessary, but generally safe.
             return "style" # Treat as low-risk

        # High-Risk Categories (Potential Logic, Bugs, Complexity, Security)
        # F: Fatal errors (except F401), e.g., F821 (undefined name), F841 (unused variable - high risk!)
        # B: Bugbear warnings (often subtle logic issues)
        # C90: Complexity warnings (fixing requires understanding logic)
        # S: Security warnings
        # PL: Pylint rules (often logic-related)
        # TRY: Try/except block issues
        # PD: Pandas-specific issues (often logic/correctness related)
        # SIM: Simplification suggestions (naive application can break logic)
        # Anything else not explicitly classified as low-risk
        
        # Special Case: Issue reported fixable but wasn't applied by --fix
        # This often indicates a more complex situation needing review.
        if "fix reported but not applied" in error.get("message", ""):
             return "logic"

        # If not classified as low-risk, assume high-risk
        return "logic"

    # Default classification for any unknown source
    return "logic" 


def format_classified_output(
    results: dict[str, dict[str, Any]],
    skip_tests: bool,
    config: dict[str, Any],
) -> tuple[str, str]:
    """Format classified issues into two separate reports."""
    style_output = []
    logic_output = []

    # Generate guidelines once
    guidelines_header = format_ai_guidelines(config)
    separator = "\n\n" + "=" * 50 + "\n"
    issues_title = "# CI Issues to Fix\n"

    style_output.append(guidelines_header)
    style_output.append(separator)
    style_output.append(issues_title)
    style_output.append("## Style, Documentation, and Annotation Issues\n")

    logic_output.append(guidelines_header)
    logic_output.append(separator)
    logic_output.append(issues_title)
    logic_output.append("## Potential Code Logic Issues and Test Failures\n")

    has_style_issues = False
    has_logic_issues = False

    # Classify and append Ruff issues
    if not results["ruff_check"]["success"] and results["ruff_check"]["errors"]:
        for error in results["ruff_check"]["errors"]:
            classification = classify_error(error, "ruff")
            formatted_error = f"- **{error.get('file', 'Unknown')}:{error.get('line', '??')}** - [Ruff] {error.get('code', '')}: {error.get('message', 'Unknown')}"
            if classification == "style":
                style_output.append(formatted_error)
                has_style_issues = True
            else:
                logic_output.append(formatted_error)
                has_logic_issues = True
        style_output.append("") # Add spacing if there were issues
        logic_output.append("")

    # Classify and append Mypy issues
    if not results["mypy"]["success"] and results["mypy"]["errors"]:
         logic_output.append("### Type Checking Issues (Mypy)\n") # Mypy always goes to logic for now
         for error in results["mypy"]["errors"]:
             # classification = classify_error(error, "mypy") # Currently always 'logic'
             formatted_error = f"- **{error.get('file', 'Unknown')}:{error.get('line', '??')}** - [Mypy] {error.get('message', 'Unknown')}"
             logic_output.append(formatted_error)
             has_logic_issues = True
         logic_output.append("")

    # Append Pytest issues (always logic)
    if not skip_tests and "pytest" in results and not results["pytest"]["success"]:
         if results["pytest"]["errors"]:
             logic_output.append("### Test Failures (Pytest)\n")
             has_logic_issues = True
             for error in results["pytest"]["errors"]:
                 test = error.get("test", "Unknown test")
                 message = error.get("message", "Unknown error")
                 logic_output.append(f"#### Test: {test}\n")
                 logic_output.append("```")
                 logic_output.append(message)
                 logic_output.append("```\n")
         logic_output.append("")


    if not has_style_issues:
        style_output.append("No style/doc/annotation issues found requiring manual review.")
    if not has_logic_issues:
        logic_output.append("No potential logic issues or test failures found.")

    return "\n".join(style_output), "\n".join(logic_output)


def main():
    """Main function to run all checks and collect results."""
    args = parse_args()
    skip_tests = args.skip_tests
    source_dir = args.source_dir
    test_dir = args.test_dir
    examples_dir = args.examples_dir
    style_output_file = args.style_output_file
    logic_output_file = args.logic_output_file

    try:
        config = load_pyproject_config()
        ruff_config = get_ruff_config(config)
    except (OSError, FileNotFoundError, ValueError) as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Collecting CI problems for {source_dir}...")

    check_dirs = [source_dir]
    if test_dir and Path(test_dir).exists():
        print(f"Including test directory: {test_dir}")
        check_dirs.append(test_dir)
    if examples_dir and Path(examples_dir).exists():
        print(f"Including examples directory: {examples_dir}")
        check_dirs.append(examples_dir)
    if skip_tests:
        print("Skipping pytest execution as requested")

    # Install dev dependencies if needed
    try:
        import mypy
        import ruff

        if not skip_tests:
            import pytest
    except ImportError:
        print("Installing development dependencies...")
        run_command(
            [sys.executable, "-m", "pip", "install", "-e", ".[dev,viz]"],
            capture_output=False,
        )

    # --- Run checks in order ---
    ruff_format_result = run_ruff_format(check_dirs)
    if not ruff_format_result["success"]:
         print("Warning: Ruff formatting failed. Proceeding with checks...", file=sys.stderr)
    ruff_check_result = run_ruff_check(check_dirs, ruff_config)
    mypy_result = run_mypy(source_dir, config)
    pytest_result = {}
    if not skip_tests:
        pytest_result = run_pytest(config)

    # --- Aggregate results ---
    results = {
        "ruff_format": ruff_format_result,
        "ruff_check": ruff_check_result,
        "mypy": mypy_result,
    }
    if not skip_tests:
        results["pytest"] = pytest_result

    # --- Format Classified Output ---
    style_report, logic_report = format_classified_output(results, skip_tests, config)

    # --- Save to separate files ---
    with open(style_output_file, "w") as f:
        f.write(style_report)
    print(f"\nStyle/Doc/Annotation issues saved to {style_output_file}")

    with open(logic_output_file, "w") as f:
        f.write(logic_report)
    print(f"Potential Logic issues and Test failures saved to {logic_output_file}")


    # --- Print Summary ---
    print("\nSummary:")
    print(f"Formatting (Ruff): {'PASS' if results['ruff_format']['success'] else 'FAIL'}")
    # Linting PASS means no *remaining* issues after fixing
    print(f"Linting (Ruff): {'PASS' if results['ruff_check']['success'] else 'FAIL'} ({len(results['ruff_check']['errors'])} remaining)")
    print(f"Type checking (Mypy): {'PASS' if results['mypy']['success'] else 'FAIL'} ({len(results['mypy']['errors'])} errors)")
    if not skip_tests:
         pytest_errors = len(results["pytest"].get("errors", []))
         print(f"Tests (Pytest): {'PASS' if results['pytest']['success'] else 'FAIL'} ({pytest_errors} failures)")
    else:
        print("Tests (Pytest): SKIPPED")

    print(f"\nReview the contents of {style_output_file} and {logic_output_file}.")


if __name__ == "__main__":
    main()
