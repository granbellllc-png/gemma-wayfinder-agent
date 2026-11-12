"""
5-Gate Forensic Pre-Submission Audit (`verify_submission_integrity.py`)
=======================================================================
Enforces strict zero-defect parity across code, formal proofs, live Gemma 4 GPU telemetry,
bibliography, LaTeX/PDF/Markdown artifacts, and public Git repository hygiene:

  Gate 1: Cryptographic SHA-256 & Live Gemma 4 GPU Telemetry Grounding
  Gate 2: Bidirectional Citation Completeness & Primary-Source Bibliographic Check
  Gate 3: AST Anti-Tautology Lint & 4-Inequality QF_RDL Formula-to-Code Parity
  Gate 4: Competition Word-Limit (<3,000 words) & Cross-Artifact SHA-256 Synchronization
  Gate 5: Public Repository Hygiene (no local paths, no internal guides, no `# BUG:` hints)
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


def _canonical_sha256(obj: dict, exclude_keys: tuple[str, ...]) -> str:
    raw = json.dumps({k: v for k, v in obj.items() if k not in exclude_keys}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def check_ast_no_self_division(py_path: Path) -> None:
    tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            # Detect `x / x` or `c * x / max(1, x)`
            left_names = {n.id for n in ast.walk(node.left) if isinstance(n, ast.Name)}
            if isinstance(node.right, ast.Name) and node.right.id in left_names:
                raise AssertionError(f"Potential self-division tautology in {py_path.name}:{node.lineno}")
            if (
                isinstance(node.right, ast.Call)
                and isinstance(node.right.func, ast.Name)
                and node.right.func.id == "max"
            ):
                arg_names = {n.id for a in node.right.args for n in ast.walk(a) if isinstance(n, ast.Name)}
                if left_names and arg_names and left_names == arg_names:
                    raise AssertionError(
                        f"Tautological `x / max(const, x)` division in {py_path.name}:{node.lineno}: {left_names}"
                    )


def main() -> None:
    root = Path(__file__).resolve().parent

    # -------------------------------------------------------------------------
    # GATE 1: Live GPU Telemetry & Benchmark Digest Verification
    # -------------------------------------------------------------------------
    gpu_path = root / "results" / "live_gemma4_gpu_telemetry.json"
    bench_path = root / "results" / "benchmark_telemetry.json"
    assert gpu_path.exists(), "Missing results/live_gemma4_gpu_telemetry.json"
    assert bench_path.exists(), "Missing results/benchmark_telemetry.json"

    gpu_data = json.loads(gpu_path.read_text(encoding="utf-8"))
    gpu_sha = _canonical_sha256(gpu_data, ("execution_timestamp_utc", "sha256_digest"))
    assert gpu_sha == gpu_data["sha256_digest"], f"GPU telemetry SHA mismatch: {gpu_sha} != {gpu_data['sha256_digest']}"
    assert "gemma_4_e2b_it" in gpu_data["models"] and "gemma_4_e4b_it" in gpu_data["models"]

    bench_data = json.loads(bench_path.read_text(encoding="utf-8"))
    bench_sha = _canonical_sha256(bench_data, ("elapsed_seconds", "sha256_digest"))
    assert bench_sha == bench_data["sha256_digest"], "Benchmark telemetry SHA mismatch"
    assert bench_data["experiment_b_live_gemma4_gpu_suite"]["live_gpu_telemetry_summary"]["digest_verified"] is True

    # -------------------------------------------------------------------------
    # GATE 2: Bidirectional Citations & Verified Primary References
    # -------------------------------------------------------------------------
    paper_txt = (root / "PAPER.md").read_text(encoding="utf-8")
    assert "## References" in paper_txt, "PAPER.md missing ## References section"
    body_txt, refs_txt = paper_txt.split("## References", 1)

    expected_citation_markers = [
        ("Gemma Team", "arXiv:2607.02770"),
        ("Jimenez", "SWE-bench"),
        ("Markowitz, E.", "Tree-of-Traversals"),
        ("Markowitz, E.", "KG-LLM-Bench"),
        ("Zhang, F.", "RepoCoder"),
        ("Yao", "ReAct"),
        ("Banach", "1922"),
        ("Khalil", "Nonlinear Systems"),
        ("Nieuwenhuis", "Davis"),
    ]
    for author_marker, title_marker in expected_citation_markers:
        assert author_marker in refs_txt and title_marker in refs_txt, (
            f"Missing verified reference ({author_marker}, {title_marker}) in ## References"
        )
        surname = author_marker.split(",")[0].split()[0]
        assert surname in body_txt, f"Reference `{author_marker}` listed in References but NEVER cited in PAPER.md body!"

    assert "Zhang, Y., et al. (2023)" not in paper_txt, "Wrong first author for RepoCoder detected!"

    # Check every live GPU resolve rate appears verbatim in PAPER.md
    for m_key in ("gemma_4_e2b_it", "gemma_4_e4b_it"):
        for cfg in gpu_data["models"][m_key]["configurations"]:
            rate_str = f"{cfg['resolve_rate_pct']}%"
            assert rate_str in body_txt, f"Live GPU resolve rate {rate_str} ({cfg['configuration']}) missing from PAPER.md"

    # -------------------------------------------------------------------------
    # GATE 3: AST Anti-Tautology & 4-Inequality QF_RDL Parity
    # -------------------------------------------------------------------------
    for fname in ("harness.py", "benchmark_runner.py", "live_gemma4_gpu_benchmark.py"):
        check_ast_no_self_division(root / fname)

    harness_txt = (root / "harness.py").read_text(encoding="utf-8")
    for req_label in (
        "lyapunov_contraction_envelope",
        "non_negative_potential",
        "max_blast_radius_bound",
        "max_prompt_tokens_bound",
    ):
        assert req_label in harness_txt, f"Missing QF_RDL constraint `{req_label}` in harness.py"
    assert "TRIZContradictionMatrixRouter" not in harness_txt, "Stale TRIZ boilerplate still in harness.py"

    # -------------------------------------------------------------------------
    # GATE 4: Word Budget (<3,000 Words), Academic Register & SHA-256 Sync
    # -------------------------------------------------------------------------
    words = re.findall(r"\S+", paper_txt)
    assert len(words) < 3000, f"PAPER.md exceeds 3,000-word limit: {len(words)} words"

    title_line = paper_txt.splitlines()[0].lstrip("# ").strip()
    assert len(title_line.split()) <= 16, f"Title too verbose ({len(title_line.split())} words): {title_line}"
    prose_no_code = re.sub(r"```.*?```", "", body_txt, flags=re.DOTALL)
    prose_no_math = re.sub(r"\$\$.*?\$\$|\$[^$]+\$", "", prose_no_code, flags=re.DOTALL)
    assert "!" not in prose_no_math, "Exclamation mark found in PAPER.md academic prose"
    assert re.search(r"`\d[\d,.]*%?`", prose_no_math) is None, "Backticked plain number found in PAPER.md prose"

    tex_txt = (root / "paper.tex").read_text(encoding="utf-8")
    readme_txt = (root / "README.md").read_text(encoding="utf-8")
    nb_txt = (root / "kaggle_notebook" / "gemma-4-tcgs-lyapunov-swe-agent.ipynb").read_text(encoding="utf-8")

    short_gpu_sha = gpu_sha[:16]
    for doc_name, content in (("PAPER.md", paper_txt), ("paper.tex", tex_txt), ("README.md", readme_txt), ("Notebook", nb_txt)):
        assert short_gpu_sha in content, f"{doc_name} out of sync with live GPU telemetry SHA {short_gpu_sha}"
        for stale_token in ("Wayfinder v3.0", "56.8%", "68.4%", "99c1504723884779", "TRIZContradictionMatrixRouter"):
            assert stale_token not in content, f"Stale v3.0 token `{stale_token}` leaked into {doc_name}"

    # Verify every code cell in the Kaggle verification notebook parses cleanly via Python AST
    nb_obj = json.loads(nb_txt)
    code_cells = [c for c in nb_obj.get("cells", []) if c.get("cell_type") == "code"]
    assert len(code_cells) >= 4, f"Expected >=4 code cells in Kaggle notebook, found {len(code_cells)}"
    for idx, cell in enumerate(code_cells):
        cell_src = "".join(cell.get("source", []))
        ast.parse(cell_src, filename=f"notebook_code_cell_{idx}.py")

    # Verify directional claims do not assert Wayfinder raises resolution if resolved_count <= baseline
    e2b_cfgs = gpu_data["models"]["gemma_4_e2b_it"]["configurations"]
    if e2b_cfgs[3]["resolved_count"] <= e2b_cfgs[0]["resolved_count"]:
        assert "raises resolution from" not in body_txt.split("on E2B-it")[0][-80:], (
            "Directional mismatch: PAPER.md claims E2B-it resolution rise when Wayfinder <= BM25!"
        )

    # -------------------------------------------------------------------------
    # GATE 5: Public Repository Hygiene
    # -------------------------------------------------------------------------
    assert not (root / "KAGGLE_RESUBMIT_GUIDE.md").exists(), "Internal KAGGLE_RESUBMIT_GUIDE.md must not exist in repo"
    tracked = subprocess.check_output(["git", "ls-files"], cwd=str(root), text=True).splitlines()
    forbidden_local_prefix = "/" + "Users" + "/" + "amg"
    for rel in tracked:
        p = root / rel
        if not p.exists() or p.suffix in (".png", ".pdf"):
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        assert forbidden_local_prefix not in txt, f"Leaked local path in {rel}"
        if rel == "live_gemma4_gpu_benchmark.py":
            assert "# BUG:" not in txt, "Leaked `# BUG:` hint comment in live_gemma4_gpu_benchmark.py"

    print(
        f"ALL 5 INTEGRITY GATES PASSED: words={len(words)}/3000 | "
        f"gpu_sha={gpu_sha[:16]} | bench_sha={bench_sha[:16]} | tracked_files={len(tracked)}"
    )


if __name__ == "__main__":
    main()
