"""
Comprehensive Unit, Property-Based, and Formal QF_RDL SMT Verification Suite (v3.1)
for Gemma 4 TCGS + DCLV + QF_RDL Harness
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from benchmark_runner import run_experiment_d_live_multifile_bugfix
from harness import (
    BanachContractionValidator,
    DifferenceLogicSMTVerifier,
    DiscreteLyapunovValidator,
    TopologicalCodeGraph,
)
from live_gemma4_gpu_benchmark import run_self_check


def test_cross_file_ast_dependency_and_inheritance_linking() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "base.py").write_text(
            "class BaseEngine:\n"
            "    def validate(self, payload):\n"
            "        return bool(payload)\n"
        )
        (root / "core.py").write_text(
            "from base import BaseEngine\n"
            "class Engine(BaseEngine):\n"
            "    def start(self, data):\n"
            "        return self.validate(data)\n"
        )
        (root / "service.py").write_text(
            "from core import Engine\n"
            "def run_job(payload):\n"
            "    eng = Engine()\n"
            "    return eng.start(payload)\n"
        )

        tcg = TopologicalCodeGraph(root)
        total_nodes = tcg.index_repository()
        assert total_nodes >= 5
        assert "core.py::Engine" in tcg.nodes["service.py::run_job"].dependencies
        assert "service.py::run_job" in tcg.nodes["core.py::Engine"].callers
        assert "base.py::BaseEngine" in tcg.nodes["core.py::Engine"].dependencies

        slice_0 = tcg.extract_topological_slice("run_job", depth=0)
        assert list(slice_0.keys()) == ["service.py::run_job"]

        slice_1 = tcg.extract_topological_slice("run_job", depth=1)
        assert "service.py::run_job" in slice_1
        assert "core.py::Engine" in slice_1

        slice_2 = tcg.extract_topological_slice("run_job", depth=2)
        assert "base.py::BaseEngine" in slice_2
        assert set(slice_0).issubset(set(slice_1))
        assert set(slice_1).issubset(set(slice_2))


def test_token_budget_enforcement() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "mod_a.py").write_text("def alpha():\n    return '" + ("x" * 600) + "'\n")
        (root / "mod_b.py").write_text("from mod_a import alpha\ndef beta():\n    return alpha()\n")
        tcg = TopologicalCodeGraph(root)
        tcg.index_repository()
        sliced = tcg.extract_topological_slice("beta", depth=2, token_budget=60)
        assert "mod_b.py::beta" in sliced
        assert "mod_a.py::alpha" not in sliced


def test_strict_lyapunov_contraction_convergence() -> None:
    dclv = DiscreteLyapunovValidator(contraction_factor=0.80, max_turns=6)
    ok0, _ = dclv.register_turn(0, {"a.py": "h0"}, test_failures=4, unresolved_symbols=2)
    assert ok0
    assert dclv.history[-1].distance_to_target == 8.0

    ok1, msg1 = dclv.register_turn(1, {"a.py": "h1"}, test_failures=2, unresolved_symbols=1, prompt_tokens=1200)
    assert ok1 and "Contractive step valid" in msg1
    assert dclv.history[-1].smt_verified
    assert dclv.smt_proofs[-1]["witness_independently_verified"] is True

    ok2, msg2 = dclv.register_turn(
        2, {"a.py": "h2", "b.py": "h3"}, test_failures=0, unresolved_symbols=0, prompt_tokens=1100
    )
    assert ok2 and "Fixed-point manifold achieved" in msg2


def test_stagnation_and_duplicate_hash_limit_cycle_blocked() -> None:
    bcv = BanachContractionValidator(contraction_factor=0.75, max_turns=6)
    bcv.register_turn(0, {"a.py": "h0"}, test_failures=4, unresolved_symbols=0)

    ok_dup, msg_dup = bcv.register_turn(1, {"a.py": "h0"}, test_failures=4, unresolved_symbols=0)
    assert not ok_dup and "duplicate cryptographic state hash" in msg_dup

    ok_weak, msg_weak = bcv.register_turn(1, {"a.py": "h1"}, test_failures=3, unresolved_symbols=0)
    assert not ok_weak and "Contraction violated" in msg_weak


def test_transactional_history_isolation_on_rollback() -> None:
    dclv = DiscreteLyapunovValidator(contraction_factor=0.80, max_turns=6)
    dclv.register_turn(0, {"a.py": "h0"}, test_failures=4, unresolved_symbols=0)

    ok_reg, _ = dclv.register_turn(1, {"a.py": "bad"}, test_failures=6, unresolved_symbols=0)
    assert not ok_reg
    assert len(dclv.history) == 1
    assert dclv.history[-1].distance_to_target == 6.0

    ok_next, _ = dclv.register_turn(1, {"a.py": "still_bad"}, test_failures=5, unresolved_symbols=0)
    assert not ok_next
    assert len(dclv.rejected_history) == 2


def test_neutral_refactoring_step_with_taboo_guard() -> None:
    dclv = DiscreteLyapunovValidator(contraction_factor=0.75, max_turns=6, allow_neutral_refactor_steps=1)
    dclv.register_turn(0, {"a.py": "h0"}, test_failures=2, unresolved_symbols=1)

    ok_n1, msg_n1 = dclv.register_turn(1, {"a.py": "h1"}, test_failures=1, unresolved_symbols=2)
    assert ok_n1 and "Neutral refactoring step" in msg_n1

    ok_n2, _ = dclv.register_turn(2, {"a.py": "h2"}, test_failures=2, unresolved_symbols=0)
    assert not ok_n2


def test_discrete_lattice_step_bound_theorem() -> None:
    k_bound = DiscreteLyapunovValidator.theoretical_max_steps(8.0, 0.75)
    assert k_bound == 10
    val = 8.0 * (0.75 ** k_bound)
    assert val < 0.5


def test_state_adaptive_lattice_envelope() -> None:
    fixed_v = DiscreteLyapunovValidator(contraction_factor=0.75, adaptive_lattice_envelope=False)
    fixed_v.register_turn(0, {"a.py": "h0"}, test_failures=5, unresolved_symbols=0)
    ok_fixed, _ = fixed_v.register_turn(1, {"a.py": "h1"}, test_failures=4, unresolved_symbols=0)
    assert not ok_fixed

    adapt_v = DiscreteLyapunovValidator(contraction_factor=0.75, adaptive_lattice_envelope=True)
    adapt_v.register_turn(0, {"a.py": "h0"}, test_failures=5, unresolved_symbols=0)
    ok_adapt, msg_adapt = adapt_v.register_turn(1, {"a.py": "h1"}, test_failures=4, unresolved_symbols=0)
    assert ok_adapt and "Contractive step valid" in msg_adapt


def test_qf_rdl_smt_sat_and_unsat_core_extraction() -> None:
    smt = DifferenceLogicSMTVerifier()
    smt.add_constraint("v0", "__zero__", 6.0, "v0_upper")
    smt.add_constraint("__zero__", "v0", -6.0, "v0_lower")
    smt.add_constraint("v1", "__zero__", 4.0, "v1_upper")
    smt.add_constraint("__zero__", "v1", -4.0, "v1_lower")
    smt.add_constraint("v1", "v0", -1.5, "strict_contraction")
    res_sat = smt.check_sat()
    assert res_sat["status"] == "SAT" and res_sat["verified"] is True
    assert res_sat["witness_independently_verified"] is True
    assert abs(res_sat["witness"]["v1"] - res_sat["witness"]["v0"] - (-2.0)) < 1e-5

    smt_bad = DifferenceLogicSMTVerifier()
    smt_bad.add_constraint("v0", "__zero__", 6.0, "v0_upper")
    smt_bad.add_constraint("__zero__", "v0", -6.0, "v0_lower")
    smt_bad.add_constraint("v1", "__zero__", 5.5, "v1_upper")
    smt_bad.add_constraint("__zero__", "v1", -5.5, "v1_lower")
    smt_bad.add_constraint("v1", "v0", -1.5, "strict_contraction_violated")
    res_unsat = smt_bad.check_sat()
    assert res_unsat["status"] == "UNSAT" and res_unsat["verified"] is False
    assert len(res_unsat["unsat_core"]) > 0


def test_qf_rdl_token_and_blast_radius_bounds_enforced() -> None:
    dclv = DiscreteLyapunovValidator(
        contraction_factor=0.80,
        max_turns=6,
        max_blast_radius_files=2,
        max_prompt_tokens=3500,
    )
    dclv.register_turn(0, {"a.py": "h0", "b.py": "h0", "c.py": "h0"}, test_failures=4, unresolved_symbols=0)

    # Exceeds max_prompt_tokens=3500 -> rejected by QF_RDL token bound even when tests pass!
    ok_tok, msg_tok = dclv.register_turn(
        1,
        {"a.py": "h1", "b.py": "h0", "c.py": "h0"},
        test_failures=0,
        unresolved_symbols=0,
        prompt_tokens=4200,
    )
    assert not ok_tok and "max_prompt_tokens_bound" in msg_tok

    # Exceeds max_blast_radius_files=2 (3 files changed) -> rejected by QF_RDL blast bound!
    ok_blast, msg_blast = dclv.register_turn(
        1,
        {"a.py": "h1", "b.py": "h1", "c.py": "h1"},
        test_failures=0,
        unresolved_symbols=0,
        prompt_tokens=1800,
    )
    assert not ok_blast and "max_blast_radius_bound" in msg_blast


def test_deterministic_governor_stress_suite() -> None:
    exp_c = run_experiment_d_live_multifile_bugfix()
    assert exp_c["total_live_repositories"] == 15
    assert exp_c["tcgs_k2_cross_file_recall_pct"] == 100.0
    assert exp_c["wayfinder_v3_tcgs_dclv_qf_rdl"]["resolve_rate_pct"] == 100.0
    assert exp_c["wayfinder_v3_tcgs_dclv_qf_rdl"]["cyclic_thrash_rate_pct"] == 0.0
    assert (
        exp_c["wayfinder_v3_tcgs_dclv_qf_rdl"]["independently_verified_qf_rdl_witnesses"]
        == exp_c["wayfinder_v3_tcgs_dclv_qf_rdl"]["accepted_steps_count"]
    )
    assert exp_c["wayfinder_v3_tcgs_dclv_qf_rdl"]["accepted_steps_qf_rdl_sat_pct"] == 100.0
    assert exp_c["unguided_react"]["resolve_rate_pct"] < 50.0


def test_20_repo_multifile_benchmark_self_check() -> None:
    stats = run_self_check()
    assert stats["repos_verified"] == 20
    assert stats["total_unit_tests"] == 100
    assert stats["tcgs_k2_2hop_causal_chain_recall_pct"] == 100.0
    assert stats["bm25_2hop_causal_chain_recall_pct"] < stats["tcgs_k2_2hop_causal_chain_recall_pct"]


if __name__ == "__main__":
    test_cross_file_ast_dependency_and_inheritance_linking()
    test_token_budget_enforcement()
    test_strict_lyapunov_contraction_convergence()
    test_stagnation_and_duplicate_hash_limit_cycle_blocked()
    test_transactional_history_isolation_on_rollback()
    test_neutral_refactoring_step_with_taboo_guard()
    test_discrete_lattice_step_bound_theorem()
    test_state_adaptive_lattice_envelope()
    test_qf_rdl_smt_sat_and_unsat_core_extraction()
    test_qf_rdl_token_and_blast_radius_bounds_enforced()
    test_deterministic_governor_stress_suite()
    test_20_repo_multifile_benchmark_self_check()
    print("All 12 verification suites passed successfully (Exit 0)!")
