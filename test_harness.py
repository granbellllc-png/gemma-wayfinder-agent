"""
Comprehensive Unit, Property-Based, and Formal QF_RDL SMT Verification Suite (v3.0)
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
    TRIZContradictionMatrixRouter,
    TopologicalCodeGraph,
)


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

        # 0-hop: seed only
        slice_0 = tcg.extract_topological_slice("run_job", depth=0)
        assert list(slice_0.keys()) == ["service.py::run_job"]

        # 1-hop: must pull in core.py::Engine and core.py::Engine.start
        slice_1 = tcg.extract_topological_slice("run_job", depth=1)
        assert "service.py::run_job" in slice_1
        assert "core.py::Engine" in slice_1

        # 2-hop: must transitively pull in base.py::BaseEngine
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
    # Turn 0: 4 test failures, 2 unresolved symbols -> V(S_0) = 4*1.5 + 2*1.0 = 8.0
    ok0, _ = dclv.register_turn(0, {"a.py": "h0"}, test_failures=4, unresolved_symbols=2)
    assert ok0
    assert dclv.history[-1].distance_to_target == 8.0

    # Turn 1: 2 failures, 1 symbol, 1 file changed -> V(S_1) = 3.0 + 1.0 + 0.5 = 4.5 <= 0.8 * 8.0 (6.4)
    ok1, msg1 = dclv.register_turn(1, {"a.py": "h1"}, test_failures=2, unresolved_symbols=1)
    assert ok1 and "Contractive step valid" in msg1
    assert dclv.history[-1].smt_verified

    # Turn 2: 0 failures, 0 unresolved -> V(S_2) = 0.0 (Fixed-point manifold achieved, churn=0 when T=0)
    ok2, msg2 = dclv.register_turn(2, {"a.py": "h2", "b.py": "h3"}, test_failures=0, unresolved_symbols=0)
    assert ok2 and "Fixed-point manifold achieved" in msg2


def test_stagnation_and_duplicate_hash_limit_cycle_blocked() -> None:
    bcv = BanachContractionValidator(contraction_factor=0.75, max_turns=6)
    bcv.register_turn(0, {"a.py": "h0"}, test_failures=4, unresolved_symbols=0)  # V = 6.0

    # Identical state hash -> blocked immediately by SHA-256 Taboo
    ok_dup, msg_dup = bcv.register_turn(1, {"a.py": "h0"}, test_failures=4, unresolved_symbols=0)
    assert not ok_dup and "duplicate cryptographic state hash" in msg_dup

    # Different hash, weak improvement V = 5.0 > 0.75 * 6.0 = 4.5 -> blocked by strict lambda
    ok_weak, msg_weak = bcv.register_turn(1, {"a.py": "h1"}, test_failures=3, unresolved_symbols=0)
    assert not ok_weak and "Contraction violated" in msg_weak


def test_transactional_history_isolation_on_rollback() -> None:
    dclv = DiscreteLyapunovValidator(contraction_factor=0.80, max_turns=6)
    dclv.register_turn(0, {"a.py": "h0"}, test_failures=4, unresolved_symbols=0)  # V = 6.0

    # Regression V = 9.5 -> rejected; must NOT pollute dclv.history[-1]
    ok_reg, _ = dclv.register_turn(1, {"a.py": "bad"}, test_failures=6, unresolved_symbols=0)
    assert not ok_reg
    assert len(dclv.history) == 1
    assert dclv.history[-1].distance_to_target == 6.0

    # Subsequent candidate V = 8.0 (< 9.5, so a polluted history would accept, but > 0.8*6.0 = 4.8!)
    ok_next, _ = dclv.register_turn(1, {"a.py": "still_bad"}, test_failures=5, unresolved_symbols=0)
    assert not ok_next
    assert len(dclv.rejected_history) == 2


def test_neutral_refactoring_step_with_taboo_guard() -> None:
    dclv = DiscreteLyapunovValidator(contraction_factor=0.75, max_turns=6, allow_neutral_refactor_steps=1)
    dclv.register_turn(0, {"a.py": "h0"}, test_failures=2, unresolved_symbols=1)  # V = 4.0

    # Neutral step: V = 1*1.5 + 2*1.0 + 0.5*1 = 4.0 (new hash, consumes 1 neutral step)
    ok_n1, msg_n1 = dclv.register_turn(1, {"a.py": "h1"}, test_failures=1, unresolved_symbols=2)
    assert ok_n1 and "Neutral refactoring step" in msg_n1

    # Second neutral step must be rejected once budget=0
    ok_n2, _ = dclv.register_turn(2, {"a.py": "h2"}, test_failures=2, unresolved_symbols=0)
    assert not ok_n2


def test_discrete_lattice_step_bound_theorem() -> None:
    # Theorem 1: For V_0 = 8.0 and lambda = 0.75, K = floor(ln(0.5 / 8.0) / ln(0.75)) + 1 = 10
    k_bound = DiscreteLyapunovValidator.theoretical_max_steps(8.0, 0.75)
    assert k_bound == 10
    val = 8.0 * (0.75 ** k_bound)
    assert val < 0.5  # Strictly below minimum positive lattice quantum -> implies V = 0


def test_state_adaptive_lattice_envelope() -> None:
    # On large initial defect V_0 = 7.5 (5 failing tests), fixing 1 test in 1 file yields V_1 = 6.5.
    # Fixed lambda=0.75 rejects 6.5 > 5.625, whereas adaptive envelope lambda_eff = max(0.75, 1 - 0.5/7.5) = 0.933 accepts 6.5 <= 7.0!
    fixed_v = DiscreteLyapunovValidator(contraction_factor=0.75, adaptive_lattice_envelope=False)
    fixed_v.register_turn(0, {"a.py": "h0"}, test_failures=5, unresolved_symbols=0)
    ok_fixed, _ = fixed_v.register_turn(1, {"a.py": "h1"}, test_failures=4, unresolved_symbols=0)
    assert not ok_fixed

    adapt_v = DiscreteLyapunovValidator(contraction_factor=0.75, adaptive_lattice_envelope=True)
    adapt_v.register_turn(0, {"a.py": "h0"}, test_failures=5, unresolved_symbols=0)
    ok_adapt, msg_adapt = adapt_v.register_turn(1, {"a.py": "h1"}, test_failures=4, unresolved_symbols=0)
    assert ok_adapt and "Contractive step valid" in msg_adapt


def test_qf_rdl_smt_sat_and_unsat_core_extraction() -> None:
    # 1. Valid contractive transition: v_0 = 6.0, v_1 = 4.0, v_1 - v_0 <= -1.5 -> SAT
    smt = DifferenceLogicSMTVerifier()
    smt.add_constraint("v0", "__zero__", 6.0, "v0_upper")
    smt.add_constraint("__zero__", "v0", -6.0, "v0_lower")
    smt.add_constraint("v1", "__zero__", 4.0, "v1_upper")
    smt.add_constraint("__zero__", "v1", -4.0, "v1_lower")
    smt.add_constraint("v1", "v0", -1.5, "strict_contraction")
    res_sat = smt.check_sat()
    assert res_sat["status"] == "SAT" and res_sat["verified"] is True
    assert abs(res_sat["witness"]["v1"] - res_sat["witness"]["v0"] - (-2.0)) < 1e-5

    # 2. Contradictory transition: v_0 = 6.0, v_1 = 5.5, but required v_1 - v_0 <= -1.5 -> UNSAT
    smt_bad = DifferenceLogicSMTVerifier()
    smt_bad.add_constraint("v0", "__zero__", 6.0, "v0_upper")
    smt_bad.add_constraint("__zero__", "v0", -6.0, "v0_lower")
    smt_bad.add_constraint("v1", "__zero__", 5.5, "v1_upper")
    smt_bad.add_constraint("__zero__", "v1", -5.5, "v1_lower")
    smt_bad.add_constraint("v1", "v0", -1.5, "strict_contraction_violated")
    res_unsat = smt_bad.check_sat()
    assert res_unsat["status"] == "UNSAT" and res_unsat["verified"] is False
    assert len(res_unsat["unsat_core"]) > 0


def test_triz_contradiction_matrix_mappings() -> None:
    m_tcgs = TRIZContradictionMatrixRouter.lookup(31, 12)
    assert m_tcgs["principles"] == [1, 35]
    m_dclv = TRIZContradictionMatrixRouter.lookup(9, 27)
    assert m_dclv["principles"] == [11, 35, 27, 28]


def test_live_multifile_bugfix_execution_suite() -> None:
    exp_d = run_experiment_d_live_multifile_bugfix()
    assert exp_d["total_live_repositories"] == 15
    assert exp_d["tcgs_k2_cross_file_recall_pct"] == 100.0
    assert exp_d["wayfinder_v3_tcgs_dclv_qf_rdl"]["resolve_rate_pct"] == 100.0
    assert exp_d["wayfinder_v3_tcgs_dclv_qf_rdl"]["cyclic_thrash_rate_pct"] == 0.0
    assert exp_d["wayfinder_v3_tcgs_dclv_qf_rdl"]["accepted_steps_qf_rdl_sat_pct"] == 100.0
    assert exp_d["unguided_react"]["resolve_rate_pct"] < 50.0


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
    test_triz_contradiction_matrix_mappings()
    test_live_multifile_bugfix_execution_suite()
    print("All 11 verification suites passed successfully (Exit 0)!")
