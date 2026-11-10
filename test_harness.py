"""
Unit and Integration Tests for Gemma Developer Agent Harness
"""

import tempfile
from pathlib import Path
from harness import TopologicalCodeGraph, BanachContractionValidator


def test_topological_code_graph_indexing():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        f1 = root / "core.py"
        f1.write_text("class Engine:\n    def start(self):\n        return True\n")
        f2 = root / "service.py"
        f2.write_text("from core import Engine\ndef run_job(engine):\n    return engine.start()\n")

        tcg = TopologicalCodeGraph(root)
        total_nodes = tcg.index_repository()
        assert total_nodes >= 2
        assert "core.py::Engine" in tcg.nodes
        assert "service.py::run_job" in tcg.nodes

        # Slice extraction
        sliced = tcg.extract_topological_slice("run_job", depth=1)
        assert len(sliced) >= 1


def test_banach_contraction_success():
    bcv = BanachContractionValidator(contraction_factor=0.8, max_turns=5)

    # Turn 0: initial state, 4 failures
    valid, msg = bcv.register_turn(turn=0, modified_files={"a.py": "hash1"}, test_failures=4, unresolved_symbols=2)
    assert valid

    # Turn 1: improvement, 2 failures
    valid, msg = bcv.register_turn(turn=1, modified_files={"a.py": "hash2"}, test_failures=2, unresolved_symbols=1)
    assert valid
    assert "Contractive step valid" in msg

    # Turn 2: convergence, 0 failures
    valid, msg = bcv.register_turn(turn=2, modified_files={"a.py": "hash3"}, test_failures=0, unresolved_symbols=0)
    assert valid
    assert "Fixed-point achieved" in msg


def test_banach_contraction_regression_rejection():
    bcv = BanachContractionValidator(contraction_factor=0.8, max_turns=5)

    bcv.register_turn(turn=0, modified_files={"a.py": "hash1"}, test_failures=2, unresolved_symbols=0)

    # Turn 1: regression (test failures increased to 5)
    valid, msg = bcv.register_turn(turn=1, modified_files={"a.py": "hash2"}, test_failures=5, unresolved_symbols=1)
    assert not valid
    assert "Contraction violated" in msg


if __name__ == "__main__":
    test_topological_code_graph_indexing()
    test_banach_contraction_success()
    test_banach_contraction_regression_rejection()
    print("All 3 test suites passed successfully!")
