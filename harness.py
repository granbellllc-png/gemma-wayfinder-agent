"""
Topological Code-Graph Slicing (TCGS), Discrete Contractive Lyapunov Verification (DCLV),
and Quantifier-Free Real Difference Logic (QF_RDL) Formal SMT Verification (v3.1)
Target Architecture: On-Device Google Gemma 4 SWE Agents (E2B, E4B, 12B, 26B A4B MoE, 31B)
License: Apache License 2.0
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """Deterministic subword token estimator (~3.8 chars/token for Python source)."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 3.8))


@dataclass(frozen=True)
class DifferenceConstraint:
    """
    Represents a Quantifier-Free Real Difference Logic (QF_RDL) inequality:
        x_var - y_var <= bound   (with optional human-readable label).
    """

    x_var: str
    y_var: str
    bound: float
    label: str = ""


class DifferenceLogicSMTVerifier:
    """
    Exact Bellman-Ford Quantifier-Free Real Difference Logic (DPLL(QF_RDL)) solver.
    Decides satisfiability of conjunctions of linear difference constraints:
        bigwedge_k (x_{i_k} - x_{j_k} <= c_k)
    by constructing a weighted constraint graph G_c and searching for negative-weight cycles.
    - If no negative cycle exists, returns status='SAT' with an exact witness assignment
      pi: Vars -> R independently verified against every inequality pi(x) - pi(y) <= c.
    - If a negative cycle exists, extracts a minimal cyclic UNSAT core.
    """

    EPSILON: float = 1e-9

    def __init__(self) -> None:
        self.constraints: List[DifferenceConstraint] = []

    def add_constraint(self, x_var: str, y_var: str, bound: float, label: str = "") -> None:
        self.constraints.append(
            DifferenceConstraint(x_var=x_var, y_var=y_var, bound=float(bound), label=label)
        )

    def clear(self) -> None:
        self.constraints.clear()

    def verify_witness(self, witness: Dict[str, float], tol: float = 1e-6) -> bool:
        """Independently checks that witness pi satisfies pi(x) - pi(y) <= bound for all constraints."""
        for c in self.constraints:
            vx = 0.0 if c.x_var == "__zero__" else float(witness.get(c.x_var, math.nan))
            vy = 0.0 if c.y_var == "__zero__" else float(witness.get(c.y_var, math.nan))
            if math.isnan(vx) or math.isnan(vy):
                return False
            if (vx - vy) > c.bound + tol:
                return False
        return True

    def check_sat(self) -> Dict[str, Any]:
        variables: Set[str] = {"__zero__"}
        for c in self.constraints:
            variables.add(c.x_var)
            variables.add(c.y_var)

        ordered_vars = sorted(variables)
        dist: Dict[str, float] = {v: 0.0 for v in ordered_vars}
        pred_var: Dict[str, Optional[str]] = {v: None for v in ordered_vars}
        pred_constraint: Dict[str, Optional[DifferenceConstraint]] = {v: None for v in ordered_vars}

        n = len(ordered_vars)
        last_relaxed_var: Optional[str] = None

        # Bellman-Ford relaxation over n iterations
        for _ in range(n):
            last_relaxed_var = None
            for c in self.constraints:
                # x - y <= bound  <=>  dist[x] <= dist[y] + bound (edge y -> x with weight bound)
                cand = dist[c.y_var] + c.bound
                if cand < dist[c.x_var] - self.EPSILON:
                    dist[c.x_var] = cand
                    pred_var[c.x_var] = c.y_var
                    pred_constraint[c.x_var] = c
                    last_relaxed_var = c.x_var
            if last_relaxed_var is None:
                break

        if last_relaxed_var is not None:
            # Reconstruct negative-weight cycle for UNSAT core
            curr: Optional[str] = last_relaxed_var
            for _ in range(n):
                if curr is not None:
                    curr = pred_var[curr]

            cycle_constraints: List[str] = []
            visited: Set[str] = set()
            walk = curr
            while walk is not None and walk not in visited:
                visited.add(walk)
                pc = pred_constraint.get(walk)
                if pc is not None:
                    desc = pc.label or f"{pc.x_var} - {pc.y_var} <= {pc.bound:.4f}"
                    cycle_constraints.append(desc)
                walk = pred_var.get(walk)

            return {
                "status": "UNSAT",
                "verified": False,
                "witness_independently_verified": False,
                "constraints_checked": len(self.constraints),
                "unsat_core": list(reversed(cycle_constraints)),
                "witness": {},
            }

        # Normalize witness potentials relative to __zero__ = 0.0
        zero_offset = dist.get("__zero__", 0.0)
        witness = {
            v: round(dist[v] - zero_offset, 6)
            for v in ordered_vars
            if v != "__zero__"
        }
        witness_ok = self.verify_witness(witness)
        return {
            "status": "SAT" if witness_ok else "UNSAT",
            "verified": witness_ok,
            "witness_independently_verified": witness_ok,
            "constraints_checked": len(self.constraints),
            "unsat_core": [] if witness_ok else ["witness_verification_failed"],
            "witness": witness if witness_ok else {},
        }


@dataclass
class CodeNode:
    name: str
    kind: str  # 'function', 'class', 'method'
    file_path: str
    lineno: int
    end_lineno: int
    dependencies: Set[str] = field(default_factory=set)
    callers: Set[str] = field(default_factory=set)
    signature: str = ""
    docstring: str = ""
    source_snippet: str = ""
    method_signatures: List[str] = field(default_factory=list)

    @property
    def skeleton_text(self) -> str:
        lines: List[str] = [f"{self.signature}:"]
        if self.docstring:
            lines.append(f'    """{self.docstring}"""')
        if self.method_signatures:
            lines.extend(self.method_signatures)
        elif not self.docstring:
            return f"{self.signature}: ..."
        else:
            lines.append("    ...")
        return "\n".join(lines)

    @property
    def token_estimate(self) -> int:
        return estimate_tokens(self.source_snippet or f"{self.signature}\n{self.docstring}")

    @property
    def skeleton_token_estimate(self) -> int:
        return estimate_tokens(f"{self.signature}\n{self.docstring}".strip())


class TopologicalCodeGraph:
    """
    Constructs a deterministic, execution-free directed AST dependency manifold G = (V, E)
    across multi-file Python repositories without executing untrusted third-party code.
    """

    EXCLUDED_DIRS: Tuple[str, ...] = (
        ".git",
        "__pycache__",
        "venv",
        ".venv",
        "node_modules",
        "dist",
        "build",
    )

    def __init__(self, root_dir: Path, max_hub_degree: int = 24):
        self.root_dir = Path(root_dir).resolve()
        self.max_hub_degree = max_hub_degree
        self.nodes: Dict[str, CodeNode] = {}
        self.file_to_nodes: Dict[str, List[str]] = {}
        self._file_trees: Dict[str, ast.Module] = {}

    def index_repository(self, extensions: Tuple[str, ...] = (".py",)) -> int:
        self.nodes.clear()
        self.file_to_nodes.clear()
        self._file_trees.clear()

        for p in sorted(self.root_dir.rglob("*")):
            if not p.is_file() or p.suffix not in extensions:
                continue
            rel_parts = p.relative_to(self.root_dir).parts
            if any(part.startswith(".") or part in self.EXCLUDED_DIRS for part in rel_parts):
                continue
            try:
                self._parse_file(p)
            except Exception as exc:
                logger.debug("Skipping %s: %s", p, exc)

        self._link_dependencies()
        self._file_trees.clear()
        return len(self.nodes)

    def _parse_file(self, file_path: Path) -> None:
        rel_path = file_path.relative_to(self.root_dir).as_posix()
        content = file_path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(content, filename=rel_path)
        except SyntaxError:
            return

        lines = content.splitlines()
        self._file_trees[rel_path] = tree
        self.file_to_nodes[rel_path] = []

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._register_callable(rel_path, lines, node, prefix="")
            elif isinstance(node, ast.ClassDef):
                cls_id = f"{rel_path}::{node.name}"
                end_ln = getattr(node, "end_lineno", node.lineno) or node.lineno
                bases = [ast.unparse(b) if hasattr(ast, "unparse") else "" for b in node.bases]
                base_sig = f"({', '.join(b for b in bases if b)})" if any(bases) else ""
                snippet = "\n".join(lines[node.lineno - 1 : min(end_ln, node.lineno + 35)])
                cnode = CodeNode(
                    name=node.name,
                    kind="class",
                    file_path=rel_path,
                    lineno=node.lineno,
                    end_lineno=end_ln,
                    signature=f"class {node.name}{base_sig}",
                    docstring=(ast.get_docstring(node) or "")[:360],
                    source_snippet=snippet,
                )
                self.nodes[cls_id] = cnode
                self.file_to_nodes[rel_path].append(cls_id)

                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_id = self._register_callable(rel_path, lines, item, prefix=f"{node.name}.")
                        cnode.dependencies.add(method_id)
                        mnode = self.nodes[method_id]
                        mnode.callers.add(cls_id)
                        dec_lines: List[str] = []
                        for dec in getattr(item, "decorator_list", []):
                            dname = dec.id if isinstance(dec, ast.Name) else (dec.attr if isinstance(dec, ast.Attribute) else "")
                            if dname in ("staticmethod", "classmethod", "property"):
                                dec_lines.append(f"    @{dname}")
                        clean_msig = (
                            mnode.signature.replace(f"def {node.name}.", "def ", 1)
                            .replace(f"async def {node.name}.", "async def ", 1)
                        )
                        if mnode.docstring:
                            mskel = "\n".join(dec_lines + [f"    {clean_msig}:", f'        """{mnode.docstring}"""', "        ..."])
                        else:
                            mskel = "\n".join(dec_lines + [f"    {clean_msig}: ..."])
                        cnode.method_signatures.append(mskel)

    def _register_callable(
        self,
        rel_path: str,
        lines: List[str],
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        prefix: str,
    ) -> str:
        qual_name = f"{prefix}{node.name}"
        node_id = f"{rel_path}::{qual_name}"
        end_ln = getattr(node, "end_lineno", node.lineno) or node.lineno
        args = [a.arg for a in node.args.args]
        kwonly = [a.arg for a in node.args.kwonlyargs]
        all_args = args + ([f"*, {', '.join(kwonly)}"] if kwonly else [])
        async_kw = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
        ret_ann = ""
        if getattr(node, "returns", None) is not None and hasattr(ast, "unparse"):
            try:
                ret_ann = f" -> {ast.unparse(node.returns)}"
            except Exception:
                ret_ann = ""
        snippet = "\n".join(lines[node.lineno - 1 : min(end_ln, node.lineno + 45)])
        cnode = CodeNode(
            name=node.name,
            kind="method" if prefix else "function",
            file_path=rel_path,
            lineno=node.lineno,
            end_lineno=end_ln,
            signature=f"{async_kw}def {qual_name}({', '.join(all_args)}){ret_ann}",
            docstring=(ast.get_docstring(node) or "")[:360],
            source_snippet=snippet,
        )
        self.nodes[node_id] = cnode
        self.file_to_nodes[rel_path].append(node_id)
        return node_id

    def _link_dependencies(self) -> None:
        symbol_index: Dict[str, List[str]] = {}
        module_stem_to_files: Dict[str, List[str]] = {}

        for nid, cnode in self.nodes.items():
            symbol_index.setdefault(cnode.name, []).append(nid)
            qual = nid.split("::", 1)[1]
            if qual != cnode.name:
                symbol_index.setdefault(qual, []).append(nid)

        for rel_path in self.file_to_nodes:
            stem = Path(rel_path).stem
            module_stem_to_files.setdefault(stem, []).append(rel_path)

        for rel_path, tree in self._file_trees.items():
            imported_symbols: Dict[str, Optional[str]] = {}
            for top in ast.walk(tree):
                if isinstance(top, ast.ImportFrom):
                    mod_stem = (top.module or "").split(".")[-1] or None
                    for alias in top.names:
                        local_name = alias.asname or alias.name
                        imported_symbols[local_name] = mod_stem
                elif isinstance(top, ast.Import):
                    for alias in top.names:
                        mod_stem = alias.name.split(".")[-1]
                        local_name = alias.asname or mod_stem
                        imported_symbols[local_name] = mod_stem

            for top in tree.body:
                if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self._bind_ast_scope(
                        f"{rel_path}::{top.name}", top, rel_path, imported_symbols, symbol_index, module_stem_to_files
                    )
                elif isinstance(top, ast.ClassDef):
                    cls_id = f"{rel_path}::{top.name}"
                    for base in top.bases:
                        if isinstance(base, ast.Name):
                            self._add_resolved_edges(
                                cls_id, base.id, rel_path, imported_symbols, symbol_index, module_stem_to_files
                            )
                        elif isinstance(base, ast.Attribute):
                            self._add_resolved_edges(
                                cls_id, base.attr, rel_path, imported_symbols, symbol_index, module_stem_to_files
                            )
                    for item in top.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            m_id = f"{rel_path}::{top.name}.{item.name}"
                            self._bind_ast_scope(
                                m_id, item, rel_path, imported_symbols, symbol_index, module_stem_to_files
                            )

    def _bind_ast_scope(
        self,
        src_id: str,
        scope_node: ast.AST,
        rel_path: str,
        imported_symbols: Dict[str, Optional[str]],
        symbol_index: Dict[str, List[str]],
        module_stem_to_files: Dict[str, List[str]],
    ) -> None:
        if src_id not in self.nodes:
            return
        referenced: Set[str] = set()
        for sub in ast.walk(scope_node):
            if isinstance(sub, ast.Call):
                if isinstance(sub.func, ast.Name):
                    referenced.add(sub.func.id)
                elif isinstance(sub.func, ast.Attribute):
                    referenced.add(sub.func.attr)
            elif isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                if sub.id in imported_symbols or sub.id in symbol_index:
                    referenced.add(sub.id)

        for sym in referenced:
            self._add_resolved_edges(src_id, sym, rel_path, imported_symbols, symbol_index, module_stem_to_files)

    def _add_resolved_edges(
        self,
        src_id: str,
        symbol: str,
        rel_path: str,
        imported_symbols: Dict[str, Optional[str]],
        symbol_index: Dict[str, List[str]],
        module_stem_to_files: Dict[str, List[str]],
    ) -> None:
        candidates = symbol_index.get(symbol, [])
        if not candidates:
            return

        preferred_files: Set[str] = {rel_path}
        if symbol in imported_symbols and imported_symbols[symbol]:
            mod_stem = imported_symbols[symbol]
            for fpath in module_stem_to_files.get(mod_stem or "", []):
                preferred_files.add(fpath)

        scoped = [cid for cid in candidates if self.nodes[cid].file_path in preferred_files]
        targets = scoped if scoped else (candidates if len(candidates) <= 4 else [])

        for dst_id in targets:
            if dst_id != src_id:
                self.nodes[src_id].dependencies.add(dst_id)
                self.nodes[dst_id].callers.add(src_id)

    def extract_topological_slice_with_hops(
        self,
        target_symbol: str,
        depth: int = 2,
        token_budget: int = 3500,
        skeletonize_distant_hops: bool = True,
    ) -> Tuple[Dict[str, CodeNode], Dict[str, int]]:
        if target_symbol in self.nodes:
            matched = [target_symbol]
        else:
            exact = [
                nid
                for nid in self.nodes
                if nid.endswith(f"::{target_symbol}") or self.nodes[nid].name == target_symbol
            ]
            matched = exact if exact else [nid for nid in self.nodes if target_symbol in nid]
        if not matched:
            return {}, {}

        selected: Dict[str, CodeNode] = {}
        hop_map: Dict[str, int] = {}
        used_tokens = 0
        frontier: List[str] = sorted(matched)

        for hop in range(depth + 1):
            next_frontier: Set[str] = set()
            for nid in frontier:
                if nid in selected or nid not in self.nodes:
                    continue
                node = self.nodes[nid]
                node_tokens = (
                    node.skeleton_token_estimate
                    if (skeletonize_distant_hops and hop >= 2)
                    else node.token_estimate
                )
                if selected and (used_tokens + node_tokens > token_budget):
                    continue
                selected[nid] = node
                hop_map[nid] = hop
                used_tokens += node_tokens

                if hop < depth:
                    neighbors = sorted(
                        node.dependencies | node.callers,
                        key=lambda x: (
                            len(self.nodes[x].dependencies) + len(self.nodes[x].callers)
                            if x in self.nodes
                            else 999,
                            x,
                        ),
                    )
                    for nb in neighbors[: self.max_hub_degree]:
                        if nb not in selected:
                            next_frontier.add(nb)
            frontier = sorted(next_frontier)
            if not frontier:
                break

        return selected, hop_map

    def extract_topological_slice(
        self,
        target_symbol: str,
        depth: int = 2,
        token_budget: int = 3500,
        skeletonize_distant_hops: bool = True,
    ) -> Dict[str, CodeNode]:
        """
        Extracts a bounded k-hop subgraph G_k = (V_k, E_k) centered around target_symbol,
        pruning high-degree utility hubs and respecting the active KV-cache token_budget.
        For hop >= 2 when skeletonize_distant_hops=True, emits typed interface skeletons
        (signature + docstring) to maximize transitive structural recall per token.
        """
        selected, _ = self.extract_topological_slice_with_hops(
            target_symbol=target_symbol,
            depth=depth,
            token_budget=token_budget,
            skeletonize_distant_hops=skeletonize_distant_hops,
        )
        return selected

    def render_slice_context(
        self,
        target_symbol: str,
        depth: int = 2,
        token_budget: int = 3500,
        skeletonize_distant_hops: bool = True,
    ) -> Tuple[str, Dict[str, CodeNode]]:
        """
        Renders a structured prompt context from the k-hop topological slice:
        - Files with any hop <= 1 node include full file source (if within budget) or full AST bodies.
        - Files with only hop >= 2 nodes include typed interface skeletons.
        """
        selected, hop_map = self.extract_topological_slice_with_hops(
            target_symbol=target_symbol,
            depth=depth,
            token_budget=token_budget,
            skeletonize_distant_hops=skeletonize_distant_hops,
        )
        if not selected:
            return "", {}

        files_min_hop: Dict[str, int] = {}
        files_nodes: Dict[str, List[CodeNode]] = {}
        for nid, node in selected.items():
            h = hop_map[nid]
            fp = node.file_path
            files_min_hop[fp] = min(h, files_min_hop.get(fp, h))
            files_nodes.setdefault(fp, []).append(node)

        blocks: List[str] = []
        for fp in sorted(files_min_hop.keys(), key=lambda p: (files_min_hop[p], p)):
            min_h = files_min_hop[fp]
            if min_h <= 1 or not skeletonize_distant_hops:
                full_text = (self.root_dir / fp).read_text(encoding="utf-8", errors="replace").strip()
                blocks.append(f"# === FILE: {fp} (Hop {min_h} - Editable Full Source) ===\n{full_text}")
            else:
                cls_names_in_file = {n.name for n in files_nodes[fp] if n.kind == "class"}
                skel_items: List[str] = []
                for n in sorted(files_nodes[fp], key=lambda x: x.lineno):
                    if n.kind == "method" and any(
                        n.signature.startswith(f"def {cn}.") or n.signature.startswith(f"async def {cn}.")
                        for cn in cls_names_in_file
                    ):
                        continue
                    skel_items.append(n.skeleton_text)
                skels = "\n\n".join(skel_items)
                blocks.append(
                    f"# === INTERFACE SUMMARY: {fp} (Hop {min_h} - Already Implemented Dependency; Call Directly, Do NOT Edit) ===\n{skels}"
                )

        return "\n\n".join(blocks), selected


@dataclass
class AgentEditState:
    turn: int
    modified_files: Dict[str, str]  # path -> sha256 content hash
    test_failures: int
    unresolved_symbols: int
    distance_to_target: float
    prompt_tokens: int = 0
    state_hash: str = ""
    smt_verified: bool = True


class DiscreteLyapunovValidator:
    """
    Enforces Discrete Contractive Lyapunov Verification (DCLV) on lattice (1/2) N_0:
        V(S_t) = alpha * T_t + beta * U_t + gamma * I(T_t > 0) * |F_t Delta F_0|
    coupled with formal Quantifier-Free Real Difference Logic (DPLL(QF_RDL)) verification.
    Guarantees finite-step termination to the zero-defect manifold S* = {S | V(S) = 0}
    in at most K = floor(ln(0.5 / V(S_0)) / ln(lambda_0)) + 1 accepted steps (or 2*V_0 + B_neutral
    under the State-Adaptive Lattice Envelope), with cryptographic state-hash Taboo filtering
    that blocks 100% of cyclic revisits.
    """

    ALPHA: float = 1.5
    BETA: float = 1.0
    GAMMA: float = 0.5
    LATTICE_QUANTUM: float = 0.5

    def __init__(
        self,
        contraction_factor: float = 0.75,
        max_turns: int = 6,
        allow_neutral_refactor_steps: int = 0,
        baseline_files: Optional[Dict[str, str]] = None,
        adaptive_lattice_envelope: bool = False,
        max_blast_radius_files: int = 12,
        max_prompt_tokens: int = 3500,
    ):
        if not (0.0 < contraction_factor < 1.0):
            raise ValueError("contraction_factor lambda must satisfy 0 < lambda < 1")
        self.contraction_factor = contraction_factor
        self.max_turns = max_turns
        self.initial_neutral_budget = allow_neutral_refactor_steps
        self.remaining_neutral_steps = allow_neutral_refactor_steps
        self.baseline_files: Dict[str, str] = dict(baseline_files or {})
        self.adaptive_lattice_envelope = adaptive_lattice_envelope
        self.max_blast_radius_files = max_blast_radius_files
        self.max_prompt_tokens = max_prompt_tokens
        self.history: List[AgentEditState] = []
        self.rejected_history: List[AgentEditState] = []
        self.seen_hashes: Set[str] = set()
        self.smt_proofs: List[Dict[str, Any]] = []

    @staticmethod
    def compute_state_hash(modified_files: Dict[str, str], test_failures: int, unresolved_symbols: int) -> str:
        payload = json.dumps(
            {
                "files": dict(sorted(modified_files.items())),
                "t": test_failures,
                "u": unresolved_symbols,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def theoretical_max_steps(cls, initial_potential: float, contraction_factor: float) -> int:
        if initial_potential <= 0.0:
            return 0
        if initial_potential < cls.LATTICE_QUANTUM:
            return 1
        return int(math.floor(math.log(cls.LATTICE_QUANTUM / initial_potential) / math.log(contraction_factor)) + 1)

    def _changed_file_count(self, modified_files: Dict[str, str]) -> int:
        if not self.baseline_files:
            return len(modified_files)
        changed = 0
        all_keys = set(self.baseline_files) | set(modified_files)
        for k in all_keys:
            if self.baseline_files.get(k) != modified_files.get(k):
                changed += 1
        return changed

    def compute_state_distance(
        self,
        test_failures: int,
        unresolved_symbols: int,
        file_delta_count: int = 0,
    ) -> float:
        churn = self.GAMMA * file_delta_count if test_failures > 0 else 0.0
        return float(self.ALPHA * test_failures + self.BETA * unresolved_symbols + churn)

    def verify_step_qf_rdl(
        self,
        v_prev: float,
        v_cand: float,
        strict_bound: float,
        delta_files: int,
        prompt_tokens: int,
        is_neutral_allowed: bool,
    ) -> Dict[str, Any]:
        """
        Formally checks transition admissibility in QF_RDL difference logic:
          1. V(cand) - V(prev) <= Delta_allowed(prev)   [Lyapunov contraction envelope]
          2. 0 - V(cand) <= 0                           [Lattice non-negativity]
          3. |F_cand Delta F_0| - 0 <= B_files          [Syntactic blast-radius ceiling]
          4. tokens(cand) - 0 <= B_token                [Active KV-cache prompt ceiling]
        """
        smt = DifferenceLogicSMTVerifier()
        # Pin observed scalars relative to reference __zero__ = 0.0
        smt.add_constraint("v_prev", "__zero__", v_prev, "v_prev_upper")
        smt.add_constraint("__zero__", "v_prev", -v_prev, "v_prev_lower")
        smt.add_constraint("v_cand", "__zero__", v_cand, "v_cand_upper")
        smt.add_constraint("__zero__", "v_cand", -v_cand, "v_cand_lower")
        smt.add_constraint("__zero__", "v_cand", 0.0, "non_negative_potential")

        allowed_diff = 0.0 if is_neutral_allowed else (strict_bound - v_prev)
        smt.add_constraint("v_cand", "v_prev", allowed_diff + 1e-9, "lyapunov_contraction_envelope")

        smt.add_constraint("blast_radius", "__zero__", float(delta_files), "blast_obs_upper")
        smt.add_constraint("__zero__", "blast_radius", -float(delta_files), "blast_obs_lower")
        smt.add_constraint(
            "blast_radius",
            "__zero__",
            float(self.max_blast_radius_files),
            "max_blast_radius_bound",
        )

        smt.add_constraint("prompt_tokens", "__zero__", float(prompt_tokens), "tokens_obs_upper")
        smt.add_constraint("__zero__", "prompt_tokens", -float(prompt_tokens), "tokens_obs_lower")
        smt.add_constraint(
            "prompt_tokens",
            "__zero__",
            float(self.max_prompt_tokens),
            "max_prompt_tokens_bound",
        )
        return smt.check_sat()

    def register_turn(
        self,
        turn: int,
        modified_files: Dict[str, str],
        test_failures: int,
        unresolved_symbols: int,
        prompt_tokens: int = 0,
    ) -> Tuple[bool, str]:
        if not self.history and not self.baseline_files:
            self.baseline_files = dict(modified_files)

        delta_files = 0 if not self.history else self._changed_file_count(modified_files)
        dist = self.compute_state_distance(test_failures, unresolved_symbols, delta_files)
        sig = self.compute_state_hash(modified_files, test_failures, unresolved_symbols)

        state = AgentEditState(
            turn=turn,
            modified_files=dict(modified_files),
            test_failures=test_failures,
            unresolved_symbols=unresolved_symbols,
            distance_to_target=dist,
            prompt_tokens=prompt_tokens,
            state_hash=sig,
        )

        if not self.history:
            self.history.append(state)
            self.seen_hashes.add(sig)
            return True, f"Initial state registered: V(S_0) = {dist:.2f}"

        prev_state = self.history[-1]

        if dist == 0.0:
            smt_res = self.verify_step_qf_rdl(
                v_prev=prev_state.distance_to_target,
                v_cand=0.0,
                strict_bound=0.0,
                delta_files=delta_files,
                prompt_tokens=prompt_tokens,
                is_neutral_allowed=False,
            )
            self.smt_proofs.append(smt_res)
            if not smt_res["verified"]:
                state.smt_verified = False
                self.rejected_history.append(state)
                core = ", ".join(smt_res.get("unsat_core", []))
                return False, f"QF_RDL invariant violated at candidate zero state ({core})."
            state.smt_verified = True
            self.history.append(state)
            self.seen_hashes.add(sig)
            return True, "Fixed-point manifold achieved: zero test failures and resolved symbols (QF_RDL SAT)"

        if sig in self.seen_hashes:
            state.smt_verified = False
            self.rejected_history.append(state)
            return False, "Contraction violated: duplicate cryptographic state hash (limit-cycle blocked)."

        eff_lambda = self.contraction_factor
        if self.adaptive_lattice_envelope and prev_state.distance_to_target > self.LATTICE_QUANTUM:
            eff_lambda = max(
                self.contraction_factor,
                1.0 - (self.LATTICE_QUANTUM / prev_state.distance_to_target),
            )
        strict_bound = eff_lambda * prev_state.distance_to_target
        is_neutral_candidate = (
            math.isclose(dist, prev_state.distance_to_target) and self.remaining_neutral_steps > 0
        )

        smt_res = self.verify_step_qf_rdl(
            v_prev=prev_state.distance_to_target,
            v_cand=dist,
            strict_bound=strict_bound,
            delta_files=delta_files,
            prompt_tokens=prompt_tokens,
            is_neutral_allowed=is_neutral_candidate,
        )
        self.smt_proofs.append(smt_res)

        if dist <= strict_bound + 1e-9 and smt_res["verified"]:
            if turn > self.max_turns:
                state.smt_verified = False
                self.rejected_history.append(state)
                return False, f"Max turns ({self.max_turns}) exceeded without reaching zero-defect manifold."
            state.smt_verified = True
            self.history.append(state)
            self.seen_hashes.add(sig)
            ratio = dist / prev_state.distance_to_target if prev_state.distance_to_target > 0 else 0.0
            return True, f"Contractive step valid (QF_RDL SAT): decay ratio = {ratio:.2f} <= {eff_lambda:.2f}"

        if is_neutral_candidate and smt_res["verified"]:
            self.remaining_neutral_steps -= 1
            state.smt_verified = True
            self.history.append(state)
            self.seen_hashes.add(sig)
            return True, "Neutral refactoring step accepted under bounded Taboo & QF_RDL guard"

        state.smt_verified = False
        self.rejected_history.append(state)
        core = ", ".join(smt_res.get("unsat_core", []))
        return (
            False,
            f"Contraction violated: V={dist:.2f} > bound={strict_bound:.2f} (UNSAT core: {core}). Reverting state.",
        )


# Backward-compatible alias
BanachContractionValidator = DiscreteLyapunovValidator
