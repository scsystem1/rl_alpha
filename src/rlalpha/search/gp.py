from __future__ import annotations

import base64
import hashlib
import importlib
import importlib.util
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..dsl.ast import Constant, Node
from ..dsl.operators import (
    BINARY,
    CONSTANTS,
    CROSS_SECTIONAL,
    FEATURES,
    PAIR_ROLLING,
    ROLLING,
    UNARY,
    WINDOWS,
)
from ..dsl.parser import parse_expression
from .models import Candidate, CandidateOutcome, SearchContext


def _dummy_fitness(y: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    """Let AlphaGen construct programs; RLAlpha supplies their real fitness."""
    del y, y_pred, weights
    return 0.0


@dataclass(frozen=True)
class _AlphaGenEngine:
    Program: type
    parallel_evolve: Callable[..., list[Any]]
    make_fitness: Callable[..., Any]
    make_function: Callable[..., Any]
    source: Path


_ENGINES: dict[Path, _AlphaGenEngine] = {}


def _load_alphagen_engine(alphagen_root: str | Path) -> _AlphaGenEngine:
    """Load AlphaGen's vendored gplearn under a collision-free module name."""
    root = Path(alphagen_root).resolve()
    cached = _ENGINES.get(root)
    if cached is not None:
        return cached
    package_root = root / "gplearn"
    init = package_root / "__init__.py"
    if not init.is_file():
        raise FileNotFoundError(f"AlphaGen modified gplearn is missing under {package_root}")
    package_name = f"_rlalpha_alphagen_gplearn_{hashlib.sha256(str(root).encode()).hexdigest()[:12]}"
    if package_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package_name,
            init,
            submodule_search_locations=[str(package_root)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load AlphaGen gplearn from {init}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        spec.loader.exec_module(module)
    program_module = importlib.import_module(f"{package_name}._program")
    genetic_module = importlib.import_module(f"{package_name}.genetic")
    fitness_module = importlib.import_module(f"{package_name}.fitness")
    functions_module = importlib.import_module(f"{package_name}.functions")
    engine = _AlphaGenEngine(
        Program=program_module._Program,
        parallel_evolve=genetic_module._parallel_evolve,
        make_fitness=fitness_module.make_fitness,
        make_function=functions_module.make_function,
        source=Path(genetic_module.__file__).resolve(),
    )
    _ENGINES[root] = engine
    return engine


def _unary_renderer(operator: str, window: int | None = None) -> Callable[[np.ndarray], np.ndarray]:
    def render(values: np.ndarray) -> np.ndarray:
        suffix = "" if window is None else f",{window}"
        return np.asarray([f"{operator}({value}{suffix})" for value in values], dtype=object)

    return render


def _binary_renderer(operator: str, window: int | None = None) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def render(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        suffix = "" if window is None else f",{window}"
        return np.asarray(
            [f"{operator}({first},{second}{suffix})" for first, second in zip(left, right, strict=True)],
            dtype=object,
        )

    return render


class GPSearcher:
    """Eight-candidate generational GP powered by AlphaGen's gplearn.

    AlphaGen owns random tree construction, tournament selection, crossover,
    subtree/hoist/point mutation and reproduction.  RLAlpha only translates
    its programs into the shared DSL and feeds the frozen-pool candidate delta
    back as fitness.  A population is exactly one proposal group, so every
    generation is scored against one pool snapshot and followed by at most one
    pool admission.
    """

    ENGINE_NAME = "alphagen_modified_gplearn"

    def __init__(
        self,
        seed: int,
        alphagen_root: str | Path,
        *,
        population_size: int = 8,
        tournament_size: int = 5,
        init_depth: tuple[int, int] = (2, 6),
        p_crossover: float = 0.5882352941,
        p_subtree_mutation: float = 0.1960784314,
        p_hoist_mutation: float = 0.0196078431,
        p_point_mutation: float = 0.1960784314,
        p_reproduction: float = 0.0,
        p_point_replace: float = 0.60,
    ):
        self.seed = int(seed)
        self.alphagen_root = Path(alphagen_root).resolve()
        self.population_size = int(population_size)
        self.tournament_size = int(tournament_size)
        self.init_depth = tuple(map(int, init_depth))
        self.p_crossover = float(p_crossover)
        self.p_subtree_mutation = float(p_subtree_mutation)
        self.p_hoist_mutation = float(p_hoist_mutation)
        self.p_point_mutation = float(p_point_mutation)
        self.p_reproduction = float(p_reproduction)
        self.p_point_replace = float(p_point_replace)
        self._validate_config()

        self.engine = _load_alphagen_engine(self.alphagen_root)
        self.rng = np.random.RandomState(self.seed)
        self.terminals = tuple(sorted(FEATURES)) + tuple(Constant(value).canonical() for value in CONSTANTS)
        self._x = np.asarray([self.terminals], dtype=object)
        self.function_set, self.function_by_name = self._build_function_set()
        self.arities: dict[int, list[Any]] = {}
        for function in self.function_set:
            self.arities.setdefault(int(function.arity), []).append(function)
        self.metric = self.engine.make_fitness(
            function=_dummy_fitness,
            greater_is_better=True,
            wrap=False,
        )
        self.population: list[Any] = []
        self.pending: list[Any] = []
        self.pending_pool_version: int | None = None
        self.generation = 0
        self.fitness_pool_version: int | None = None
        self.seen_program_hashes: set[str] = set()

    def _validate_config(self) -> None:
        if self.population_size <= 0:
            raise ValueError("GP population_size must be positive")
        if not 1 <= self.tournament_size <= self.population_size:
            raise ValueError("GP tournament_size must be in [1, population_size]")
        if len(self.init_depth) != 2 or self.init_depth[0] <= 0 or self.init_depth[1] < self.init_depth[0]:
            raise ValueError("GP init_depth must be an increasing pair of positive integers")
        probabilities = (
            self.p_crossover,
            self.p_subtree_mutation,
            self.p_hoist_mutation,
            self.p_point_mutation,
            self.p_reproduction,
        )
        if any(value < 0 or value > 1 for value in probabilities):
            raise ValueError("GP operation probabilities must be in [0, 1]")
        if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("GP operation probabilities must sum to one")
        if not 0 <= self.p_point_replace <= 1:
            raise ValueError("GP p_point_replace must be in [0, 1]")

    def _build_function_set(self) -> tuple[list[Any], dict[str, Any]]:
        functions = []
        for operator in sorted(UNARY | CROSS_SECTIONAL):
            functions.append(
                self.engine.make_function(
                    function=_unary_renderer(operator), name=operator, arity=1, wrap=False
                )
            )
        for operator in sorted(BINARY):
            functions.append(
                self.engine.make_function(
                    function=_binary_renderer(operator), name=operator, arity=2, wrap=False
                )
            )
        for operator in sorted(ROLLING):
            for window in WINDOWS:
                name = f"{operator}_{window}"
                functions.append(
                    self.engine.make_function(
                        function=_unary_renderer(operator, window), name=name, arity=1, wrap=False
                    )
                )
        for operator in sorted(PAIR_ROLLING):
            for window in WINDOWS:
                name = f"{operator}_{window}"
                functions.append(
                    self.engine.make_function(
                        function=_binary_renderer(operator, window), name=name, arity=2, wrap=False
                    )
                )
        return functions, {function.name: function for function in functions}

    def _params(self) -> dict[str, Any]:
        cumulative = np.cumsum(
            [
                self.p_crossover,
                self.p_subtree_mutation,
                self.p_hoist_mutation,
                self.p_point_mutation,
            ]
        )
        return {
            "tournament_size": self.tournament_size,
            "function_set": self.function_set,
            "arities": self.arities,
            "init_depth": self.init_depth,
            "init_method": "half and half",
            "const_range": None,
            "_metric": self.metric,
            "_transformer": None,
            "parsimony_coefficient": 0.0,
            "method_probs": cumulative,
            "p_point_replace": self.p_point_replace,
            "max_samples": 1.0,
            "feature_names": list(self.terminals),
        }

    def _expression(self, program: Any) -> str:
        rendered = program.execute(self._x)
        if rendered is None or len(rendered) != 1:
            raise RuntimeError("AlphaGen gplearn program did not render exactly one expression")
        return str(rendered[0])

    def _program_hash(self, program: Any) -> str | None:
        try:
            return parse_expression(self._expression(program)).expr_hash
        except (TypeError, ValueError):
            return None

    def _parent_hashes(self, program: Any) -> tuple[str, ...]:
        if not program.parents:
            return ()
        hashes = []
        for key in ("parent_idx", "donor_idx"):
            if key not in program.parents:
                continue
            index = int(program.parents[key])
            value = self._program_hash(self.population[index])
            if value is not None and value not in hashes:
                hashes.append(value)
        return tuple(hashes)

    def propose(self, context: SearchContext, n: int) -> list[Candidate]:
        if self.pending:
            raise RuntimeError("cannot propose a new GP generation before observing the pending generation")
        if int(n) != self.population_size:
            raise ValueError(
                f"AlphaGen GP requires proposal_group_size == population_size == {self.population_size}, got {n}"
            )
        if self.population and any(program.fitness_ is None for program in self.population):
            raise RuntimeError("AlphaGen GP parent population contains unobserved fitness")
        # AlphaGen's generic trees are syntactically closed, but its original
        # task permits deeper/more heavily nested rolling expressions than the
        # shared RLAlpha DSL.  Reject before proposal rather than charging the
        # market-evaluation budget for an engine-internal constraint mismatch.
        generated: list[Any] = []
        candidates: list[Candidate] = []
        attempts = 0
        max_attempts = self.population_size * 256
        while len(candidates) < self.population_size and attempts < max_attempts:
            batch_size = self.population_size - len(candidates)
            seeds = self.rng.randint(np.iinfo(np.int32).max, size=batch_size)
            batch = self.engine.parallel_evolve(
                batch_size,
                self.population or None,
                self._x,
                np.zeros(1, dtype=float),
                None,
                seeds,
                self._params(),
            )
            attempts += batch_size
            for program in batch:
                expression = self._expression(program)
                try:
                    node = parse_expression(expression)
                except (TypeError, ValueError):
                    continue
                if node.expr_hash in self.seen_program_hashes:
                    continue
                self.seen_program_hashes.add(node.expr_hash)
                generated.append(program)
                candidates.append(
                    Candidate(node, "gp_alphagen", self._parent_hashes(program), raw_text=expression)
                )
        if len(candidates) != self.population_size:
            raise RuntimeError(
                f"AlphaGen GP produced only {len(candidates)} valid programs in {attempts} attempts"
            )
        self.pending = generated
        self.pending_pool_version = int(context.pool_version)
        return candidates

    def observe(self, outcomes: list[CandidateOutcome]) -> None:
        if not self.pending or self.pending_pool_version is None:
            raise RuntimeError("cannot observe GP outcomes without a pending generation")
        if len(outcomes) != len(self.pending):
            raise ValueError(f"GP observed {len(outcomes)} outcomes for {len(self.pending)} pending programs")
        versions = {
            int(outcome.metadata["pre_group_pool_version"])
            for outcome in outcomes
            if "pre_group_pool_version" in outcome.metadata
        }
        if versions != {self.pending_pool_version}:
            raise ValueError(
                f"GP generation mixed frozen pool versions {sorted(versions)}; expected {self.pending_pool_version}"
            )
        for program, outcome in zip(self.pending, outcomes, strict=True):
            fitness = (
                float(outcome.delta_objective)
                if outcome.valid and outcome.market_evaluated and np.isfinite(outcome.delta_objective)
                else float("-inf")
            )
            program.raw_fitness_ = fitness
            program.fitness_ = fitness
        self.population = self.pending
        self.pending = []
        self.fitness_pool_version = self.pending_pool_version
        self.pending_pool_version = None
        self.generation += 1

    @staticmethod
    def _serialize_program(program: Any) -> list[dict[str, Any]]:
        tokens = []
        for token in program.program:
            if hasattr(token, "arity") and hasattr(token, "name"):
                tokens.append({"function": str(token.name)})
            elif isinstance(token, (int, np.integer)):
                tokens.append({"terminal": int(token)})
            else:
                raise TypeError(f"unsupported AlphaGen program token {token!r}")
        return tokens

    def _restore_program(self, state: dict[str, Any]) -> Any:
        tokens = []
        for token in state["tokens"]:
            if "function" in token:
                name = str(token["function"])
                if name not in self.function_by_name:
                    raise ValueError(f"checkpoint references unknown AlphaGen function {name}")
                tokens.append(self.function_by_name[name])
            else:
                index = int(token["terminal"])
                if not 0 <= index < len(self.terminals):
                    raise ValueError(f"checkpoint terminal index {index} is out of range")
                tokens.append(index)
        program = self.engine.Program(
            function_set=self.function_set,
            arities=self.arities,
            init_depth=self.init_depth,
            init_method="half and half",
            n_features=len(self.terminals),
            const_range=None,
            metric=self.metric,
            p_point_replace=self.p_point_replace,
            parsimony_coefficient=0.0,
            random_state=self.rng,
            feature_names=list(self.terminals),
            program=tokens,
        )
        fitness = state.get("fitness")
        program.raw_fitness_ = float("-inf") if fitness is None else float(fitness)
        program.fitness_ = program.raw_fitness_
        return program

    def _config_state(self) -> dict[str, Any]:
        return {
            "population_size": self.population_size,
            "tournament_size": self.tournament_size,
            "init_depth": list(self.init_depth),
            "p_crossover": self.p_crossover,
            "p_subtree_mutation": self.p_subtree_mutation,
            "p_hoist_mutation": self.p_hoist_mutation,
            "p_point_mutation": self.p_point_mutation,
            "p_reproduction": self.p_reproduction,
            "p_point_replace": self.p_point_replace,
        }

    def state_dict(self) -> dict[str, Any]:
        if self.pending:
            raise RuntimeError("GP checkpoints are only valid at generation boundaries")
        encoded_rng = base64.b64encode(pickle.dumps(self.rng.get_state())).decode("ascii")
        return {
            "engine": self.ENGINE_NAME,
            "engine_source": str(self.engine.source),
            "seed": self.seed,
            "rng_state": encoded_rng,
            "generation": self.generation,
            "fitness_pool_version": self.fitness_pool_version,
            "seen_program_hashes": sorted(self.seen_program_hashes),
            "config": self._config_state(),
            "population": [
                {
                    "tokens": self._serialize_program(program),
                    "fitness": float(program.fitness_) if np.isfinite(program.fitness_) else None,
                }
                for program in self.population
            ],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("engine") != self.ENGINE_NAME:
            raise ValueError("checkpoint is not an AlphaGen modified-gplearn GP state")
        if Path(str(state.get("engine_source", ""))).resolve() != self.engine.source:
            raise ValueError("AlphaGen GP checkpoint engine source changed")
        if state.get("config") != self._config_state():
            raise ValueError("AlphaGen GP checkpoint configuration changed")
        self.rng.set_state(pickle.loads(base64.b64decode(state["rng_state"])))
        self.population = [self._restore_program(item) for item in state.get("population", [])]
        if self.population and len(self.population) != self.population_size:
            raise ValueError("AlphaGen GP checkpoint population size mismatch")
        self.pending = []
        self.pending_pool_version = None
        self.generation = int(state["generation"])
        value = state.get("fitness_pool_version")
        self.fitness_pool_version = None if value is None else int(value)
        self.seen_program_hashes = set(map(str, state.get("seen_program_hashes", [])))

    @property
    def retained_hashes(self) -> set[str]:
        return {value for program in self.population if (value := self._program_hash(program)) is not None}
