from __future__ import annotations

import base64
import pickle
import random
from dataclasses import dataclass
from typing import Any, Iterable

from ..dsl.ast import Call, Constant, Feature, Node, Window, validate_limits
from ..dsl.grammar import sample_ast
from ..dsl.operators import ARITY, BINARY, CONSTANTS, CROSS_SECTIONAL, FEATURES, OPERATORS, PAIR_ROLLING, ROLLING, UNARY, WINDOWS
from ..dsl.parser import parse_expression
from .models import Candidate, CandidateOutcome, SearchContext


def _paths(node: Node, prefix: tuple[int, ...] = ()) -> Iterable[tuple[tuple[int, ...], Node]]:
    yield prefix, node
    if isinstance(node, Call):
        for index, child in enumerate(node.args):
            if not isinstance(child, Window):
                yield from _paths(child, prefix + (index,))


def _replace(node: Node, path: tuple[int, ...], replacement: Node) -> Node:
    if not path:
        return replacement
    if not isinstance(node, Call):
        raise ValueError("invalid subtree path")
    args = list(node.args)
    args[path[0]] = _replace(args[path[0]], path[1:], replacement)
    return Call(node.operator, tuple(args))


def _point_mutate(node: Node, rng: random.Random) -> Node:
    choices = list(_paths(node))
    path, target = rng.choice(choices)
    if isinstance(target, Feature):
        replacement: Node = Feature(rng.choice(sorted(FEATURES - {target.name})))
    elif isinstance(target, Constant):
        replacement = Constant(rng.choice([value for value in CONSTANTS if value != target.value]))
    elif isinstance(target, Call):
        if target.operator in UNARY: family = UNARY
        elif target.operator in BINARY: family = BINARY
        elif target.operator in ROLLING: family = ROLLING
        elif target.operator in PAIR_ROLLING: family = PAIR_ROLLING
        elif target.operator in CROSS_SECTIONAL: family = CROSS_SECTIONAL
        else: family = OPERATORS
        alternatives = sorted(family - {target.operator})
        if alternatives:
            replacement = Call(rng.choice(alternatives), target.args)
        else:
            replacement = target
    else:
        replacement = target
    return _replace(node, path, replacement)


@dataclass
class _Individual:
    node: Node
    fitness: float = float("-inf")


class GPSearcher:
    def __init__(self, seed: int, population_size: int = 128, tournament_size: int = 5, elitism: int = 4):
        self.rng = random.Random(seed)
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.population: list[_Individual] = [_Individual(parse_expression(sample_ast(self.rng).canonical())) for _ in range(population_size)]
        self.pending: dict[str, Node] = {}
        self.pool_version = -1
        self.stale: list[str] = [item.node.expr_hash for item in self.population]

    def _tournament(self) -> _Individual:
        sample = self.rng.sample(self.population, min(self.tournament_size, len(self.population)))
        return max(sample, key=lambda item: item.fitness)

    def _valid(self, node: Node) -> bool:
        try:
            validate_limits(node)
            return True
        except ValueError:
            return False

    def _offspring(self) -> Candidate:
        draw = self.rng.random()
        first = self._tournament()
        parents = (first.node.expr_hash,)
        node = first.node
        if draw < 0.50:
            second = self._tournament()
            left_path, _ = self.rng.choice(list(_paths(first.node)))
            _, right_subtree = self.rng.choice(list(_paths(second.node)))
            try:
                candidate = _replace(first.node, left_path, right_subtree)
                if self._valid(candidate): node = candidate
            except ValueError:
                pass
            parents = (first.node.expr_hash, second.node.expr_hash)
        elif draw < 0.75:
            path, _ = self.rng.choice(list(_paths(first.node)))
            try:
                candidate = _replace(first.node, path, sample_ast(self.rng, max_depth=3))
                if self._valid(candidate): node = candidate
            except ValueError:
                pass
        elif draw < 0.90:
            try:
                candidate = _point_mutate(first.node, self.rng)
                if self._valid(candidate): node = candidate
            except ValueError:
                pass
        return Candidate(parse_expression(node.canonical()), "gp", parents)

    def propose(self, context: SearchContext, n: int) -> list[Candidate]:
        if context.pool_version != self.pool_version:
            self.pool_version = context.pool_version
            for item in self.population:
                item.fitness = float("-inf")
            self.stale = [item.node.expr_hash for item in self.population]
        if self.stale:
            by_hash = {item.node.expr_hash: item.node for item in self.population}
            hashes, self.stale = self.stale[:n], self.stale[n:]
            proposed = [Candidate(by_hash[value], "gp_rescore") for value in hashes]
        else:
            proposed = [self._offspring() for _ in range(n)]
        self.pending.update({item.expr_hash: item.node for item in proposed})
        return proposed

    def observe(self, outcomes: list[CandidateOutcome]) -> None:
        fitness = {outcome.expr_hash: outcome.delta_objective for outcome in outcomes if outcome.expr_hash in self.pending and outcome.market_evaluated}
        for item in self.population:
            if item.node.expr_hash in fitness:
                item.fitness = fitness[item.node.expr_hash]
        population_hashes = {item.node.expr_hash for item in self.population}
        additions = [_Individual(self.pending[outcome.expr_hash], outcome.delta_objective) for outcome in outcomes if outcome.expr_hash in self.pending and outcome.market_evaluated and outcome.expr_hash not in population_hashes]
        ranked = sorted(self.population + additions, key=lambda item: item.fitness, reverse=True)
        unique: dict[str, _Individual] = {}
        for item in ranked:
            unique.setdefault(item.node.expr_hash, item)
        self.population = list(unique.values())[: self.population_size]
        while len(self.population) < self.population_size:
            self.population.append(_Individual(parse_expression(sample_ast(self.rng).canonical())))
        self.pending.clear()

    def state_dict(self) -> dict[str, Any]:
        encoded = base64.b64encode(pickle.dumps(self.rng.getstate())).decode("ascii")
        return {"rng_state": encoded, "population": [{"expression": item.node.canonical(), "fitness": item.fitness} for item in self.population], "pool_version": self.pool_version, "stale": self.stale}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rng.setstate(pickle.loads(base64.b64decode(state["rng_state"])))
        self.population = [_Individual(parse_expression(item["expression"]), float(item["fitness"])) for item in state["population"]]
        self.pool_version = int(state["pool_version"])
        self.stale = list(state.get("stale", []))

    @property
    def retained_hashes(self) -> set[str]:
        return {item.node.expr_hash for item in self.population}
