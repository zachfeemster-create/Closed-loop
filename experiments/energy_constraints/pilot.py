#!/usr/bin/env python3
"""Classical constraint-energy pilot inspired by CRAM; NOT a CRAM reproduction.

Hoover, Blayney & Whittington, arXiv:2611.17309, Prefrontal-AI/CRAM
(licensed Apache-2.0; this standalone pilot imports no third-party code).

Graph coloring is a common constraint core for:
- quantum temporary-register interference/ancilla reuse (toy conflict graph)
- scheduling overlapping pickup or kitchen resources (toy conflict graph)

We test a hand-specified pairwise energy, stochastic local descent, and
an independent exhaustive exact checker. No LLM, GPU, training, live app
changes or verifier claims. The energy score may guide proposals; the
EXACT checker decides feasibility.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from pathlib import Path


def checked_edges(n: int, edges):
    if not 1 <= n <= 9:
        raise ValueError("1..9 vertices only for bounded exact benchmark")
    out = set()
    for pair in edges:
        if len(pair) != 2:
            raise ValueError("Edges must be vertex pairs")
        a, b = map(int, pair)
        if a == b or min(a, b) < 0 or max(a, b) >= n:
            raise ValueError("Invalid or self-loop conflict edge")
        out.add(tuple(sorted((a, b))))
    return tuple(sorted(out))


def violated_constraints(state, edges, k: int) -> int:
    if k < 1 or any(not isinstance(c, int) or not 0 <= c < k for c in state):
        raise ValueError("State contains invalid colors")
    if any(max(a, b) >= len(state) for a, b in edges):
        raise ValueError("State missing conflicting vertex")
    return sum(state[a] == state[b] for a, b in edges)


def exact_feasible(n: int, edges, k: int):
    """Small independent feasibility oracle; None certifies infeasible at THIS k."""
    for assignment in itertools.product(range(k), repeat=n):
        if violated_constraints(assignment, edges, k) == 0:
            return assignment
    return None


def descend(n: int, edges, k: int, *, seed: int = 7,
            restarts: int = 16, moves: int = 800):
    """Local energy descent with annealed jumps and explicit early stop."""
    if restarts < 1 or moves < 1:
        raise ValueError("Must allocate positive search budget")
    rng = random.Random(seed)
    best = None
    best_e = math.inf
    visited = 0
    for _ in range(restarts):
        state = [rng.randrange(k) for _ in range(n)]
        score = violated_constraints(state, edges, k)
        for j in range(moves):
            visited += 1
            if score < best_e:
                best_e, best = score, tuple(state)
            if score == 0:  # candidate is exactly checked before acceptance
                return {"candidate": tuple(state), "violations": 0,
                        "moves": visited, "stopped_early": True}
            i = rng.randrange(n)
            # Local conditionals of pairwise energy: count incident conflicts.
            incident = [(a, b) for a, b in edges if a == i or b == i]
            current = sum(state[a] == state[b] for a, b in incident)
            candidates = list(range(k))
            rng.shuffle(candidates)
            proposals = [(sum((state[b] if a == i else state[a]) == color
                              for a, b in incident), color)
                         for color in candidates]
            low, chosen = min(proposals)
            delta = low - current
            temp = max(0.025, 0.6 * (1 - j / max(1, moves)))
            if delta <= 0 or rng.random() < math.exp(-delta / temp):
                state[i] = chosen
                score += delta
    return {"candidate": best, "violations": best_e, "moves": visited,
            "stopped_early": False}


def fixtures():
    return [
        {"name": "5-node-path", "domain": "synthetic resource conflict",
         "n": 5, "k": 2, "edges": [[0, 1], [1, 2], [2, 3], [3, 4]]},
        {"name": "triangle", "domain": "synthetic ancilla conflict",
         "n": 3, "k": 3, "edges": [[0, 1], [1, 2], [0, 2]]},
        {"name": "triangle-impossible-two-slots", "domain": "synthetic pickup slots",
         "n": 3, "k": 2, "edges": [[0, 1], [1, 2], [0, 2]]},
        {"name": "four-clique", "domain": "synthetic live temporary registers",
         "n": 4, "k": 4, "edges": [[a, b] for a in range(4) for b in range(a + 1, 4)]},
        {"name": "four-clique-impossible-three", "domain": "synthetic live temporary registers",
         "n": 4, "k": 3, "edges": [[a, b] for a in range(4) for b in range(a + 1, 4)]},
        {"name": "six-node-bipartite", "domain": "synthetic pickup slots",
         "n": 6, "k": 2, "edges": [[a, b] for a in (0, 1, 2) for b in (3, 4, 5)]},
        {"name": "odd-cycle-impossible-two", "domain": "synthetic resource conflict",
         "n": 7, "k": 2, "edges": [[i, (i + 1) % 7] for i in range(7)]},
    ]


def run(seed: int = 20261117, restarts: int = 24, moves: int = 300):
    records = []
    for case in fixtures():
        edges = checked_edges(case["n"], case["edges"])
        t0 = time.perf_counter()
        exact = exact_feasible(case["n"], edges, case["k"])
        oracle_time = time.perf_counter() - t0
        t1 = time.perf_counter()
        trial = descend(case["n"], edges, case["k"], seed=seed, restarts=restarts, moves=moves)
        solve_time = time.perf_counter() - t1
        candidate = trial["candidate"]
        candidate_valid = (
            candidate is not None and
            violated_constraints(candidate, edges, case["k"]) == 0
        )
        if candidate_valid != (trial["violations"] == 0):
            raise AssertionError("Solver mislabeled a candidate")
        records.append({
            "name": case["name"], "domain": case["domain"],
            "n": case["n"], "colors": case["k"], "conflicts": len(edges),
            "exact_feasible_at_k": exact is not None,
            "energy_candidate_valid_at_k": candidate_valid,
            "energy_violations": trial["violations"],
            "candidate": candidate,
            "energy_moves": trial["moves"], "energy_early_stop": trial["stopped_early"],
            "exact_wall_seconds": oracle_time, "energy_wall_seconds": solve_time,
            "false_accept": candidate_valid and exact is None,
            "missed_feasible": exact is not None and not candidate_valid,
        })
    return {
        "schema": "closedloop.constraint-energy-pilot.v1",
        "reference": "https://arxiv.org/abs/2611.17309",
        "reference_code": "https://github.com/Prefrontal-AI/CRAM",
        "claim": "HAND_SPECIFIED_QUBO_STYLE_ENERGY_NOT_TRAINED_CRAM",
        "constraints": "synthetic graphs only; no real quantum circuit or restaurant data",
        "source_seed": seed, "restarts": restarts, "moves_per_restart": moves,
        "records": records,
        "false_accept_count": sum(x["false_accept"] for x in records),
        "missed_feasible_count": sum(x["missed_feasible"] for x in records),
        "valid_cases": sum(x["energy_candidate_valid_at_k"] for x in records),
        "note": ("Exact oracle proves small-fixture feasibility only; neither this "
                 "test nor CRAM's GPU benchmark implies gains on VQCC or Smoke Signal."),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed", type=int, default=20261117)
    p.add_argument("--restarts", type=int, default=24)
    p.add_argument("--moves", type=int, default=300)
    args = p.parse_args()
    result = run(seed=args.seed, restarts=args.restarts, moves=args.moves)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "fixtures": len(result["records"]),
        "exactly_valid_candidates": result["valid_cases"],
        "missed_feasible": result["missed_feasible_count"],
        "false_accepts": result["false_accept_count"],
        "artifact": str(args.out),
    }))


if __name__ == "__main__":
    main()
