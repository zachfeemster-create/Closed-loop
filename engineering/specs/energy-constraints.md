# Spec: independently check the CPU constraint-energy pilot

## Problem statement
The original Closed Loop now has a classical, hand-defined energy-descent prototype. An agent must not describe an energy-minimum guess as a proven quantum advantage, or declare an invalid graph-coloring assignment feasible. Every change to this module needs deterministic source-based acceptance evidence.

## Non-goals
No reproduction of trained CRAM, no model fine-tuning, no GPU purchases, no new Kaggle ZIP or score, no autonomous editing of the live Smoke Signal application, and no claim that a tiny graph proves a quantum optimization. Never overwrite historical Closed Loop experiments.

## Assumptions
The repository provides Python 3.12 and the existing files in experiments/energy_constraints and tests/test_energy_constraints.py. Each acceptance job has a clean checkout and can execute Python standard-library scripts without credentials or paid APIs.

## Reference implementation
experiments/energy_constraints/pilot.py is the baseline. The historical tests in tests/test_energy_constraints.py include both feasible and provably impossible small graphs. Preserve their oracle logic. The source paper is arXiv:2611.17309, but this implementation is NOT a reproduction of the original learned energy model.

## Architecture
A maker agent may edit on an isolated branch. The checker independently runs the unchanged repo's acceptance contract with no model or shell scripting from untrusted input. It runs bounded commands, stops on the first failed invariant, hashes the spec and contract, stores the result and refuses to label a failure successful. No automatic merge, deploy, or code modification.

## Test plan
Scenario: feasible synthetic graph
Given a bounded fixed graph and the unchanged exact oracle
When the energy descent and independent tests execute
Then an actually valid zero-conflict assignment must pass the oracle and the tests must pass.

Scenario: impossible synthetic graph
Given a triangle with only two colors and an exact feasibility check
When the energy method proposes an assignment
Then the independent oracle rejects impossible colorings and the run reports zero false accepted solutions.

Machine acceptance is engineering/contracts/energy-constraints.json. Passing this contract is narrower than a real quantum or food-domain verifier.
