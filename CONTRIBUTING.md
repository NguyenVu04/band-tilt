# Contributing

Start with [PROJECT.md](PROJECT.md) for the problem formulation and
[docs/adr/](docs/adr/) for why it is formulated that way.

## The one thing to understand first

**Almost every function in `src/` raises `NotImplementedError` with its own dotted
path. That is the intended state, not a backlog of bugs.**

The docstrings, configs, tests and decision records were written before any
implementation so that the shape of the problem is settled first. A `raise` is a
contract waiting to be filled, and the numbered `# TODO(n)` comments inside it are
the intended implementation order.

`src/data/split.py` is the reference for the stub shape: a module docstring with a
rationale section, Google-style function docstrings with
`Args`/`Returns`/`Raises`/`Notes`/`Example`, numbered TODOs, then the raise.

## Setup

```bash
task setup     # every extra, plus the pre-commit hooks
task sync      # or: base + dev only, no Sionna-RT or optimizers
```

## The loop

```bash
task format    # ruff --fix and ruff format
task check     # lint + tests
```

`task check` must be green before a pull request. Expect **69 skipped tests** —
each skip reason names the module that unblocks it. A collection *error* is a real
failure; a skip is not.

## Rules that are easy to break

- **Thresholds are config, never literals.** `-120`, `-90` and `6` come from
  `configs/kpi.yaml`. A hard-coded value is a bug even when it matches the config,
  because the surrogate acceptance report and the MARL reward read the same
  numbers and would drift.
- **Single sources of truth.** The KPIs live in `src/kpi/`, the angle conversion
  in `src/radio/geometry.py`, the search space in `src/optim/space.py`, the
  splitter in `src/data/split.py`, the cell-band table in
  `src/radio/cell_band.py`. A second implementation of any of them does not
  raise — it silently diverges and invalidates a result.
- **Expected test values are computed by hand.** A test that derives its
  expectation from the code under test asserts only that the implementation agrees
  with itself. The KPI values documented on the `rsrp_grid` fixture were worked
  out by hand and are load-bearing.
- **No test may reach the network, the real dataset, or Sionna-RT.** A suite that
  only runs after `dvc pull` stops being run.
- **Reported KPIs come from Sionna-RT, never the surrogate.** Where a prediction
  appears in a report, label it and show the ground truth beside it.
- **Nothing in `src/data/` may learn from the data.** Cleaning applies externally
  sourced bounds only. Anything fitted belongs in `src/surrogate/features.py`, on
  the training partition.
- **`data/` and `models/` are DVC-tracked.** Never commit their contents.

## Notebooks

Cell 3 — the environment bootstrap — is identical across all eight notebooks.
Change all eight or none. `nbstripout` runs on commit, so outputs are never
committed.

## When you need an ADR

Before implementing, not after, for any change to:

- the KPI definitions, thresholds, or priority order;
- the decision variable, search space, or tilt bounds;
- the coordinate or angle conventions;
- the split scheme;
- what a reported result may be computed from;
- a simulator, optimization framework, or third-party service.

Hyperparameters are not ADRs — `configs/` and MLflow record those. See
[docs/adr/README.md](docs/adr/README.md) for the process.

## Pull requests

- Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
- Short-lived branches off `main`; squash merge, linear history.
- Say what you ran and what it printed. If something was skipped or left out, say
  so and why.
- Update `CHANGELOG.md` under `[Unreleased]`.
