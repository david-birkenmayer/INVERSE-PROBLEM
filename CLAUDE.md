# CLAUDE.md

Guidance for working in this repository.

## What this project is

A research codebase on the **inverse problem for water distribution networks (WDNs)**:
given pressure sensors at a subset of junctions, how well can demands be recovered, and
where should sensors be placed? It combines an optimization-based sensor-placement study,
a Dirichlet scenario-generation model, a new MCMC posterior sampler, and a GNN
predictor pipeline, all tied together by a PyQt6 GUI. The written thesis lives in
`docs/` (Typst); the chapters are the ground truth for the math.

### The steady-state model (see `docs/chapters/2_the-steady-state-model.typ`)

A WDN is a graph `G=(V,E)` with one reservoir. Unknowns per node/edge: demand `d_v`,
flow `q_e`, pressure head `h_v`. Coupled by **mass conservation** (`Bq = d`) and
**energy conservation** (`hᵤ − hᵥ = rₑ·qₑ·|qₑ|^(x−1)`), where `x=1.852` is
Hazen–Williams and `x=2` is Darcy–Weisbach. Given demands and resistances, the flow is
the unique minimizer of a strictly convex potential `Σ (rₑ/3)|qₑ|³` subject to `Bq=d`;
pressures are the Lagrange multipliers, unique once the reservoir head is fixed.

### Key quantities (see `docs/chapters/4_bounds.typ`)

Two scenarios are **measurement-equivalent at S** if they share pressures on the sensor
set `S` and the same total reservoir demand — i.e. sensors can't tell them apart. A
measurement `M = (h_S, d_reservoir)` names an equivalence class. Central metrics:

- `W_d(M)` — worst-case demand distance within a measurement class ("radius").
- `C_d(M)` — Chebyshev (minimax) demand distance; `W_h`, `C_h` are the pressure analogues.

Sensor placement = choose `S` to minimize these. "Radius" in code/notes ≈ `W_d`.

### Dirichlet demand model (see `docs/chapters/3_dirichlet-model*.typ`)

Scenarios: `d_j = d_j^0 + α_j·Δ`, with shares `α ~ Dirichlet(1,…,1)` on the simplex and
a global extra mass `Δ = ξ·Δ_max`, `ξ ~ U([a,b])`. Extra demand is **global**, spread
across all junctions — this is what makes the scenario law spread-out and non-degenerate.

## Layout

| Path | Role |
|---|---|
| `step1_io.py` | Load `.inp` (via WNTR) → `NetworkData`; pipe resistances (HW / DW). Foundation for everything. |
| `step2_estimation.py` | Simulate demand scenarios (random, perturbed, **Dirichlet**), base scenario, capacity bounds. |
| `step3_solver.py` | Core optimization solvers (Pyomo/IPOPT + SciPy): demand bounds, single-node, pipe bounds, feasibility, max demand distance. |
| `step3_solver_hexaly.py`, `step3_solver_xd_hexaly.py` | Hexaly-backed variants of the demand-distance / center solvers. |
| `inverse.py` | **CLI entry** for the sensor-placement study. Module-level constants (`WDN`, `MODE`, `METHOD`, `NORM`, …) configured via `--config <json>`. Orchestrates step2/step3. |
| `solver.py` | Standalone solver + plotting driver (cycles, planar faces, demand-distance plots). |
| `mh_posteriori-scenario-gen.py` | **MCMC posterior sampler** (the newest work). See below. |
| `image.py` | Plotting from saved solver output JSON. |
| `gui/app.py` | PyQt6 GUI (~5300 lines), the main user-facing entry. Tabs: Solver, posteriori, Local Search, GNN. |
| `gui/cache.py` | Content-addressed artifact caching: `compute_hash` (SHA-256 of canonical JSON), index load/save. |
| `gui/state.py` | `SolverParams` dataclass. |
| `docs/` | Typst thesis. `main.typ` includes `chapters/*.typ`. Chapter 7 = MCMC. |
| `scenario/<WDN>/` | Saved posterior scenarios: `<name>.json` (scenario) + `<name>_mh_result.json` (MCMC output), `cache_index.json`. |
| `data/`, `.gui_cache/` | Cached solver runs / GNN artifacts (see `PLAN.md` for the GNN cache schema). |
| `wdn/*.inp` | EPANET networks: Alperovits, Kadu, Hanoi, Anytown, Modena, BAK, Baghmalek, ZhiJiang. |
| `old/` | Superseded scripts + the GNN data-generator notebook referenced by `PLAN.md`. |
| `PLAN.md` | Integration plan for the GNN pipeline ↔ inverse solver (mostly done). |

## The MCMC posterior sampler (`mh_posteriori-scenario-gen.py`)

Full derivation in `docs/chapters/7_mcmc-posteriori-scenario-generation.typ`. Samples the
posterior `π(h,d | M)` over scenarios consistent with a measurement `M`. **Three methods**
exist (`cfg.method`, see below): M1 pressure-coordinate (original), M2 demand-space soft-sensor,
M5 demand-space exact. Each pairs with a `cfg.proposal` (rwm / ensemble). The bullets below
describe the **original M1**; M2/M5 are in the "Sampling method" paragraph.

- **State** = reduced pressure vector `z = h_F` over free (unobserved, non-eliminated)
  nodes. Sensor pressures are fixed; one **elimination node** `v` is recovered from the
  total-demand constraint `g(h_v; z) = Σ_j d_j − D = 0` by Newton iteration.
- **Target** (log form): Dirichlet density on the reconstructed demands + Gram-determinant
  change-of-variables factor `½·log det(J_red^T J_red)`, times a hard feasibility indicator.
- `J_red = J_F − J_v·(∂_z g / ∂_{h_v} g)` is a **rank-one** modification of the free-column
  Jacobian; there's an optional matrix-determinant-lemma fast path.
- Random-walk Metropolis with adaptive proposal scale (target accept ≈ 0.234).
- Public entry: `sample_posterior_scenarios(...) → MHSamplingResult`
  (`samples_d`, `samples_h`, `samples_z`, `log_targets`, acceptance/ESS diagnostics).

**Sampling method (`cfg.method`).** `"pressure"` (M1, default-legacy) = reduced pressure
coordinates, demands reconstructed, hard sensor + total-demand elimination + Gram-Jacobian.
`"demand"` (M2) = sample demand shares (softmax of a free vector on the simplex), pressures
forward-solved (`_forward_solve`, the well-conditioned convex direction), Dirichlet prior
evaluated natively (`Σ a_j log α_j`, incl. the reparam Jacobian), sensors imposed *softly*
by a Gaussian likelihood of width `cfg.sensor_noise_eps`. Total demand + floor are exact by
construction; `ε→0` reproduces the ABC oracle. M2 is well-conditioned on low-flow networks:
on Kadu, M1 is frozen (acc 0, R-hat 1e9) while **M2 mixes** (acc 0.24, R-hat 1.17); both
match the oracle to 0.08σ on well-conditioned Alperovits. Both `method`s work with either
`proposal`; the proposal loops are method-agnostic via `_eval`/`_StateEval.x`/`.warm`. The
GUI posteriori tab has a "Sampling Method" selector (above Scenario Choice) + an ε control.
M2 validated on Kadu: against a *same-hydraulics* ABC oracle (`abc_reference.py --hydraulics
internal`, which forward-solves with the sampler's own convex Newton instead of EPANET) M2
matches to **0.07σ mean / 0.20σ max** across all 24 junctions. The larger gap vs the *EPANET*
oracle (2.6σ) was purely a solver-consistency artifact — EPANET's 1e-3 flow accuracy vs the
internal 1e-10, amplified by Kadu's demand cancellation — not an M2 defect. (M2 R-hat on
Kadu dim-23 was ~1.45 at 8k samples; the posterior mean already matches, more samples/walkers
tighten R-hat.)

**Exact demand method (`cfg.method="demand_exact"`, M5, ε=0).** Samples the *free* demands
(dim `n−|S|−1`, or `n−|S|` when a reservoir-adjacent junction is a sensor so total demand is
implied); the sensor and slack demands are recovered by a **mixed-boundary hydraulic solve**
(`_mixed_bc_solve`) that imposes sensor pressures *exactly* and closes total demand. Target =
Dirichlet on the reconstructed shares + a Gram change-of-variables factor computed
analytically via the implicit-function theorem (`d_dep/d_theta = C·M⁻¹`, reusing the demand
Jacobian). Unlike the soft method, **sensors are eliminated, so more sensors reduce the free
dimension.** Validated on Alperovits to **0.01σ** vs the ABC oracle (most accurate of the
three methods). Caveat: on genuinely thin scenarios (low-flow Kadu, small per-node extra, 4
tight sensors) it mixes as poorly as M2 (acc ~0.07, R-hat ~3.5) — because there the
difficulty is the small *feasible* region (demands near their floor), not ε-softness, so no
local sampler escapes it. GUI exposes M5 as a method preset (ε control hidden, ε=0).

**Demand model / floor (`cfg.demand_reference`, applies to M2 & M5).** `"base"` (default) =
demands are `d_base + Dirichlet split of the extra D−D0`, so `d ≥ d_base` (the thesis model).
`"zero"` = demands are a Dirichlet split of the *total* D, so only `d ≥ 0` (physical). The
`d ≥ d_base` floor is not physically well-motivated (demands can drop below base), *but*
switching to `"zero"` **makes mixing worse**, not better (Kadu 4-sensor: R-hat 1.6→2.6 at
ε=0.2, 3.4→6.9 at ε=0.05). Reason: the floor also acts as an *informative prior* that
concentrates demands near base (near the truth); dropping it diffuses the prior over a ~2×
larger simplex while the sensor-consistent sliver stays equally thin, so the sliver is a
smaller fraction of the explored space. The principled middle path (not yet implemented) is a
**base-centred soft Dirichlet** — `α ~ Dirichlet(κ·base/base_total)`, `d = D·α`, support
`d ≥ 0` — which keeps the informative centre (aids mixing, precise estimates) without the
arbitrary hard wall; `κ` tunes trust in the base.

**Identifiability limit (the real Kadu story, not a sampler bug).** On genuinely
tightly-constrained questions — many sensors + small `ε` + small per-node extra demand on a
low-flow network — *every* local sampler (M1/M2/M5, rwm/ensemble) mixes badly, because the
measurement-consistent demand region is genuinely small: the demands are only weakly
identified. This is physics, not a bug (both M2 and M5 agree it's hard). The practical lever
for M2 is **`sensor_noise_eps`**: 5 cm (0.05) is unrealistically tight for a low-flow network;
a realistic **0.2–0.5 m** widens the posterior appropriately and makes it samplable (Kadu
4-sensor: R-hat 3.4→1.3 as ε 0.05→0.5). For the genuinely-thin regime the honest options are
larger ε, a base-centred prior, tempering/SMC, or reporting the wide posterior as the
identifiability result.

**Proposal mechanism (`cfg.proposal`).** `"rwm"` = isotropic random-walk Metropolis
(default, legacy); `"ensemble"` = affine-invariant ensemble / stretch-move sampler
(Goodman-Weare), gradient-free and reusing the same target+feasibility. On the thin
hard-floor feasible sliver, random-walk fails (acceptance ~5%, R-hat ~20, samples nowhere
near the truth) while the ensemble mixes (acceptance ~37%, R-hat ~1.03, min-ESS ~880) and
**matches the independent ABC oracle to 0.08σ** — validating the whole pressure-coordinate
+ Newton + Gram-Jacobian construction end to end. Ensemble notes: init dispersion must be
small (`ensemble_init_dispersion≈0.02`) so walkers start *inside* the feasible region;
walkers auto-set to `max(2·dim+2, 8)`. Prefer `"ensemble"` for real runs.

**Low-flow conditioning limit (why the *pressure* method M1 fails on Kadu).** M1 works in
pressure coordinates and *reconstructs* demands from head differences (`q ∝ |Δh|^(1/n)`). On low-flow networks
whose nodal demands are small net differences of larger pipe flows (Kadu: demands ~0.012,
some pipe `|Δh|` ~4e-4), this reconstruction is ill-conditioned: the sampler's demands
differ from EPANET's by ~the demand magnitude (mean 0.009 vs demand 0.012), so the *true*
scenario is deemed infeasible (18/24 nodes below floor) and every proposal is rejected —
acceptance 0 regardless of proposal/stretch tuning. This is a formulation/conditioning
limit, not a mixing one; the proper fix is sampling in **demand coordinates** (demands
primary, pressures forward-solved). Alperovits (demands ~0.19) is well-conditioned and works.

**Validation harness.** `scripts/abc_reference.py` (general; `--wdn/--sensors/--delta/...`)
computes the true posterior
by ABC (draw Dirichlet demands → EPANET forward-sim → weight by sensor match), caches it
(`scripts/abc_cache_*.npz`, independent of the sampler so reused across variants), and
scores each MCMC proposal by per-junction mean-gap in oracle-σ. This is the yardstick for
any future sampler change; re-run after touching the sampler.

**Convergence diagnostics.** `MHSamplingResult` reports mixing/efficiency, not just raw
sample count: `min_ess`/`median_ess` (effective sample size, Geweke IPS estimator),
`elapsed_seconds` + `min_ess_per_sec` (the right cross-config efficiency metric), a split-R-hat
(`rhat_per_dimension`, `max_rhat`; for the ensemble each walker is a chain), and
`max_mean_disagreement` (spread of the per-chain *demand* means as a fraction of the pooled
posterior σ). Read R-hat as: ≈1.00 mixed, >~1.01 not converged, huge = chains stuck at their
starts (classic low-acceptance failure). **`max_mean_disagreement` is the "can I trust the
estimate anyway" gauge:** the posterior *mean* (the demand estimate — Bayes-optimal under L²)
converges faster than the full distribution, so at moderate R-hat (≲1.5) a small mean-disagree
(<~0.2σ) means the mean is reliable even though tails/quantiles aren't; a large one (stuck
walker) means even the mean is suspect. The GUI status line shows `min_ess/s`, R-hat (⚠ if
>1.01), and `mean-agree` (⚠ if >0.25σ). GUI controls are proposal/method-aware: **"Walkers"**
for the ensemble vs **"Chains (R-hat)"** for random-walk (only one shown), ε only for M2,
Gram/penalty only for M1; a **progress bar** reflects real sampler progress via a
`progress_callback` threaded through `sample()`.

**Reservoir heads are fixed, not sampled.** Reservoir heads are known boundary conditions,
so `__init__` adds them to `fixed_heads` (from `predictor_heads`, else the `.inp` base head).
Leaving a reservoir head as a free `z` coordinate is a spurious, weakly-identified dimension
that destroys mixing — on Alperovits it took R-hat from ~1.1 (fixed) to ~10¹¹ (free). If you
ever see acceptance collapse or astronomical R-hat, check that every reservoir is in
`fixed_heads`.

**Elimination-node constraint (important, non-obvious):** `∂g/∂h_v = Σ(J[:, v_col])` is a
*structural zero* unless `v` shares a pipe with the reservoir. For any junction-to-junction
pipe the two Jacobian contributions cancel in the column sum; only reservoir-incident pipes
survive. So the elimination node **must be reservoir-adjacent**, otherwise Newton divides by
zero and acceptance is 0. This is math, not a bug — the total demand equals the reservoir
outflow, which only the reservoir head and its direct neighbors control. The GUI therefore
auto-selects the reservoir-adjacent junction (highest degree if several) and does not let
the user choose; the sampler's `_choose_elimination_node` uses the same rule.

## Running things

Use the project venv — Python 3.11 with WNTR/PyQt6/NumPy/SciPy installed:

```bash
/home/birkenma/venv_3-11/bin/python3 <script>
```

(The `.vscode/settings.json` interpreter path is stale — ignore it, use the venv above.)

- **GUI:** `python3 gui/app.py` (main way to drive everything).
- **CLI sensor-placement study:** `python3 inverse.py --config <config.json>`. The config
  JSON overrides the module-level `WDN`/`MODE`/… constants at the top of `inverse.py`.
- The GUI runs `inverse.py` as a **subprocess** (streams `PROGRESS_TOTAL:`/progress lines)
  and loads `mh_posteriori-scenario-gen.py` via **`runpy.run_path`** (note the hyphen in
  the filename — it's not importable as a module).
- Some solvers need **Hexaly** (license at `~/opt/Hexaly_14_5/license.dat`) or **IPOPT**
  via Pyomo. Cached runs replay without either; check `data/<WDN>/…/index.json`.

## Conventions

- Node/pipe IDs are **strings** throughout (`"3"`, not `3`); junction ordering comes from
  `NetworkData.junctions` insertion order.
- Caching is content-addressed: hash the canonical-JSON of inputs (`gui/cache.py`), store
  under a per-WDN `index.json`. Reuse before recomputing.
- Demands in m³/s (SI, as WNTR returns). Heads in meters.
- The GUI file is large; classes are top-level (`grep -nE "^class " gui/app.py`), main
  window logic under `MainWindow`, network canvas under `NetworkPlot`.
- Verify Python edits with `python3 -c "import ast; ast.parse(open('gui/app.py').read())"`.
- Match existing style: tabs for indentation in `gui/app.py` and `mh_posteriori-*.py`.

## Git

Work happens on the `overhaul` branch; `main` is the default for PRs. `scenario/*.json`,
`data/`, `.gui_cache/`, and `temp.*` are working artifacts that show up dirty — don't
commit them unless asked.
