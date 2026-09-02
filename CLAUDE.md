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
| `wdn/*.inp` | EPANET networks: Alperovits, Kadu, Hanoi, Anytown, Modena, BAK, Baghmalek, ZhiJiang, **L-TOWN_C** (BattLeDIM Area C — own section below). |
| `docs/2020 BATTLEDIM Introduction.pdf` | BattLeDIM 2020 challenge deck: sensors, leak model/sizing, nominal-vs-real model. |
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

**Gaussian-prior methods (`cfg.method="gaussian"` = M3, `="gaussian_map"` = MAP).** Prior
`N(d_base, diag(σ²))`, `σ_j = cfg.prior_sigma·max(d_base_j, mean demand)`, on the **raw
demands** (no simplex/softmax); total demand imposed hard by a linear slack (`_gauss_full_demand`
closes `Σd=D` at the largest-base-demand junction); pressures forward-solved; sensors via the
same Gaussian `ε` likelihood; `d ≥ 0` a soft (rarely-binding) constraint. **M3** samples the
full non-linear posterior by MCMC (reuses the ensemble; mixes well — smooth log-concave prior,
no hard floor). **MAP** (`_run_map`) instead finds the posterior mode by Gauss-Newton
(`scipy.optimize.least_squares` on `[(h_S−M_S)/ε ; (d−μ)/σ]`) and returns a **Laplace-Gaussian**
sample cloud `N(d_MAP, (JᵀJ)⁻¹)` so it plugs into the same GUI/result path. MAP reproduces the
standard formal-Bayesian WDN baseline; M3 is the non-linearised version — comparing them shows
where non-Gaussianity matters. Validated on Alperovits vs a Gaussian-prior ABC oracle
(`abc_reference.py --prior gaussian`): M3 → 0.07σ, MAP → 0.04σ; M3 and MAP agree to ~0.01. GUI
exposes both as presets with a "Prior σ" control; MAP hides the walkers/chains knobs (it's an
optimizer, not a sampler).

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

## L-TOWN_C / BattLeDIM (current focus, added 2026-08-31)

`wdn/L-TOWN_C.inp` is Area C of the **BattLeDIM 2020** benchmark (intro deck:
`docs/2020 BATTLEDIM Introduction.pdf`). It has no companion `wdn/L-TOWN_C.json` yet.

**Structure — fits the model class exactly.** 92 junctions, 109 pipes (→ 17 independent
loops), **one** reservoir `1` at head 102 m, no tanks/pumps/valves, Hazen–Williams, single
connected component. Junction ids come in two blocks: `n1`–`n45` and `n343`–`n389`. Only
**one** reservoir-adjacent junction, `n343` — so it is the forced M1 elimination node (see
the elimination-node rule above), and putting a sensor there changes the total-demand
elimination.

**Extreme low-flow regime — this is the dominant fact.** Base-case forward solve:

| quantity | value |
|---|---|
| total base demand `D⁰` (= avg inflow) | 5.15e-3 m³/s = 18.5 m³/h |
| head spread across the *whole* network | **0.088 m** |
| median pipe head loss | 2.0e-5 m |
| pipes with \|Δh\| < 1e-4 m | 95 / 109 |
| peak demand pattern multiplier | 1.51 (`P-Residential`) |

Head loss ~ `q^1.852`, so even at pattern peak the spread is ≈0.19 m; a ~8.4× demand
multiplier would be needed for a 5 m spread. **Consequence:** at any realistic sensor noise
the likelihood `K_ε` is near-flat over the prior support, so `P(s|z) ≈ P(s)` — demands are
weakly identified and the reported a-posteriori error will be close to the *prior* width.
That holds for *diffuse* priors (Dirichlet/Gaussian over all 92 demands). It also flips the
difficulty: unlike Kadu's thin sliver, a near-flat likelihood is a wide, well-conditioned
posterior that **mixes easily** even at dim 92. M1 (pressure space) is a non-starter here
(`Δh ~ 1e-5` reconstruction); use M2/M3. **But note:** for the concentrated single-leak prior
at realistic BattLeDIM leak sizes the signal is *far* above the noise (see the sizing table
below) — that problem is well identified. Low flow makes the diffuse-demand question hard,
not the leak-localization question.

**Hard resolution floor on ε.** The BattLeDIM deck states sensor readings are *rounded to 2
decimal places* → **0.01 m is a floor on ε regardless of sensor quality.**

**Sizing `extra_demand` (leak magnitude) — percentages are of the SYSTEM inflow.** The deck
sizes leaks as a fraction of average inflow (background 1–5%, medium burst 5–10%, large burst
>10%) and states **average system inflow ≈ 180 m³/h** for all of L-Town. Area C's inflow is
only 18.5 m³/h, i.e. **~10% of the system** — so the brackets must be applied to 180, *not*
to 18.5. In absolute terms BattLeDIM leaks are ≈1.8 m³/h (smallest background) up to >18 m³/h
(medium/large burst), which for Area C is 10%–150% of its entire demand. Sweep of a single-node
leak λ over all 92 junctions, `max |Δh|` over the network (best-case oracle sensor):

| BattLeDIM class | λ (m³/h) | λ (m³/s) | % of Area-C inflow | median max\|Δh\| | invisible at 0.01 m |
|---|---|---|---|---|---|
| background 1% | 1.8 | 5.0e-4 | 9.7% | 0.017 m | 6 / 92 |
| background 5% | 9.0 | 2.5e-3 | 48% | 0.107 m | 2 / 92 |
| medium 10% | 18.0 | 5.0e-3 | 97% | 0.281 m | 0 / 92 |
| large 15% | 27.0 | 7.5e-3 | 146% | 0.514 m | 0 / 92 |

→ **Use `extra_demand ≈ 5.0e-3 m³/s` (18 m³/h)** so `λ ~ U[0, Δ]` spans the real BattLeDIM
range from smallest background leak to medium burst. (An earlier note here recommended
5.15e-4 m³/s by mis-reading the percentages as relative to Area-C inflow; that value is in
fact only the *smallest* BattLeDIM leak, not a representative one.)

**The single-leak prior collapses to exact enumeration (no MCMC needed).** Current prior of
interest: pick `v ∈ J` uniformly, `λ ~ U[0, Δ]`, set `d = d⁰ + λ·e_v`. Then
`1ᵀd = D⁰ + λ`, and since the model conditions on total demand **exactly** (`δ(D^z − D^s)`;
in BattLeDIM D really is measured — flow sensors at the DMA entrances), **the observation
determines λ**. What remains is a 92-term *discrete* posterior

    P(v | z) ∝ K_ε( h_Y(d⁰ + λ·e_v) − h_Y^z )

computable exactly with 92 forward solves (seconds) — no Metropolis, no Gram-Jacobian, no
mixing diagnostics. Worth building anyway: it is an **exact** ground truth, strictly better
than the sampled ABC oracle for validating M1/M2/M5. If D⁰ is not trusted (the nominal model
randomizes base demands ±10%), λ stays continuous and the posterior lives on
`92 × [0, Δ]` — still exact by 1-D quadrature per node (~5000 forward solves). MCMC only
becomes necessary once the prior is complicated to multiple simultaneous leaks or a
continuous demand field.

**Sensor set: RESOLVED — the Area C pressure sensors are `n1`, `n4`, `n31`.** Source:
`~/Seafile/Arbeitsordner/SMARTWINE_data/python/networks/l-town_edt.inp`. That file is Area C
**renumbered `1`–`92`** (not, as first assumed, a different network) and its `[JUNCTIONS]`
lines carry inline annotations: three nodes tagged `;AMR & PRESSURE SENSOR` (edt ids 1, 4, 31)
and 79 tagged plain `;AMR`. Mapping edt→L-TOWN_C by elevation is exact and unambiguous (all 92
elevations are unique, 92/92 matched): edt `1,4,31` → **`n1`, `n4`, `n31`**. This matches the
`n1/n4/n31` entries of the published BattLeDIM 33-sensor list, so two independent sources agree.
Note 82/92 junctions carry AMR meters (only `n5, n12, n14, n15, n37, n38, n348, n359, n363,
n380` do not), confirming the deck's statement that Area C is the metered district. When
searching for this again, grep the *renumbered* ids (`31`), not `n31`. A greedy
sensitivity-matrix selection run independently (`scripts/ltown_c_posterior.py`, no access to
this file) picked `n1` and `n4` among its first choices — a useful check on the placement code.

**Built: `scripts/ltown_c_enumerate.py` (exact single-leak posterior, no MCMC) and
`scripts/ltown_c_posterior.py` (M3 demand posterior, MCMC).** The enumerator implements the
collapse described above: 92 forward solves give the exact `P(v|z)`, scored by shortest-path
distance (BattLeDIM's own metric, `x_max = 50 m`). `--eps-sweep`, `--leak-size`, `--sensors`
and `--leak-node` make it a seconds-per-configuration study tool. Results with the official
sensors `n1/n4/n31`, medians over all 92 possible leak positions, `eps = 0.015 m`:

| λ (m³/h) | BattLeDIM class | MAP err | within 50 m | exact node | eff. candidates (of 92) |
|---|---|---|---|---|---|
| 1.8 | background 1% | 208 m | 18% | 13% | 86.7 |
| 5.0 | background ~3% | 94 m | 39% | 29% | 64.0 |
| 9.0 | background 5% | 28 m | 61% | 49% | 33.5 |
| 18.0 | medium burst 10% | 0 m | 84% | 72% | 13.4 |
| 27.0 | large burst 15% | 0 m | 91% | 84% | 6.7 |

(uniform guessing ≈ 391 m expected error; network diameter 1142 m.) **Area C with its 3 real
sensors cannot localize background leaks and can localize bursts.** Two non-obvious findings:
(1) **MAP error and the rank of the true node are exactly constant in ε** — ε only rescales a
monotone likelihood, so it changes posterior *spread* (entropy, expected error) but never the
ranking; only the sensitivity of the *spread* to ε is real. (2) At λ=1.8 the MAP error is
already 208 m at ε=0.001, so **the binding information limit is the 0.01 m reading
quantization, not ε** — no sensor-quality improvement helps, only bigger leaks or more sensors.
Worked example (λ=18, true leak `n370`): the true node ranks 2nd with P=0.082 against 0.0826
for `n369` 41.8 m away — a near-tie the *node* metric scores as a failure and BattLeDIM's 50 m
metric scores as a success. This is the concrete case for reporting distance, not node identity.
EPANET's shipped `Accuracy 0.01` is far too loose for Area C's ~1e-2 m head differences; both
scripts set `hydraulic.accuracy = 1e-8`.

**Sampler trap on low-flow networks: `ensemble_init_dispersion` is ADDITIVE in state units.**
M3's state is the raw demand vector (`initial_x = d0[...]`, ~5e-5 m³/s on Area C), and walkers
start at `initial_x + ensemble_init_dispersion * N(0,1)`. The 0.05 default is ~1000× the demand
scale here → acceptance 0.000, R-hat ~1e14. `scripts/ltown_c_posterior.py` sets it to
`0.2 × median prior σ`. The sampler default is untouched (Alperovits/Kadu depend on it) but is
scale-dependent and will bite on any low-flow network — worth fixing properly in the sampler.

**Nominal-vs-real caveat (from the deck).** The distributed `.inp` is the *nominal* model:
base demands and pipe parameters are randomized ±10% vs the "real" network used to generate
the data, industrial patterns are withheld, and pipes `p37`/`p251` are closed in the real
network. Any validation against BattLeDIM time series inherits that mismatch.

## GNN vs posterior on the SMARTWINE-trained models (added 2026-09-02)

**The real trained GNNs live in `/home/birkenma/Dokumente/SMARTWINE/old/data/<WDN>/`** —
`gnn_model/best_model.pt` plus `data_generator/{train,val,test}_dataset.pt`, for Alperovits,
Anytown, BAK, Baghmalek, Hanoi, Kadu (Modena/ZhiJiang have datasets but no model). Not the
Seafile `SMARTWINE_data/` tree, which has no pipeline. Sensor sets come from
`data_generator/graph_with_measurements.pickle` (`meas_*` nodes); `extra_demand` from
`data_generator/parameters.json`. Alperovits `['1']` Δ=1.2 · Hanoi `['2','12','25']` Δ=5.5 ·
Kadu `['3','19','24','14']` Δ=0.3 · BAK `['1','8','26','32']` · Anytown `['20','140']`.

**Running these models correctly — four traps, all of which silently degrade results.**
(1) The datasets store only `y` (normalised *pressure*, not head): recover head as
`y*(max_p-min_p) + min_p + elevation`, with `min_p/max_p/reservoir_head/reservoir_node` from
`dataset_stats.json`. (2) `x[:,0]` (base pressure) and `x[:,2]` (node type) are **constant
across samples** — take them verbatim from any stored sample rather than re-deriving them;
node type is `0` junction / `2` reservoir / **`3`** for the `meas_` node. Only `x[:,1]`, the
observed sensor reading broadcast to every node, depends on the observation. (3) Use the
dataset's own `edge_index` **and pass `edge_attr`** — rebuilding edge_index from the pickled
(multi)graph gives 30 directed edges where the dataset stores 15, and dropping `edge_attr`
changes predictions materially. (4) `Data.mask` marks the reservoir and `meas_` nodes, whose
outputs are garbage by design — exclude them, and take the reservoir head from `dataset_stats`.
With all four right, the Alperovits GCN predicts junction pressures to ~0.005 normalised.

**Demand reconstruction from heads is BROKEN on most networks — systematic, not conditioning.**
Reconstructing `q_e = sign(Δh)(|Δh|/r_e)^(1/n)` then mass balance, *from exact EPANET heads*:

| network | recon demand sum vs true | verdict |
|---|---|---|
| Alperovits | 1.120 vs 1.120 | **exact (0.0% error)** |
| Kadu | 0.147 vs 0.294 | exactly 2× low |
| Anytown | 0.189 vs 0.473 | exactly 2.5× low |
| BAK | 0.229 vs 1.146 | exactly 5× low |
| Hanoi | 0.554 vs 5.539 | exactly 10× low |

The ratios are **constant in the demand scale**, so this is a units/resistance bug in
`compute_pipe_resistances_hw` or the reconstruction, *not* the low-flow ill-conditioning
documented elsewhere (all five networks are LPS + H-W, 1 reservoir, no tanks/pumps/valves).
**Only Alperovits can currently be used for demand-space evaluation.** Fixing this unblocks
Hanoi/Kadu/BAK/Anytown and is the highest-value next task.

**`scripts/observation_vs_prediction.py`** — draws a random single-leak observation
(`v` uniform, `λ ~ U[0,Δ]`, noiseless sensors), finds every candidate node admitting a `λ_v`
that reproduces the reading, and plots those exactly-consistent scenarios against the truth and
the GNN. Note: conditioning on total demand as well (`--observe-total`) makes the noiseless
single-leak posterior **collapse to a point mass** (λ = D − D⁰ pins it), so total demand is
treated as unknown by default. Tolerance must be ≥ ~1e-4: EPANET reports heads to ~3e-5 m, and
a tighter tolerance rejects the *true* node.

Result on Alperovits over 10 random observations: **all 6 candidates are exactly consistent
with the single sensor** (each with its own λ) — the ambiguity is genuine, not numerical.
Posterior-mean error 0.27–0.97 (mean ≈0.55); **GNN error 0.77–3.73 (mean ≈1.5), i.e. 2–4×
worse than the posterior mean and usually worse than the *worst* consistent scenario**, and it
sometimes predicts negative demands. Caveat: the GNN was trained on *Dirichlet* scenarios and a
single leak is a simplex corner, so part of that gap is distribution shift, not GNN quality —
an in-distribution (Dirichlet) comparison is needed before blaming the model.

## Visualisations + the ABC pivot (added 2026-09-02)

**`scripts/viz_gnn_vs_posterior.py`** — AED-style network plots comparing GNN / ABC / exact
posterior in **demand** space. One panel per method; each junction is two concentric discs
(**outer = posterior std**, **inner = |estimate − truth|**) on `RdYlGn_r`, reservoir a grey
square, sensors a blue hexagon drawn *behind* the node so a leak at a sensor stays visible, and
each node annotated with its demand increase `Δ = d − d⁰`. Marker/figure sizes auto-scale with
junction count. Flags: `--wdn --leak-node --seed --eps --abc-draws --no-gnn --out`.
All three methods use the *same* single-leak prior and the same observation, so panels are
directly comparable. Figures produced: `gnn_vs_posterior_aed.png` (Alperovits, leak at n4),
`gnn_leak_at_sensor.png` (leak at the sensor), `kadu_abc_vs_exact_aed.png` (Kadu, no GNN).

| case | GNN | ABC | exact |
|---|---|---|---|
| Alperovits, leak n4 | 0.1664 | 0.0285 | 0.0297 |
| Alperovits, leak at sensor n1 | 0.1612 | 0.0307 | 0.0325 |
| Kadu, leak n21, ε=0.2 | (n/a) | 0.0108 | 0.0094 |

(mean per-node |error|.) **ABC and exact agree to a few %, mutually validating both**; the GNN
is ~5× worse. Use `--no-gnn` on every network except Alperovits — the demand-reconstruction
bug above makes GNN demands meaningless elsewhere.

**ABC beats MCMC decisively in the flat-likelihood regime.** L-TOWN_C, same M3 model:
ABC N=8000 in **147 s**, ESS **7968** (0.996 of N), 54.3 ESS/s — versus MCMC 4568 s, min-ESS
1010, 0.2 ESS/s, **max R-hat 2.96 (never converged)**. ~270× more ESS/s, and ABC reproduces the
exact linear-Gaussian answer while the MCMC's "0.77 max variance reduction" was a
non-convergence artifact. **Do not quote that MCMC number.**

**The criterion for which method to use:** `SNR = (prior-predictive sensor spread) / ε`.
Estimate it with a handful of forward solves before choosing.

| SNR | regime | tool |
|---|---|---|
| ≲ 1 | likelihood flatter than prior | **ABC — nearly free** (L-TOWN_C: SNR 0.17–0.34, ESS/N 0.996) |
| ≫ 1 | posterior concentrated | ABC collapses; use **enumeration** if the prior is structured |
| any, ε = 0 | noiseless | **only enumeration works** — ABC acceptance is exactly 0 (measure zero) |

Measured on Kadu (4 sensors, Δ=0.3, prior-predictive spread 0.426 m): ESS/N = 0.28 at ε=0.5,
0.049 at ε=0.2, 0.003 at ε=0.05, collapsed below. **MCMC fails in the same place** (R-hat 3.4 at
ε=0.05) — that wall is identifiability, not algorithm. Realistic ε for Kadu is **0.2–0.5 m**
(transducer 0.1–0.5 m at 0.1–0.5% of a ~100 m full scale, plus ±10% model mismatch contributing
median 0.11 m / p95 0.45 m). Note Kadu's *entire* head spread is 2.00 m, so ε=0.5 is ~25% of the
whole signal — ABC is efficient there precisely because the measurement is weak. Also: Kadu
sensor `'3'` has prior-predictive spread **0.000 m** — it is completely insensitive to demand,
so that placement is effectively 3 sensors, not 4.

**Change-of-variables correction (was missing, now in `observation_vs_prediction.py`).** When
λ is solved from the sensor constraint rather than observed, the induced posterior over the
discrete candidate `v` carries a factor `‖∂h_Y/∂λ‖⁻¹` — a candidate whose reading responds
slowly to λ explains a neighbourhood of observations more readily. Without it you silently get
a *uniform* posterior over the consistent candidates. On Alperovits it moves P(v|z) from a flat
0.167 to 0.14–0.20. Same Gram-determinant idea as M1/M5.

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
