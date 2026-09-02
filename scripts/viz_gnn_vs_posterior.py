#!/usr/bin/env python3
"""AED-style network plots: GNN vs ABC posterior vs exact posterior, in demand space.

One panel per method.  Each junction is drawn as two concentric discs, matching the AED
figure convention:
    outer ring  = posterior standard deviation at that node  (the irreducible uncertainty)
    inner disc  = |estimate - truth|                          (the actual error)
The GNN is a point estimate, so its ring is drawn hollow.

All three methods use the SAME single-leak prior (node v uniform, lambda ~ U[0, Delta]) and
the SAME observation, so the panels are directly comparable:
  * exact  -- enumeration over the n candidate nodes, lambda solved from the sensor
              constraint, weighted by the change-of-variables factor.  No sampling.
  * ABC    -- draws from the same prior, weighted by K_eps(residual).  Should reproduce
              "exact"; the gap between them is Monte-Carlo error.
  * GNN    -- the trained SMARTWINE model.  NOTE it was trained on *Dirichlet* scenarios,
              so a single leak is out-of-distribution for it; see the caveat printed at the end.

The per-node demand increase of the true scenario (d_true - d0) is printed under each node.
"""
from __future__ import annotations
import argparse, importlib.util, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _load_ovp():
    spec = importlib.util.spec_from_file_location(
        "ovp", os.path.join(ROOT, "scripts", "observation_vs_prediction.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def coordinates(inp_path):
    xy, on = {}, False
    for ln in open(inp_path):
        s = ln.strip()
        if s.startswith("["):
            on = s.upper().startswith("[COORDINATES]"); continue
        if not on or not s or s.startswith(";"): continue
        f = s.split()
        if len(f) >= 3:
            xy[f[0]] = (float(f[1]), float(f[2]))
    return xy


def draw_prior(rng, d0, delta, prior):
    """One draw from the chosen prior."""
    n = len(d0)
    if prior == "dirichlet":
        # d = d0 + f*Delta*alpha, matching the SMARTWINE data generator (the observed total
        # demand varies, so f is a genuine random deviation factor, not a constant).
        f = float(rng.uniform(0.0, 1.0))
        return d0 + f * delta * rng.dirichlet(np.ones(n))
    v = int(rng.integers(n)); lam = float(rng.uniform(0, delta))
    d = d0.copy(); d[v] += lam
    return d


def abc_posterior(heads, d0, sc, obs, delta, eps, n_draws, seed=7, prior="leak"):
    """Monte-Carlo posterior under the chosen prior."""
    rng = np.random.default_rng(seed)
    n = len(d0)
    D = np.empty((n_draws, n)); res = np.empty(n_draws)
    for i in range(n_draws):
        d = draw_prior(rng, d0, delta, prior)
        D[i] = d
        res[i] = float(np.linalg.norm(heads(d)[sc] - obs))
    w = np.exp(-0.5 * (res / eps) ** 2)
    if w.sum() <= 0:
        return None, None, 0.0
    w = w / w.sum()
    mean = w @ D
    std = np.sqrt(np.maximum(w @ (D - mean) ** 2, 0.0))
    ess = 1.0 / float(np.sum(w ** 2))
    return mean, std, ess


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wdn", default="Alperovits")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=0.05, help="sensor kernel width for ABC (m)")
    ap.add_argument("--prior", choices=["leak", "dirichlet"], default="leak",
        help="leak: v~Unif(J), lambda~U[0,Delta], d = d0 + lambda*e_v (exact enumeration "
             "applies).  dirichlet: d = d0 + f*Delta*alpha, f~U[0,1], alpha~Dir(1..1) -- the "
             "law the SMARTWINE GNNs were actually TRAINED on, so the in-distribution test. "
             "No exact panel there: the support is a continuum, not a finite candidate set.")
    ap.add_argument("--abc-draws", type=int, default=4000)
    ap.add_argument("--leak-node", default=None, help="force the true leak node")
    ap.add_argument("--no-gnn", action="store_true",
        help="skip the GNN panel (needed where demand reconstruction is broken, "
             "i.e. every network except Alperovits)")
    ap.add_argument("--trials", type=int, default=0,
        help="average over N random observations, AED-style: inner disc = MAE over trials, "
             "outer ring = std of the error over trials. 0 = single-observation mode, where "
             "the ring is instead the POSTERIOR std for that one observation.")
    ap.add_argument("--abc-prior", choices=["leak", "dirichlet"], default=None,
        help="prior used by ABC; defaults to --prior.  Setting it differently (e.g. --prior "
             "leak --abc-prior dirichlet) puts all three methods on the SAME observations "
             "while ABC assumes the demand model the GNN was trained on.")
    ap.add_argument("--score", choices=["vs-truth", "vs-posterior"], default="vs-truth",
        help="vs-truth: |estimate - d_true| (needs ground truth, simulation only).  "
             "vs-posterior: score the GNN against the posterior instead -- per node "
             "|d_GNN - E[d|z]| / sigma_post, i.e. how many posterior sigmas the prediction "
             "lies outside what the sensors permit.  Needs NO ground truth, so it is "
             "computable on real data, and by the identity E||d_GNN-d||^2 = "
             "||d_GNN-E[d|z]||^2 + Var[d|z] it isolates the predictor's own excess.")
    ap.add_argument("--sigma-floor", type=float, default=0.01,
        help="floor on the posterior std used by --score vs-posterior, as a fraction of the "
             "mean base demand.  Needed because a near-point-mass posterior has sigma -> 0, "
             "making |d_GNN - E[d|z]|/sigma diverge: 'how many sigma outside' is ill-posed "
             "when the posterior has no width.  Reported z is thus a LOWER bound on the "
             "violation; the absolute-excess panel is floor-free and always meaningful.")
    ap.add_argument("--exclude-scale", nargs="*", default=[],
        help="nodes excluded from the colour-scale range (still drawn, colour clipped, value "
             "annotated).  Use for a single node whose error dwarfs the rest.")
    ap.add_argument("--out", default="gnn_vs_posterior_aed.png")
    args = ap.parse_args()
    if args.abc_prior is None:
        args.abc_prior = args.prior

    ovp = _load_ovp()
    inp, net, J, d0, sensors, delta, G, stats, mult = ovp.load_case(args.wdn)
    heads = ovp.solver(inp, J, mult)
    sc = [J.index(s) for s in sensors]
    rng = np.random.default_rng(args.seed)

    def _gnn(heads, d_true):
        if args.no_gnn:
            return None
        import torch
        ds = torch.load(f"{ovp.SM}/{args.wdn}/data_generator/test_dataset.pt", weights_only=False)
        X = ds[0].x.numpy().copy()
        minp, maxp = stats["min_p"], stats["max_p"]
        h_all = heads(d_true)
        X[:, 1] = float(np.mean([(h_all[J.index(sn)] - net.nodes[sn].elevation_m - minp)
                                 / (maxp - minp) for sn in sensors]))
        return ovp.gnn_predict(args.wdn, net, J, G, stats, X)

    def one_observation(v_true, lam_true):
        """Errors + posterior stds for a single synthetic observation."""
        d_true = d0.copy(); d_true[v_true] += lam_true
        obs = heads(d_true)[sc]
        if args.prior == "dirichlet":
            abc_mean, abc_std, ess = abc_posterior(heads, d0, sc, obs, delta, args.eps,
                                                   args.abc_draws, prior=args.abc_prior)
            d_gnn = _gnn(heads, d_true)
            return dict(d_true=d_true, ex_mean=None, ex_std=None, keff=float("nan"),
                        abc_mean=abc_mean, abc_std=abc_std, ess=ess, d_gnn=d_gnn)
        cand, wts, resid = [], [], []
        for v in range(len(J)):
            lam, r, _ = ovp.consistent_lambda(heads, d0, v, sc, obs, delta)
            w, _ = ovp.cov_weight(heads, d0, v, sc, lam)
            dd = d0.copy(); dd[v] += lam
            cand.append(dd); resid.append(r)
            wts.append(w * float(np.exp(-0.5 * (r / args.eps) ** 2)))
        cand = np.array(cand); wts = np.array(wts)
        if wts.sum() <= 0:
            return None
        wts /= wts.sum()
        ex_mean = wts @ cand
        ex_std = np.sqrt(np.maximum(wts @ (cand - ex_mean) ** 2, 0.0))
        keff = float(1.0 / np.sum(wts ** 2))
        abc_mean, abc_std, ess = abc_posterior(heads, d0, sc, obs, delta, args.eps, args.abc_draws, prior=args.abc_prior)
        d_gnn = _gnn(heads, d_true)
        return dict(d_true=d_true, ex_mean=ex_mean, ex_std=ex_std, keff=keff,
                    abc_mean=abc_mean, abc_std=abc_std, ess=ess, d_gnn=d_gnn)

    if args.trials > 0:
        rng_t = np.random.default_rng(args.seed)
        acc = {k: [] for k in ("gnn", "abc", "exact", "gnn_z", "gnn_ex", "gnn_z_abc")}
        inc_acc = []
        for t in range(args.trials):
            if args.prior == "dirichlet":
                d_t = draw_prior(rng_t, d0, delta, "dirichlet")
                obs_t = heads(d_t)[sc]
                am, asd, ess_t = abc_posterior(heads, d0, sc, obs_t, delta, args.eps,
                                               args.abc_draws, prior=args.abc_prior)
                if am is None:
                    print(f"  trial {t}: weights underflowed, skipped"); continue
                r = dict(d_true=d_t, ex_mean=None, ex_std=None, abc_mean=am,
                         abc_std=asd, d_gnn=_gnn(heads, d_t))
                v, lam = -1, float(d_t.sum() - d0.sum())
            else:
                v = int(rng_t.integers(len(J))); lam = float(rng_t.uniform(0, delta))
                r = one_observation(v, lam)
            if r is None:
                print(f"  trial {t}: all weights underflowed, skipped"); continue
            inc_acc.append(r["d_true"] - d0)
            if r["ex_mean"] is not None:
                acc["exact"].append(np.abs(r["ex_mean"] - r["d_true"]))
            acc["abc"].append(np.abs(r["abc_mean"] - r["d_true"]))
            if r["d_gnn"] is not None:
                acc["gnn"].append(np.abs(r["d_gnn"] - r["d_true"]))
                # posterior-referenced scores: no ground truth involved
                sfloor = args.sigma_floor * float(np.mean(d0))
                if r.get("ex_mean") is not None and r.get("ex_std") is not None:
                    ex_sd = np.sqrt(r["ex_std"] ** 2 + sfloor ** 2)
                    acc["gnn_ex"].append(np.abs(r["d_gnn"] - r["ex_mean"]))
                    acc["gnn_z"].append(np.abs(r["d_gnn"] - r["ex_mean"]) / ex_sd)
                if r.get("abc_mean") is not None and r.get("abc_std") is not None:
                    ab_sd = np.sqrt(r["abc_std"] ** 2 + sfloor ** 2)
                    acc["gnn_z_abc"].append(np.abs(r["d_gnn"] - r["abc_mean"]) / ab_sd)
            lbl = f"leak {J[v]}" if v >= 0 else "dirichlet"
            print(f"  trial {t+1}/{args.trials}: {lbl} extra {lam:.3f}", flush=True)
        panels = []
        if args.score == "vs-posterior":
            if not acc["gnn_z"]:
                print("ERROR: --score vs-posterior needs the GNN", file=sys.stderr); return 2
            Z = np.array(acc["gnn_z"]); E = np.array(acc["gnn_ex"])
            panels.append((f"GNN vs EXACT posterior  (mean |d_GNN-E[d|z]|/sigma, {len(Z)} obs)",
                           Z.mean(0), Z.std(0), "posterior sigmas"))
            if acc["gnn_z_abc"]:
                Za = np.array(acc["gnn_z_abc"])
                panels.append((f"GNN vs ABC posterior  (mean |d_GNN-E[d|z]|/sigma, {len(Za)} obs)",
                               Za.mean(0), Za.std(0), "posterior sigmas"))
            panels.append((f"GNN excess |d_GNN - E[d|z]|  (absolute, {len(E)} obs)",
                           E.mean(0), E.std(0), "demand units (m3/s)"))
            for pnl in panels:
                name, m = pnl[0], pnl[1]
                print(f"  {name:<62} mean {m.mean():.4f}  max {m.max():.4f}")
            inc = np.array(inc_acc).mean(0)
            _plot(args, inp, net, J, d0, d0 + inc, sensors, panels,
                  subtitle=("scored against the POSTERIOR, not ground truth --- no true "
                            f"demand vector is used anywhere; averaged over {len(Z)} observations"))
            return 0
        if acc["gnn"]:
            A = np.array(acc["gnn"]); panels.append((f"GNN  (MAE over {len(A)} obs)", A.mean(0), A.std(0)))
        A = np.array(acc["abc"]); panels.append((f"ABC posterior mean  (MAE over {len(A)} obs, eps={args.eps})", A.mean(0), A.std(0)))
        if acc["exact"]:
            A = np.array(acc["exact"]); panels.append((f"Exact posterior mean  (MAE over {len(A)} obs)", A.mean(0), A.std(0)))
        for name, m, sd in panels:
            print(f"  {name:<52} mean {m.mean():.4f}  max {m.max():.4f}")
        inc = np.array(inc_acc).mean(0)
        _plot(args, inp, net, J, d0, d0 + inc, sensors, panels,
              subtitle=rf"averaged over {args.trials} random observations — "
                       rf"outer ring = std of the error across observations; "
                       rf"$\Delta$ = mean true demand increase at that node")
        return 0

    if args.leak_node is not None:
        if args.leak_node not in J:
            print(f"ERROR: unknown node {args.leak_node}; have {J}", file=sys.stderr); return 2
        v_true = J.index(args.leak_node)
    else:
        v_true = int(rng.integers(len(J)))
    lam_true = float(rng.uniform(0, delta))
    d_true = d0.copy(); d_true[v_true] += lam_true
    obs = heads(d_true)[sc]
    print(f"{args.wdn}: true leak {J[v_true]}, lambda {lam_true:.4f}, sensors {sensors}, Delta {delta}")

    # --- exact posterior (enumeration + change of variables)
    # Weight every candidate by  (change-of-variables) x K_eps(residual).  With one sensor the
    # constraint is exactly solvable for every node, all residuals are ~0 and only the Jacobian
    # discriminates.  With more sensors than unknowns the system is over-determined, most
    # candidates cannot fit exactly, and K_eps is what separates them -- the same eps ABC uses,
    # so the two panels answer the identical question.
    cand, wts, resid = [], [], []
    for v in range(len(J)):
        lam, r, _ = ovp.consistent_lambda(heads, d0, v, sc, obs, delta)
        w, _ = ovp.cov_weight(heads, d0, v, sc, lam)
        d = d0.copy(); d[v] += lam
        cand.append(d); resid.append(r)
        wts.append(w * float(np.exp(-0.5 * (r / args.eps) ** 2)))
    cand = np.array(cand); wts = np.array(wts); resid = np.array(resid)
    if wts.sum() <= 0:
        print("ERROR: all candidate weights underflowed; raise --eps", file=sys.stderr); return 2
    wts /= wts.sum()
    keff = float(1.0 / np.sum(wts ** 2))
    ex_mean = wts @ cand
    ex_std = np.sqrt(np.maximum(wts @ (cand - ex_mean) ** 2, 0.0))
    print(f"exact posterior: {len(cand)} candidates, effective {keff:.1f} "
          f"(residuals {resid.min():.2e}-{resid.max():.2e} m)")

    # --- ABC on the same prior
    abc_mean, abc_std, ess = abc_posterior(heads, d0, sc, obs, delta, args.eps, args.abc_draws, prior=args.abc_prior)
    print(f"ABC: {args.abc_draws} draws, eps={args.eps}, ESS {ess:.0f} ({ess/args.abc_draws:.3f} of N)")

    # --- GNN
    d_gnn = None
    if not args.no_gnn:
        import torch
        ds = torch.load(f"{ovp.SM}/{args.wdn}/data_generator/test_dataset.pt", weights_only=False)
        X = ds[0].x.numpy().copy()
        minp, maxp = stats["min_p"], stats["max_p"]
        h_all = heads(d_true)
        X[:, 1] = float(np.mean([(h_all[J.index(s)] - net.nodes[s].elevation_m - minp) / (maxp - minp)
                                 for s in sensors]))
        d_gnn = ovp.gnn_predict(args.wdn, net, J, G, stats, X)

    panels = []
    if d_gnn is not None:
        panels.append(("GNN prediction", np.abs(d_gnn - d_true), None))
    panels += [(f"ABC posterior (eps={args.eps}, ESS={ess:.0f})", np.abs(abc_mean - d_true), abc_std),
               (f"Exact posterior (enumeration, {keff:.1f} eff. candidates)",
                np.abs(ex_mean - d_true), ex_std)]
    for name, err, sd in panels:
        print(f"  {name:<42} mean |err| {err.mean():.4f}   max {err.max():.4f}")

    _plot(args, inp, net, J, d0, d_true, sensors, panels)
    return 0


def _plot(args, inp, net, J, d0, d_true, sensors, panels, subtitle=None):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    xy = coordinates(inp)
    res_ids = [str(r) for r in net.reservoirs]
    ex = set(str(x) for x in getattr(args, "exclude_scale", []) or [])
    keep = np.array([i for i, j in enumerate(J) if j not in ex], dtype=int)
    if keep.size == 0:
        keep = np.arange(len(J))
    # Panels may declare a scale group (4th tuple element).  Panels measured in DIFFERENT
    # units must not share a colour bar: mixing sigma-scores (order 10^3) with absolute
    # demands (order 10^-1) renders the latter uniformly green and unreadable.
    groups = [(p[3] if len(p) > 3 else "default") for p in panels]
    gmax = {}
    for p, g in zip(panels, groups):
        v = float(np.max(p[1][keep]))
        if p[2] is not None:
            v = max(v, float(np.max(p[2][keep])))
        gmax[g] = max(gmax.get(g, 0.0), v)
    cmap = plt.get_cmap("RdYlGn_r")
    norms = {g: plt.Normalize(0.0, v if v > 0 else 1.0) for g, v in gmax.items()}
    inc = d_true - d0

    # Scale the canvas and the markers with the network size, otherwise anything past ~10
    # junctions overlaps its own labels.
    n = len(J)
    side = 6.2 if n <= 10 else 6.2 + 0.16 * (n - 10)
    S_out = 1500 if n <= 10 else max(420, 1500 * 10.0 / n)
    S_in = S_out * 0.51
    S_hex = S_out * 2.0
    fs_id = 9 if n <= 10 else max(6.0, 9 * (10.0 / n) ** 0.35)
    fs_d = 7.5 if n <= 10 else max(5.0, 7.5 * (10.0 / n) ** 0.35)
    fig, axes = plt.subplots(1, len(panels), figsize=(side * len(panels), side * 1.05))
    for ax, pnl, g in zip(np.atleast_1d(axes), panels, groups):
        title, err, sd = pnl[0], pnl[1], pnl[2]
        norm = norms[g]
        for pid, p in net.pipes.items():
            a, b = p.start_node, p.end_node
            if a in xy and b in xy:
                ax.annotate("", xy=xy[b], xytext=xy[a],
                            arrowprops=dict(arrowstyle="->", color="0.35", lw=1.2))
        for r in res_ids:
            if r in xy:
                ax.scatter(*xy[r], s=S_out * 0.6, marker="s", c="grey", edgecolors="k", zorder=3)
                ax.annotate(r, xy[r], ha="center", va="center", zorder=4, fontsize=fs_id)
        for k, j in enumerate(J):
            if j not in xy: continue
            x, y = xy[j]
            # Sensors get a blue hexagon drawn BEHIND the node so their error and posterior
            # std stay visible -- a leak can sit at a measurement site, and that case is
            # exactly the one worth seeing.
            if j in sensors:
                ax.scatter(x, y, s=S_hex, marker="H", c="cornflowerblue",
                           edgecolors="k", lw=.8, zorder=2)
            outer = cmap(norm(sd[k])) if sd is not None else "white"
            ax.scatter(x, y, s=S_out, marker="o", c=[outer], edgecolors="k", lw=.8, zorder=3)
            ax.scatter(x, y, s=S_in, marker="o", c=[cmap(norm(err[k]))], edgecolors="k", lw=.5, zorder=4)
            if j in ex:      # off-scale: mark with a dashed ring and print the real value
                ax.scatter(x, y, s=S_out * 1.55, marker="o", facecolors="none",
                           edgecolors="k", lw=1.4, linestyle="--", zorder=5)
                ax.annotate(f"{err[k]:.3f}", (x, y), textcoords="offset points",
                            xytext=(0, 13 * (1.0 if n <= 10 else (10.0 / n) ** 0.35)),
                            ha="center", fontsize=fs_d, weight="bold", zorder=6)
            ax.annotate(j, (x, y), ha="center", va="center", zorder=5, fontsize=fs_id, weight="bold")
            dy = (-34 if j in sensors else -26) * (1.0 if n <= 10 else (10.0 / n) ** 0.35)
            ax.annotate(rf"$\Delta$={inc[k]:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, dy), ha="center", fontsize=fs_d, color="black", zorder=6)
        ax.set_title(title, fontsize=11)
        ax.set_axis_off(); ax.margins(.16)

    handles = [Line2D([], [], marker="s", ls="", mfc="grey", mec="k", ms=12, label="Reservoir"),
               Line2D([], [], marker="H", ls="", mfc="cornflowerblue", mec="k", ms=17,
                      label="Measurement site (hexagon behind node)"),
               Line2D([], [], marker="o", ls="", mfc="w", mec="k", ms=13,
                      label=("Junction: outer = std of error over observations, inner = MAE"
                             if args.trials > 0 else
                             "Junction: outer = posterior std, inner = |error|"))]
    if ex:
        handles.append(Line2D([], [], marker="o", ls="--", mfc="none", mec="k", ms=15,
                              label=f"excluded from colour scale: {', '.join(sorted(ex))}"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=True, fontsize=9)
    axl = np.atleast_1d(axes).tolist()
    seen = []
    for g in groups:
        if g in seen:
            continue
        seen.append(g)
        member_axes = [a for a, gg in zip(axl, groups) if gg == g]
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norms[g], cmap=cmap),
                          ax=member_axes, fraction=.03, pad=.02)
        cb.set_label(g if g != "default" else "absolute demand error / posterior std")
    fig.suptitle(subtitle if subtitle else
                 rf"{args.wdn}: ONE observation, single-leak prior — outer ring = posterior std "
                 rf"for this observation; $\Delta$ = that node's TRUE demand increase $d-d^0$",
                 y=.045, fontsize=10)
    fig.savefig(args.out, dpi=145, bbox_inches="tight")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
