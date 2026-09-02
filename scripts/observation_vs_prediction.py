#!/usr/bin/env python3
"""Random observation -> exactly-consistent leak scenarios -> compare against the GNN.

The easiest well-posed case, and the one this script implements:

  prior       one leak: node v uniform over junctions, magnitude lambda ~ U[0, extra_demand],
              demands d = d0 + lambda*e_v
  observation noiseless pressures at the sensor set Y (no quantisation, no sensor noise)

With the total demand ALSO observed, lambda = D - D0 is pinned and the noiseless posterior
collapses to a point mass -- the problem is solved exactly and there is nothing to compare.
So by default the total demand is treated as UNKNOWN (--observe-total to condition on it).
Then, for each candidate node v, there is generally exactly one lambda_v reproducing the
observed sensor pressures, so the posterior is a small set of *exactly consistent* scenarios.
Their spread is the irreducible ambiguity: no predictor can do better.

We then ask how the GNN prediction sits relative to that set.
"""
from __future__ import annotations
import argparse, json, os, pickle, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SM = "/home/birkenma/Dokumente/SMARTWINE/old/data"


def demand_pattern_multipliers(inp_path, J):
    """Constant demand-pattern multiplier per junction (1.0 where there is no pattern).

    IMPORTANT: several of these .inp files carry a single-value demand pattern -- Kadu 0.5,
    Hanoi 0.1, BAK 0.2, Anytown 0.4 (Alperovits is 1.0) -- so the demand EPANET actually
    simulates is  base_demand * multiplier.  `step1_io.load_inp_network` reports only
    base_demand and drops the pattern, which makes its demand vectors inconsistent with the
    hydraulics by exactly that factor on every network except Alperovits.  Corrected here
    rather than in step1_io, whose blast radius (sampler, solvers, caches) is large.
    """
    import wntr
    wn = wntr.network.WaterNetworkModel(inp_path)
    m = np.ones(len(J), dtype=float)
    for k, j in enumerate(J):
        ts = wn.get_node(j).demand_timeseries_list
        if not ts:
            continue
        pname = ts[0].pattern_name
        if pname:
            mult = np.array(wn.get_pattern(pname).multipliers, dtype=float)
            if mult.size:
                m[k] = float(mult[0]) if mult.size == 1 else float(mult.mean())
    return m


def load_case(wdn):
    from step1_io import load_inp_network
    inp = os.path.join(ROOT, "wdn", f"{wdn}.inp")
    net = load_inp_network(inp)
    J = list(net.junctions)
    mult = demand_pattern_multipliers(inp, J)
    d0 = np.array([net.junctions[j].base_demand for j in J], dtype=float) * mult
    G = pickle.load(open(f"{SM}/{wdn}/data_generator/graph_with_measurements.pickle", "rb"))
    sensors = [str(n).replace("meas_", "") for n in G.nodes if str(n).startswith("meas_")]
    par = json.load(open(f"{SM}/{wdn}/data_generator/parameters.json"))
    # EXTRA_DEMAND was specified by the data generator in *base* units, i.e. before EPANET
    # applied the demand pattern, so scale it the same way the base demands were scaled.
    delta = float(str(par["EXTRA_DEMAND"]).strip("'\"")) * float(np.mean(mult))
    stats = json.load(open(f"{SM}/{wdn}/data_generator/dataset_stats.json"))
    return inp, net, J, d0, sensors, delta, G, stats, mult


def solver(inp, J, mult=None):
    """Forward solve.  The model is parsed ONCE and its demand timeseries mutated in place --
    re-parsing per call costs ~11%.  NOTE EpanetSimulator writes temp.inp/temp.bin/temp.rpt
    into the CURRENT WORKING DIRECTORY, so two of these running in the same cwd corrupt each
    other; run concurrent jobs from separate directories."""
    import wntr
    if mult is None:
        mult = np.ones(len(J))
    wn = wntr.network.WaterNetworkModel(inp)
    wn.options.time.duration = 0
    wn.options.hydraulic.accuracy = 1e-10
    wn.options.hydraulic.trials = 500
    ts = [wn.get_node(j).demand_timeseries_list[0] for j in J]
    def heads(d):
        # divide by the pattern multiplier so the demand EPANET *simulates* is exactly d
        for t, v, m in zip(ts, d, mult):
            t.base_value = float(v) / float(m)
        h = wntr.sim.EpanetSimulator(wn).run_sim().node["head"].iloc[0]
        return np.array([float(h[j]) for j in J], dtype=float)
    return heads


def cov_weight(heads, d0, v, sc, lam, h=1e-4):
    """Change-of-variables factor for solving the sensor constraint exactly.

    Conditioning on h_Y(v, lambda) = obs pins lambda, and the induced density over the
    discrete candidate v is NOT uniform: it picks up |d h_Y / d lambda|^-1, because a
    candidate whose sensor reading responds *slowly* to lambda explains a neighbourhood of
    observations more readily than one that responds sharply.  This is the same Gram-
    determinant correction M1/M5 apply; omitting it silently returns a uniform posterior
    over the consistent candidates.
    """
    dp = d0.copy(); dp[v] += lam + h
    dm = d0.copy(); dm[v] += max(lam - h, 0.0)
    span = (lam + h) - max(lam - h, 0.0)
    J = (heads(dp)[sc] - heads(dm)[sc]) / span
    n = float(np.linalg.norm(J))
    return (1.0 / n if n > 0 else 0.0), n


def consistent_lambda(heads, d0, v, sc, obs, delta):
    """The lambda at node v reproducing the observed sensor heads; None if none in [0, delta].

    Sensor head is monotone decreasing in lambda (more demand -> more head loss), so a scalar
    bisection on the first sensor is exact and cheap; the residual over ALL sensors then says
    whether that candidate really explains the observation."""
    def f(lam):
        d = d0.copy(); d[v] += lam
        return heads(d)[sc] - obs
    lo, hi = 0.0, float(delta)
    flo, fhi = f(lo)[0], f(hi)[0]
    if flo * fhi > 0:
        best = lo if abs(flo) < abs(fhi) else hi
        return best, float(np.linalg.norm(f(best))), False

    # Secant: the sensor head is smooth and monotone in lambda, so this converges in ~6 solves
    # where bisection needs 62, for a lambda difference of ~4e-6 -- far below the ~3e-5 m
    # precision EPANET reports heads to.  Falls back to bisection if it fails to converge.
    x0, x1, f0, f1 = lo, hi, flo, fhi
    for _ in range(30):
        if abs(f1) < 1e-9 or f1 == f0:
            break
        x2 = min(max(x1 - f1 * (x1 - x0) / (f1 - f0), lo), hi)
        x0, f0, x1 = x1, f1, x2
        f1 = f(x1)[0]
    if abs(f1) < 1e-6:
        return x1, float(np.linalg.norm(f(x1))), True

    for _ in range(30):                       # fallback
        mid = 0.5 * (lo + hi)
        if f(mid)[0] * flo > 0: lo = mid
        else: hi = mid
    lam = 0.5 * (lo + hi)
    return lam, float(np.linalg.norm(f(lam))), True


def gnn_predict(wdn, net, J, G, stats, x_features):
    """Run the trained SMARTWINE GCN and reconstruct demands from its predicted pressures."""
    import torch
    from torch_geometric.nn import GCNConv
    from step1_io import compute_pipe_resistances_hw

    class GCN(torch.nn.Module):
        def __init__(self, dim_in, dim_h=64, dim_out=1):
            super().__init__()
            H = dim_h * 4
            self.batch_norm1, self.batch_norm2, self.batch_norm3 = (torch.nn.BatchNorm1d(H) for _ in range(3))
            self.gcn1, self.gcn2, self.gcn3 = GCNConv(dim_in, H), GCNConv(H, H), GCNConv(H, H)
            self.linear1, self.linear2 = torch.nn.Linear(H, dim_h), torch.nn.Linear(dim_h, dim_out)
            self.dropout = torch.nn.Dropout(p=0.2)
        def forward(self, x, ei, ea=None):
            h = self.dropout(torch.relu(self.batch_norm1(self.gcn1(x, ei, ea))))
            h2 = self.dropout(torch.relu(self.batch_norm2(self.gcn2(h, ei, ea)))) + h
            h3 = self.dropout(torch.relu(self.batch_norm3(self.gcn3(h2, ei, ea)))) + h2
            return self.linear2(torch.relu(self.linear1(h3)))

    sd = torch.load(f"{SM}/{wdn}/gnn_model/best_model.pt", map_location="cpu", weights_only=False)
    model = GCN(x_features.shape[1]); model.load_state_dict(sd); model.eval()
    nodes = list(G.nodes)
    # Use the dataset's own edge_index AND edge_attr.  Rebuilding edge_index from the pickled
    # graph gives 30 directed edges where the dataset stores 15, and omitting edge_attr changes
    # the predictions materially -- both silently degrade the model.
    ds = torch.load(f"{SM}/{wdn}/data_generator/test_dataset.pt", weights_only=False)
    ei, ea = ds[0].edge_index, ds[0].edge_attr
    with torch.no_grad():
        y = model(torch.tensor(x_features, dtype=torch.float32), ei, ea).numpy().ravel()
    minp, maxp = stats["min_p"], stats["max_p"]
    rn = str(stats["reservoir_node"])
    h = {}
    for k, nd in enumerate(nodes):
        nm = str(nd).replace("meas_", "")
        if nm == rn: h[nm] = float(stats["reservoir_head"])   # known BC, and a masked output
        elif nm not in h:
            elev = net.nodes[nm].elevation_m if nm in net.nodes else 0.0
            h[nm] = float(y[k] * (maxp - minp) + minp + elev)
    r = {k: v["r_e"] for k, v in compute_pipe_resistances_hw(net).items()}
    d = {j: 0.0 for j in J}
    for pid, p in net.pipes.items():
        a, b = p.start_node, p.end_node
        if a not in h or b not in h: continue
        dh = h[a] - h[b]
        q = np.sign(dh) * (abs(dh) / r[pid]) ** (1 / 1.852)
        if a in d: d[a] -= q
        if b in d: d[b] += q
    return np.array([d[j] for j in J], dtype=float)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wdn", default="Alperovits")
    ap.add_argument("--seed", type=int, default=0, help="which random observation to draw")
    ap.add_argument("--observe-total", action="store_true",
                    help="also condition on the total demand (collapses the noiseless posterior)")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="residual below which a candidate counts as exactly consistent (m). EPANET reports heads to ~3e-5 m, so anything tighter rejects the TRUE node too.")
    ap.add_argument("--out", default="observation_vs_prediction.png")
    args = ap.parse_args()

    inp, net, J, d0, sensors, delta, G, stats, mult = load_case(args.wdn)
    heads = solver(inp, J, mult)
    sc = [J.index(s) for s in sensors]
    rng = np.random.default_rng(args.seed)

    v_true = int(rng.integers(len(J)))
    lam_true = float(rng.uniform(0, delta))
    d_true = d0.copy(); d_true[v_true] += lam_true
    obs = heads(d_true)[sc]
    print(f"{args.wdn}: {len(J)} junctions, sensors {sensors}, extra_demand {delta}")
    print(f"TRUE leak: node {J[v_true]}  lambda {lam_true:.4f}  (D = {d_true.sum():.4f})")
    print(f"observed sensor heads (noiseless): " + ", ".join(f"{s}={o:.6f}" for s, o in zip(sensors, obs)))

    rows = []
    for v in range(len(J)):
        lam, res, bracketed = consistent_lambda(heads, d0, v, sc, obs, delta)
        d = d0.copy(); d[v] += lam
        if args.observe_total and abs(d.sum() - d_true.sum()) > 1e-9:
            continue
        w, jnorm = cov_weight(heads, d0, v, sc, lam)
        rows.append(dict(v=v, node=J[v], lam=lam, resid=res, d=d, w=w, jnorm=jnorm,
                         ok=bracketed and res < args.tol))
    cons = [r for r in rows if r["ok"]]
    tot = sum(r["w"] for r in cons)
    for r in rows:
        r["p"] = (r["w"] / tot) if (r["ok"] and tot > 0) else 0.0
    print(f"\ncandidates exactly consistent with the observation: {len(cons)} of {len(J)}"
          + ("  (total demand also conditioned)" if args.observe_total else ""))
    print(f"  {'node':>6} {'lambda':>9} {'|dh/dlam|':>11} {'P(v|z)':>9} {'||d-d_true||':>13}")
    for r in rows:
        mark = "  <-- TRUE" if r["v"] == v_true else ("" if r["ok"] else "  (no exact fit)")
        print(f"  {r['node']:>6} {r['lam']:9.4f} {r['jnorm']:11.3f} {r['p']:9.4f} "
              f"{np.linalg.norm(r['d'] - d_true):13.4f}{mark}")

    # --- GNN prediction on this observation
    d_gnn = None
    try:
        # Columns 0 (base pressure) and 2 (node type) are CONSTANT across the dataset, so take
        # them verbatim from a stored sample rather than re-deriving the generator's convention
        # (node type is 0 junction / 2 reservoir / 3 for the meas_ node -- not what one guesses).
        # Only column 1, the observed sensor reading broadcast to every node, depends on z.
        import torch as _t
        ds = _t.load(f"{SM}/{args.wdn}/data_generator/test_dataset.pt", weights_only=False)
        X = ds[0].x.numpy().copy()
        minp, maxp = stats["min_p"], stats["max_p"]
        h_true_all = heads(d_true)
        obs_norm = float(np.mean([(h_true_all[J.index(s)] - net.nodes[s].elevation_m - minp)
                                  / (maxp - minp) for s in sensors]))
        X[:, 1] = obs_norm
        d_gnn = gnn_predict(args.wdn, net, J, G, stats, X)
        print(f"\nGNN prediction:  ||d_gnn - d_true|| = {np.linalg.norm(d_gnn - d_true):.4f}")
    except Exception as e:
        print(f"\nGNN prediction unavailable: {type(e).__name__}: {e}")

    # --- spread of the consistent set = the irreducible ambiguity
    if cons:
        Dm = np.array([r["d"] for r in cons]); pw = np.array([r["p"] for r in cons])
        pm = pw @ Dm
        spread = np.sqrt(pw @ (Dm - pm) ** 2)
        print(f"\nirreducible ambiguity (posterior std across the {len(cons)} consistent scenarios):")
        print(f"  per junction: median {np.median(spread):.4f}, max {spread.max():.4f}")
        print(f"  posterior-mean error ||E[d|z] - d_true||   = {np.linalg.norm(pm - d_true):.4f}  (weighted)")
        print(f"  unweighted-mean error                      = {np.linalg.norm(Dm.mean(axis=0) - d_true):.4f}")

    _plot(args, J, d0, d_true, v_true, rows, cons, d_gnn, sensors)
    return 0


def _plot(args, J, d0, d_true, v_true, rows, cons, d_gnn, sensors):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(J))
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[2, 1])
    for r in rows:
        if r["ok"]:
            ax[0].plot(x, r["d"], marker="o", ms=4, lw=1, alpha=.55,
                       color="tab:blue", label="_")
    if cons:
        ax[0].plot([], [], color="tab:blue", marker="o", ms=4, lw=1, alpha=.55,
                   label=f"consistent scenarios (n={len(cons)})")
    ax[0].plot(x, d0, "k--", lw=1, alpha=.6, label="base demand $d^0$")
    ax[0].plot(x, d_true, color="tab:green", marker="s", ms=7, lw=2, label="truth")
    if d_gnn is not None:
        ax[0].plot(x, d_gnn, color="tab:red", marker="^", ms=7, lw=2, label="GNN prediction")
    ax[0].set_xticks(x); ax[0].set_xticklabels(J)
    for s in sensors:
        ax[0].axvline(J.index(s), color="grey", ls=":", lw=1)
    ax[0].set_ylabel("demand"); ax[0].legend(fontsize=8, ncol=2)
    ax[0].set_title(f"{args.wdn}: scenarios consistent with one noiseless observation "
                    f"(true leak at {J[v_true]}; dotted = sensors)")

    errs = [np.linalg.norm(r["d"] - d_true) for r in rows if r["ok"]]
    ax[1].bar(np.arange(len(errs)), errs, color="tab:blue", alpha=.7, label="consistent scenarios")
    if d_gnn is not None:
        ax[1].axhline(np.linalg.norm(d_gnn - d_true), color="tab:red", lw=2, label="GNN error")
    ax[1].set_ylabel(r"$\|d - d_{true}\|_2$"); ax[1].set_xlabel("consistent scenario")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(args.out, dpi=140)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
