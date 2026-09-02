#!/usr/bin/env python3
"""Hierarchical prior: perturbed base demands + one leak, solved by Rao-Blackwellised enumeration.

Prior
-----
    d0~ ~ N(d0, Sigma0),   v ~ Unif(J),   lambda ~ U[0, Delta],   d = d0~ + lambda e_v

Conditional on a base draw d0~, the enumeration of scripts/ltown_c_enumerate.py still applies:
the candidate set is finite and lambda is recovered from the sensor constraint.  The question
this script answers empirically is how to COMBINE the per-draw results.

Three estimators, on identical observations:

  correct  P(v|z) prop  sum_m w[m,v]                       (sum of UNNORMALISED weights)
  naive    P(v|z)  =    mean_m ( w[m,:] / sum_u w[m,u] )   (average of NORMALISED posteriors)
  abc      draw (d0~, v, lambda) jointly from the prior, weight by K_eps  -- unbiased reference

with per-draw evidence  w[m,v] = p(lambda) * ||d h_Y / d lambda||^-1 * K_eps(residual).

Writing E_m = sum_u w[m,u], the correct estimator is  sum_m E_m P(v|z,d0~_m)  while the naive
one is  mean_m P(v|z,d0~_m):  the naive estimator replaces the evidence weights by uniform
weights, so it is exact iff every perturbed base explains the observation equally well --
exactly what Sigma0 > 0 destroys.

The "correct" estimator is Rao-Blackwellised: the discrete label v is integrated out exactly
rather than sampled, so by the Rao-Blackwell theorem it should have lower variance than ABC
at equal cost.  This script measures whether it does.
"""
from __future__ import annotations
import argparse, importlib.util, json, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _ovp():
    spec = importlib.util.spec_from_file_location(
        "ovp", os.path.join(ROOT, "scripts", "observation_vs_prediction.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wdn", default="Alperovits")
    ap.add_argument("--sigma", type=float, default=0.10, help="relative base-demand std (0.10 = BattLeDIM +-10%%)")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--bases", type=int, default=300, help="M, number of perturbed base draws")
    ap.add_argument("--abc", type=int, default=30000,
        help="ABC reference draws; 0 skips it.  ABC is only needed to CONFIRM the "
             "correct estimator; the naive-vs-correct gap is measured directly as "
             "TV(naive, correct), which needs no reference.")
    ap.add_argument("--scenarios", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ovp = _ovp()
    inp, net, J, d0, sensors, delta, G, stats, mult = ovp.load_case(args.wdn)
    heads = ovp.solver(inp, J, mult)
    sc = [J.index(s) for s in sensors]
    n = len(J)
    sigma = args.sigma * np.maximum(d0, d0.sum() / n)
    print(f"{args.wdn}: {n} junctions, sensors {sensors}, Delta {delta:.4f}, "
          f"eps {args.eps}, sigma_rel {args.sigma}")
    print(f"M = {args.bases} base draws, ABC reference {args.abc} draws, "
          f"{args.scenarios} scenarios\n")

    rows = []
    for sc_i in range(args.scenarios):
        rng = np.random.default_rng(args.seed + 1000 * sc_i)
        # --- truth drawn from the hierarchical prior: the model mismatch is real
        base_true = rng.normal(d0, sigma)
        v_true = int(rng.integers(n)); lam_true = float(rng.uniform(0, delta))
        d_true = base_true.copy(); d_true[v_true] += lam_true
        obs = heads(d_true)[sc]

        # --- enumeration over M perturbed bases
        t0 = time.perf_counter(); calls0 = [0]
        def counted(d, _c=calls0):
            _c[0] += 1
            return heads(d)
        W = np.zeros((args.bases, n))
        Dm = np.zeros((args.bases, n, n))
        for m in range(args.bases):
            base_m = rng.normal(d0, sigma)
            for v in range(n):
                lam, r, ok = ovp.consistent_lambda(counted, base_m, v, sc, obs, delta)
                dd = base_m.copy(); dd[v] += lam
                Dm[m, v] = dd
                if not ok or not (0.0 <= lam <= delta):
                    continue                       # outside the lambda prior support
                jw, _ = ovp.cov_weight(counted, base_m, v, sc, lam)
                W[m, v] = jw * float(np.exp(-0.5 * (r / args.eps) ** 2))
        t_enum = time.perf_counter() - t0; c_enum = calls0[0]

        Em = W.sum(axis=1)
        good = Em > 0
        if not good.any():
            print(f"scenario {sc_i}: all base draws underflowed; skipping"); continue
        P_correct = W.sum(axis=0); P_correct = P_correct / P_correct.sum()
        P_naive = (W[good] / Em[good, None]).mean(axis=0)
        # posterior mean demand under each
        Wf = W.reshape(-1); Df = Dm.reshape(-1, n)
        d_correct = (Wf @ Df) / Wf.sum()
        d_naive = np.stack([(W[m] @ Dm[m]) / Em[m] for m in np.where(good)[0]]).mean(axis=0)

        tv_nc = tv(P_naive, P_correct)
        spread = float(Em[good].max() / max(Em[good].min(), 1e-300))

        if args.abc <= 0:
            r = dict(scenario=sc_i, v_true=J[v_true], lam_true=lam_true,
                     tv_naive_correct=tv_nc, evidence_spread=spread,
                     tv_correct=float("nan"), tv_naive=float("nan"),
                     p_true_correct=float(P_correct[v_true]), p_true_naive=float(P_naive[v_true]),
                     p_true_abc=float("nan"),
                     derr_correct=float(np.abs(d_correct - d_true).mean()),
                     derr_naive=float(np.abs(d_naive - d_true).mean()),
                     derr_abc=float("nan"), ess_abc=float("nan"),
                     calls_enum=c_enum, calls_abc=0, t_enum=t_enum, t_abc=0.0)
            rows.append(r)
            print(f"scenario {sc_i}: true leak {J[v_true]}, lambda {lam_true:.4f}")
            print(f"   TV(naive, correct)  : {tv_nc:.4f}"
                  f"   | evidence spread E_max/E_min = {spread:.3g}")
            print(f"   P(true node)        : correct {P_correct[v_true]:.4f}   "
                  f"naive {P_naive[v_true]:.4f}")
            print(f"   demand MAE          : correct {r['derr_correct']:.4f}   "
                  f"naive {r['derr_naive']:.4f}")
            print(f"   cost: enum {c_enum} solves ({t_enum:.0f}s)", flush=True)
            continue

        # --- ABC reference over the full hierarchical prior
        t0 = time.perf_counter()
        Pa = np.zeros(n); wsum = 0.0; d_abc = np.zeros(n); w2 = 0.0
        for _ in range(args.abc):
            b = rng.normal(d0, sigma)
            v = int(rng.integers(n)); lam = float(rng.uniform(0, delta))
            d = b.copy(); d[v] += lam
            r = float(np.linalg.norm(heads(d)[sc] - obs))
            w = float(np.exp(-0.5 * (r / args.eps) ** 2))
            Pa[v] += w; wsum += w; d_abc += w * d; w2 += w * w
        t_abc = time.perf_counter() - t0
        if wsum <= 0:
            print(f"scenario {sc_i}: ABC underflowed; skipping"); continue
        ess_abc = wsum ** 2 / w2
        Pa = Pa / wsum; d_abc = d_abc / wsum

        r = dict(scenario=sc_i, v_true=J[v_true], lam_true=lam_true,
                 tv_naive_correct=tv_nc, evidence_spread=spread,
                 tv_correct=tv(P_correct, Pa), tv_naive=tv(P_naive, Pa),
                 p_true_correct=float(P_correct[v_true]), p_true_naive=float(P_naive[v_true]),
                 p_true_abc=float(Pa[v_true]),
                 derr_correct=float(np.abs(d_correct - d_true).mean()),
                 derr_naive=float(np.abs(d_naive - d_true).mean()),
                 derr_abc=float(np.abs(d_abc - d_true).mean()),
                 ess_abc=float(ess_abc), calls_enum=c_enum, calls_abc=args.abc,
                 t_enum=t_enum, t_abc=t_abc)
        rows.append(r)
        print(f"scenario {sc_i}: true leak {J[v_true]}, lambda {lam_true:.4f}")
        print(f"   TV to ABC reference : correct {r['tv_correct']:.4f}   naive {r['tv_naive']:.4f}")
        print(f"   P(true node)        : correct {r['p_true_correct']:.4f}   "
              f"naive {r['p_true_naive']:.4f}   abc {r['p_true_abc']:.4f}")
        print(f"   demand MAE          : correct {r['derr_correct']:.4f}   "
              f"naive {r['derr_naive']:.4f}   abc {r['derr_abc']:.4f}")
        print(f"   cost: enum {c_enum} solves ({t_enum:.0f}s) | abc {args.abc} solves "
              f"({t_abc:.0f}s, ESS {ess_abc:.0f})", flush=True)

    if rows:
        def mean(k): return float(np.mean([r[k] for r in rows]))
        print(f"\n=== averages over {len(rows)} scenarios ===")
        print(f"TV(naive, correct)  : {mean('tv_naive_correct'):.4f}"
              f"   | evidence spread E_max/E_min {mean('evidence_spread'):.3g}")
        if args.abc > 0:
            print(f"TV to ABC reference : correct {mean('tv_correct'):.4f}   naive {mean('tv_naive'):.4f}"
                  f"   -> naive is {mean('tv_naive')/max(mean('tv_correct'),1e-12):.1f}x further off")
        print(f"demand MAE          : correct {mean('derr_correct'):.5f}   "
              f"naive {mean('derr_naive'):.5f}   abc {mean('derr_abc'):.5f}")
        print(f"cost per scenario   : enum {mean('calls_enum'):.0f} solves ({mean('t_enum'):.0f}s)"
              f"  |  abc {args.abc} solves ({mean('t_abc'):.0f}s, ESS {mean('ess_abc'):.0f})")
        eff = mean('ess_abc') / max(mean('calls_abc'), 1)
        print(f"ABC effective samples per solve: {eff:.4f}")
        if args.out:
            json.dump(rows, open(args.out, "w"), indent=2)
            print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
