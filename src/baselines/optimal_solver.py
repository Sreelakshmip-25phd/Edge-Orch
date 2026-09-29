"""OPTIONAL offline upper bound - a ceiling, NOT a competing online system.

It sees a whole batch of requests at once (future arrivals, true types,
true sizes, true lifetimes), ignores real-time ordering and decision
latency, and answers: at most how many of these requests could have been
served to completion (at any degradation level the service allows)?

Formulation (a *relaxation* of every online placement policy, so its
optimum is a valid upper bound even for systems that migrate, pre-empt or
degrade):
    y[r, l] in {0,1}  request r served at level l in {1.0, 0.8, 0.6, 0.4},
                      l >= r's degradation floor; sum_l y[r, l] <= 1
    at every checkpoint (arrival instant) tau, with A(tau) the requests
    active at tau:
      per zone z:  sum_{r in A(tau), r zone_local in z} l*cpu_r*y[r,l] <= CPU_z   (and mem)
      global:      sum_{r in A(tau)} l*cpu_r*y[r,l] <= CPU_total               (and mem)
    y[r, l] = 0 if l*cpu_r / l*mem_r doesn't fit the largest node of any
    zone r may use.
    maximise sum y  (served count; ties broken towards higher levels)
Capacity is pooled per zone (no bin-packing fragmentation) and cross-zone
requests may use the whole cluster - that is what makes it a relaxation.
Node failures are ignored (more capacity -> still an upper bound).

Solved per time window (default: requests arriving in each 2-hour window,
starting from an empty cluster - also only loosens the bound) with scipy's
HiGHS MILP; if the MILP hits its time limit, the LP relaxation value is
reported instead and labelled as such ("lp_relaxation").
"""
import numpy as np

LEVELS = (1.0, 0.8, 0.6, 0.4)


def _checkpoint_rows(reqs, zone_of, zones, cap_z, var_index, lifetimes):
    from scipy.sparse import coo_matrix
    rows, cols, vals, rhs = [], [], [], []
    starts = np.array([r["t_s"] for r in reqs])
    ends = starts + np.array(lifetimes)
    order = np.argsort(starts)
    row = 0
    tot_cpu = sum(c for c, _ in cap_z.values())
    tot_mem = sum(m for _, m in cap_z.values())
    for i in order:
        tau = starts[i]
        active = np.where((starts <= tau) & (ends > tau))[0]
        if len(active) == 0:
            continue
        for res_i, cap_tot in ((0, tot_cpu), (1, tot_mem)):
            # global
            for a in active:
                for l, j in var_index[a]:
                    cols.append(j)
                    rows.append(row)
                    vals.append(l * (reqs[a]["truth"]["cpu"] if res_i == 0 else reqs[a]["truth"]["mem"]))
            rhs.append(cap_tot)
            row += 1
            # per zone, zone_local only
            for z in zones:
                loc = [a for a in active if reqs[a]["truth"]["data_locality"] == "zone_local"
                       and zone_of[a] == z]
                if not loc:
                    continue
                for a in loc:
                    for l, j in var_index[a]:
                        cols.append(j)
                        rows.append(row)
                        vals.append(l * (reqs[a]["truth"]["cpu"] if res_i == 0 else reqs[a]["truth"]["mem"]))
                rhs.append(cap_z[z][res_i])
                row += 1
    n_vars = sum(len(v) for v in var_index)
    return coo_matrix((vals, (rows, cols)), shape=(row, n_vars)).tocsr(), np.array(rhs)


def window_bound(topo, reqs, lifetimes, time_limit_s=60.0, mip=True):
    """Upper bound on requests served to completion among `reqs`."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix, vstack
    if not reqs:
        return {"n": 0, "bound": 0, "kind": "empty"}
    zones = [z["zone_id"] for z in topo["zones"]]
    cap_z = {z["zone_id"]: (sum(n["cpu"] for n in z["nodes"]), sum(n["mem_gb"] for n in z["nodes"]))
             for z in topo["zones"]}
    big_z = {z["zone_id"]: (max(n["cpu"] for n in z["nodes"]), max(n["mem_gb"] for n in z["nodes"]))
             for z in topo["zones"]}
    big_any = (max(c for c, _ in big_z.values()), max(m for _, m in big_z.values()))
    zone_of = [r["origin_zone"] for r in reqs]
    var_index, j = [], 0
    for a, r in enumerate(reqs):
        tr = r["truth"]
        bc, bm = big_z[zone_of[a]] if tr["data_locality"] == "zone_local" else big_any
        vs = []
        for l in LEVELS:
            if l + 1e-9 < tr.get("degrade_floor", 1.0):
                continue
            if l * tr["cpu"] <= bc + 1e-9 and l * tr["mem"] <= bm + 1e-9:
                vs.append((l, j))
                j += 1
        var_index.append(vs)
    n = j
    if n == 0:
        return {"n": len(reqs), "bound": 0, "kind": "none_fit"}
    A_cap, b_cap = _checkpoint_rows(reqs, zone_of, zones, cap_z, var_index, lifetimes)
    rr, cc = [], []
    for a, vs in enumerate(var_index):
        for _, jj in vs:
            rr.append(a)
            cc.append(jj)
    A_one = coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(len(reqs), n)).tocsr()
    A = vstack([A_cap, A_one]).tocsr()
    ub = np.concatenate([b_cap, np.ones(len(reqs))])
    c = np.zeros(n)
    for vs in var_index:
        for l, jj in vs:
            c[jj] = -(1.0 + 1e-3 * l)           # served count, prefer higher levels
    cons = LinearConstraint(A, -np.inf, ub)
    kind = "ilp"
    res = None
    if mip:
        res = milp(c, constraints=cons, integrality=np.ones(n), bounds=Bounds(0, 1),
                   options={"time_limit": time_limit_s, "disp": False})
        if res.status != 0 or res.x is None:
            res = None
    if res is None:
        kind = "lp_relaxation"
        res = milp(c, constraints=cons, integrality=np.zeros(n), bounds=Bounds(0, 1),
                   options={"time_limit": time_limit_s, "disp": False})
    served = float(sum(res.x[jj] for vs in var_index for _, jj in vs)) if res.x is not None else float("nan")
    return {"n": len(reqs), "bound": served, "bound_rate": served / len(reqs), "kind": kind,
            "status": int(res.status), "n_vars": n, "n_rows": A.shape[0]}


def batch_bound(topo, wl, window_s=7200.0, max_windows=None, **kw):
    from workload import lifetime_of
    reqs = wl["requests"]
    horizon = wl["meta"]["horizon_s"]
    out = []
    edges = np.arange(0.0, horizon + window_s, window_s)
    for k in range(len(edges) - 1):
        if max_windows is not None and k >= max_windows:
            break
        w = [r for r in reqs if edges[k] <= r["t_s"] < edges[k + 1]]
        lt = [lifetime_of(r, wl) for r in w]
        b = window_bound(topo, w, lt, **kw)
        b.update(window=[float(edges[k]), float(edges[k + 1])], req_ids=[r["req_id"] for r in w])
        out.append(b)
    return out
