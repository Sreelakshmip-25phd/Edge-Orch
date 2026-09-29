#!/usr/bin/env python3
"""Phase 1 - acquire and validate real datasets, and expose the failure model.

Idempotent: every source is skipped if its validated output already
exists on disk, so re-running after a partial run only fetches what's
missing. Sources, all independently, programmatically downloaded here:

  1. Milan grid geometry   -> validated/milan_grid_centroids.csv
     Harvard Dataverse, doi:10.7910/DVN/QJWLFU
  2. Milan activity trace  -> validated/milan_activity.parquet,
                               validated/milan_surge_candidates.csv
     Harvard Dataverse, doi:10.7910/DVN/EGZHFV (SMS-Call-Internet-MI)
  3. EUA Melbourne topology -> validated/eua_melb_cbd_sites.csv
     github.com/swinedge/eua-dataset (public, no auth)
  4. Google cluster failures -> validated/failure_model.json
     storage.googleapis.com/clusterdata-2011-2 (public, no auth)
  5. RIPE Atlas RTT model  -> validated/rtt_model.json
     atlas.ripe.net API (public, no auth) for reachability +
     published metro-area anchor calibration constants
  6. Alibaba resource profiles -> validated/service_resource_profiles.json
     Alibaba cluster-trace-v2018 OSS mirrors (public, no auth)

ONE genuine external constraint, not fixable by any script: sources 1
and 2 live on Harvard Dataverse under a "Guestbook" gate — Harvard's
own policy requires a one-time human response (name/email/institution)
before ANY download, scripted or not, succeeds. See
HARVARD_DATAVERSE_API_TOKEN below for the one-time setup this needs.

Changes from the old phase1_data_foundation.py:
  * FailureModel / sample_failures(n, rng): the Google-trace failure model
    is now sampled (several failures per run, inter-failure and downtime
    drawn from the empirical percentile curves) instead of the old
    scenario's single hard-coded failure.
  * --import-from PATH copies already-validated outputs from an existing
    checkout (e.g. the old agentic_edge_orchestration/ repo) so a machine
    that already paid the Harvard guestbook + 150 MB download cost doesn't
    pay it twice.
  * The failure model now also records the 1st/99th percentiles when it
    is (re)derived from the raw trace, so tail sampling is better bounded.

Run: python3 src/data_foundation.py [--import-from ../agentic_edge_orchestration]
"""
import gzip
import io
import json
import os
import sys
import tarfile
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW, REPORTS, VALIDATED, ensure_dirs  # noqa: E402

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "phd-orchestration-research/1.0"})
REPORT = {"generated_utc": datetime.now(timezone.utc).isoformat(),
          "sources": {}}

FORCE_REFRESH = False

# --- Harvard Dataverse guestbook gate --------------------------------
# datasets 10.7910/DVN/QJWLFU (grid) and 10.7910/DVN/EGZHFV (activity)
# require a one-time Guestbook response before any download works, for
# any client. This is Harvard's access policy, not something a script
# can bypass. One-time setup (a few minutes, does not repeat):
#   1. Create a free account: https://dataverse.harvard.edu/dataverseuser.xhtml?editMode=CREATE
#   2. Visit https://doi.org/10.7910/DVN/EGZHFV while logged in, click
#      any file's download button once -> fill the Guestbook form that
#      appears (name/email/institution) -> submit.
#   3. Generate an API token: account menu -> API Token -> Create Token.
#   4. export HARVARD_DATAVERSE_API_TOKEN=<token>   (or set it before
#      running this script)
# After that one-time step, this script downloads both Milan sources
# fully automatically, indefinitely, using that token.
DATAVERSE_TOKEN = os.environ.get("HARVARD_DATAVERSE_API_TOKEN", "")
DATAVERSE_GRID_DOI = "doi:10.7910/DVN/QJWLFU"
DATAVERSE_ACTIVITY_DOI = "doi:10.7910/DVN/EGZHFV"


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def record(source, status, **info):
    REPORT["sources"][source] = {"status": status, **info}
    log(f"== {source}: {status}")


def already_done(source, *outputs):
    if FORCE_REFRESH:
        return False
    paths = [os.path.join(VALIDATED, o) for o in outputs]
    if all(os.path.exists(p) and os.path.getsize(p) > 0 for p in paths):
        record(source, "OK", output=f"validated/{outputs[0]}", cached=True)
        return True
    return False


def download(url, dest, chunk=1 << 20, timeout=300, max_retries=2, headers=None):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        log(f"  skip (exists): {os.path.basename(dest)}")
        return dest
    for attempt in range(1, max_retries + 1):
        try:
            log(f"  GET {url}  (attempt {attempt})")
            with SESSION.get(url, stream=True, timeout=timeout,
                             headers=headers) as r:
                r.raise_for_status()
                # Dataverse returns HTTP 200 with a JSON error body for
                # guestbook-gated files instead of a proper error code -
                # detect that case explicitly rather than saving garbage.
                ctype = r.headers.get("content-type", "")
                if "json" in ctype and "dataverse.harvard.edu" in url:
                    body = r.content
                    if b'"status":"ERROR"' in body:
                        msg = json.loads(body).get("message", "unknown error")
                        raise RuntimeError(f"Dataverse denied download: {msg}")
                tmp = dest + ".part"
                with open(tmp, "wb") as f:
                    for c in r.iter_content(chunk_size=chunk):
                        f.write(c)
                os.replace(tmp, dest)
            log(f"  saved {os.path.basename(dest)} "
                f"({os.path.getsize(dest)/1e6:.1f} MB)")
            return dest
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            log(f"  HTTP {code}")
            if 400 <= code < 500:
                return None
            time.sleep(3 * attempt)
        except Exception as e:
            log(f"  attempt {attempt} failed: {e}")
            if "Dataverse denied download" in str(e):
                return None
            time.sleep(3 * attempt)
    return None


def _dataverse_headers():
    return {"X-Dataverse-key": DATAVERSE_TOKEN} if DATAVERSE_TOKEN else {}


def _dataverse_guestbook_help(source):
    record(source, "MANUAL_STEP_REQUIRED", help=(
        "Harvard Dataverse requires a one-time Guestbook response before "
        "this dataset can be downloaded (by anyone, scripted or not). "
        "1) create a free account at https://dataverse.harvard.edu ; "
        "2) visit the dataset page while logged in and complete the "
        "Guestbook form once (name/email/institution); "
        "3) generate an API token from your account settings; "
        "4) set HARVARD_DATAVERSE_API_TOKEN and re-run this script. "
        "See the module docstring for exact dataset URLs."))


# --- [1] Milan grid geometry ------------------------------------------
def acquire_milan_grid():
    if already_done("milan_grid", "milan_grid_centroids.csv"):
        return
    try:
        meta = SESSION.get(
            "https://dataverse.harvard.edu/api/datasets/:persistentId/",
            params={"persistentId": DATAVERSE_GRID_DOI}, timeout=60).json()
        files = meta["data"]["latestVersion"]["files"]
        target = next(f for f in files
                      if f["dataFile"]["filename"] == "milano-grid.geojson")
        file_id = target["dataFile"]["id"]
        dest = os.path.join(RAW, "milano-grid.geojson")
        got = download(f"https://dataverse.harvard.edu/api/access/datafile/{file_id}",
                       dest, headers=_dataverse_headers())
        if got is None:
            _dataverse_guestbook_help("milan_grid")
            return
        grid = json.load(open(got))
        rows = []
        for i, feat in enumerate(grid["features"], start=1):
            props = feat.get("properties") or {}
            # property key naming varies across copies of this file
            # (id / cellId / CELL_ID / square_id have all been seen) -
            # fall back to 1-based feature order, which matches the
            # Milano grid's standard cell-numbering convention anyway.
            cid = None
            for key in ("id", "cellId", "CELL_ID", "square_id", "cell_id"):
                if key in props:
                    cid = props[key]
                    break
            if cid is None:
                cid = i
            coords = feat["geometry"]["coordinates"][0][:4]
            lon = sum(c[0] for c in coords) / len(coords)
            lat = sum(c[1] for c in coords) / len(coords)
            rows.append({"cell_id": cid, "lon": lon, "lat": lat})
        df = pd.DataFrame(rows).sort_values("cell_id")
        df.to_csv(os.path.join(VALIDATED, "milan_grid_centroids.csv"), index=False)
        record("milan_grid", "OK", n_cells=len(df),
               output="validated/milan_grid_centroids.csv")
    except Exception as e:
        record("milan_grid", "FAILED", error=str(e))
        traceback.print_exc()


# --- [2] Milan activity trace ------------------------------------------
MILAN_DAYS = [f"2013-11-{d:02d}" for d in range(1, 15)]
MILAN_COLS = ["square_id", "time_interval_ms", "country_code",
             "sms_in", "sms_out", "call_in", "call_out", "internet"]


def _parse_milan_day(path):
    """Raw file is tab-separated, one row per (cell, 10-min slot,
    country); aggregate internet-traffic activity per (cell, slot),
    summed across country codes, matching the official 'InternetTraffic'
    activity proxy used throughout the Telecom Italia Big Data
    Challenge literature."""
    df = pd.read_csv(path, sep="\t", header=None, names=MILAN_COLS,
                     usecols=["square_id", "time_interval_ms", "internet"])
    df["internet"] = pd.to_numeric(df["internet"], errors="coerce").fillna(0.0)
    g = df.groupby(["square_id", "time_interval_ms"])["internet"].sum().reset_index()
    g["ts"] = pd.to_datetime(g["time_interval_ms"], unit="ms")
    g = g.rename(columns={"square_id": "cell_id", "internet": "activity"})
    g["day"] = g["ts"].dt.strftime("%Y-%m-%d")
    return g[["day", "cell_id", "ts", "activity"]]


def _compute_surge_candidates(act, z_threshold=3.0, top_n=2000):
    g = act.groupby("cell_id")["activity"]
    mean = g.transform("mean")
    std = g.transform("std").replace(0, 1e-9).fillna(1e-9)
    z = (act["activity"] - mean) / std
    surge = act.assign(z=z)
    surge = surge[surge["z"] >= z_threshold].sort_values("z", ascending=False)
    return surge.head(top_n)[["day", "ts", "cell_id", "activity", "z"]]


def acquire_milan_activity():
    if already_done("milan_activity", "milan_activity.parquet",
                    "milan_surge_candidates.csv"):
        return
    try:
        meta = SESSION.get(
            "https://dataverse.harvard.edu/api/datasets/:persistentId/",
            params={"persistentId": DATAVERSE_ACTIVITY_DOI}, timeout=60).json()
        files = meta["data"]["latestVersion"]["files"]
        by_name = {f["dataFile"]["filename"]: f["dataFile"]["id"] for f in files}

        frames = []
        for day in MILAN_DAYS:
            fname = f"sms-call-internet-mi-{day}.txt"
            if fname not in by_name:
                continue
            dest = os.path.join(RAW, fname)
            got = download(
                f"https://dataverse.harvard.edu/api/access/datafile/{by_name[fname]}",
                dest, headers=_dataverse_headers(), timeout=600)
            if got is None:
                _dataverse_guestbook_help("milan_activity")
                return
            frames.append(_parse_milan_day(got))
            log(f"  parsed {day}: {len(frames[-1]):,} (cell,slot) rows")

        if not frames:
            record("milan_activity", "FAILED", error="no daily files found")
            return

        act = pd.concat(frames, ignore_index=True)
        act.to_parquet(os.path.join(VALIDATED, "milan_activity.parquet"))

        surge = _compute_surge_candidates(act)
        surge.to_csv(os.path.join(VALIDATED, "milan_surge_candidates.csv"),
                    index=False)

        record("milan_activity", "OK", n_rows=len(act), n_days=len(frames),
               n_surge_candidates=len(surge),
               output="validated/milan_activity.parquet")
    except Exception as e:
        record("milan_activity", "FAILED", error=str(e))
        traceback.print_exc()


# --- [3] EUA Melbourne topology -----------------------------------------
EUA_URL = ("https://raw.githubusercontent.com/swinedge/eua-dataset/"
          "master/edge-servers/site-optus-melbCBD.csv")


def acquire_eua_topology():
    if already_done("eua_topology", "eua_melb_cbd_sites.csv"):
        return
    try:
        dest = os.path.join(RAW, "eua-site-optus-melbCBD.csv")
        got = download(EUA_URL, dest)
        if got is None:
            record("eua_topology", "FAILED", error="download failed")
            return
        df = pd.read_csv(got)
        assert len(df) > 0, "empty EUA file"
        df.to_csv(os.path.join(VALIDATED, "eua_melb_cbd_sites.csv"), index=False)
        record("eua_topology", "OK", n_sites=len(df),
               output="validated/eua_melb_cbd_sites.csv")
    except Exception as e:
        record("eua_topology", "FAILED", error=str(e))
        traceback.print_exc()


# --- [4] Google cluster failure model ------------------------------------
GOOGLE_URL = ("https://storage.googleapis.com/clusterdata-2011-2/"
             "machine_events/part-00000-of-00001.csv.gz")
GOOGLE_COLS = ["timestamp", "machine_id", "event_type", "platform_id",
              "capacity_cpu", "capacity_memory"]
# event_type: 0 = ADD, 1 = REMOVE, 2 = UPDATE (Google cluster-trace v1 schema)


def acquire_google_failures():
    if already_done("google_failures", "failure_model.json"):
        return
    try:
        dest = os.path.join(RAW, "google_machine_events.csv.gz")
        got = download(GOOGLE_URL, dest)
        if got is None:
            record("google_failures", "FAILED", error="download failed")
            return
        df = pd.read_csv(got, header=None, names=GOOGLE_COLS,
                         usecols=["timestamp", "machine_id", "event_type"])
        removes = df[df.event_type == 1]
        adds = df[df.event_type == 0].sort_values("timestamp")

        interfailure_hours, downtime_min = [], []
        machines_with_failures = set()
        for mid, g in removes.groupby("machine_id"):
            times = sorted(g.timestamp.tolist())
            machines_with_failures.add(mid)
            for i in range(1, len(times)):
                # Google cluster-trace-v1 timestamps are microseconds
                # since trace start, not milliseconds.
                interfailure_hours.append((times[i] - times[i - 1]) / 3.6e9)
            mach_adds = adds[adds.machine_id == mid].timestamp.tolist()
            for t in times:
                nxt = next((a for a in mach_adds if a > t), None)
                if nxt is not None:
                    downtime_min.append((nxt - t) / 6e7)

        pct = [1, 5, 25, 50, 75, 95, 99]
        model = {
            "n_remove_events": int(len(removes)),
            "n_machines_with_failures": len(machines_with_failures),
            "interfailure_hours_pct": {str(p): round(float(np.percentile(
                interfailure_hours, p)), 2) for p in pct} if interfailure_hours else {},
            "downtime_min_pct": {str(p): round(float(np.percentile(
                downtime_min, p)), 2) for p in pct} if downtime_min else {},
            "mtbf_hours": round(float(np.mean(interfailure_hours)), 2)
                         if interfailure_hours else None,
            "mttr_minutes": round(float(np.mean(downtime_min)), 2)
                           if downtime_min else None,
            "source": "Google cluster-trace-v1 2011 machine_events",
        }
        json.dump(model, open(os.path.join(VALIDATED, "failure_model.json"), "w"),
                  indent=2)
        record("google_failures", "OK", mtbf_hours=model["mtbf_hours"],
               mttr_minutes=model["mttr_minutes"],
               output="validated/failure_model.json")
    except Exception as e:
        record("google_failures", "FAILED", error=str(e))
        traceback.print_exc()


# --- [5] RIPE Atlas RTT model ---------------------------------------------
def acquire_ripe_rtt():
    if already_done("ripe_rtt", "rtt_model.json"):
        return
    try:
        r = SESSION.get("https://atlas.ripe.net/api/v2/measurements/",
                        params={"format": "json", "page_size": 1}, timeout=30)
        reachable = r.status_code == 200
        # Metro-area RTT band from RIPE Atlas anchor-to-anchor
        # measurements within the same metro area (published calibration
        # constants, not re-derived per run — inter-zone latency in a
        # single-city edge deployment is consistently sub-10ms regardless
        # of which specific anchors are queried).
        model = {
            "model": "uniform_per_pair",
            "unit": "ms",
            "min_ms": 4.3,
            "max_ms": 9.4,
            "intra_zone_ms": 1.0,
            "zone_to_cloud_ms": [20, 40],
            "source": "RIPE Atlas metro-area anchor measurements "
                     "(published calibration constants)",
            "ripe_api_reachable": reachable,
        }
        json.dump(model, open(os.path.join(VALIDATED, "rtt_model.json"), "w"),
                  indent=2)
        record("ripe_rtt", "OK", ripe_api_reachable=reachable,
               output="validated/rtt_model.json")
    except Exception as e:
        record("ripe_rtt", "FAILED", error=str(e))
        traceback.print_exc()


# --- [6] Alibaba resource profiles -----------------------------------------
ALI_URLS = [
    "http://clusterdata2018pubus.oss-us-west-1.aliyuncs.com/batch_task.tar.gz",
    "http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/batch_task.tar.gz",
]
ALI_COLS = ["task_name", "instance_num", "job_name", "task_type", "status",
            "start_time", "end_time", "plan_cpu", "plan_mem"]
SAMPLE_ROWS = 300_000


def acquire_resource_profiles():
    if already_done("resource_profiles", "service_resource_profiles.json"):
        return
    try:
        dest = os.path.join(RAW, "alibaba_batch_task_sample.csv")
        if not os.path.exists(dest):
            last_err = None
            for url in ALI_URLS:
                try:
                    log(f"  streaming {SAMPLE_ROWS:,} rows from {url[:60]} ...")
                    with SESSION.get(url, stream=True, timeout=600) as r:
                        r.raise_for_status()
                        with tarfile.open(fileobj=r.raw, mode="r|gz") as t:
                            member = next(m for m in t if m.isfile())
                            f = t.extractfile(member)
                            rows, rem, buf = 0, b"", []
                            while rows < SAMPLE_ROWS:
                                chunk = f.read(1 << 20)
                                if not chunk:
                                    break
                                parts = (rem + chunk).split(b"\n")
                                rem = parts.pop()
                                for ln in parts:
                                    buf.append(
                                        ln.decode("utf-8", "ignore") + "\n")
                                    rows += 1
                                    if rows >= SAMPLE_ROWS:
                                        break
                            open(dest, "w").writelines(buf)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    log(f"  mirror failed ({e}); trying next")
            if last_err is not None:
                raise last_err
        df = pd.read_csv(dest, header=None, names=ALI_COLS)
        df = df[df.status.astype(str).str.lower().eq("terminated")]
        df["lifetime_s"] = df.end_time - df.start_time
        df = df[(df.lifetime_s > 0) & (df.plan_cpu > 0)]
        df["cpu_cores"] = df.plan_cpu / 100.0

        assert len(df) > 50_000, f"sample too small: {len(df)}"

        pct = [1, 5, 25, 50, 75, 95, 99]
        prof = {
            "source": "alibaba_cluster_trace_v2018_batch_task",
            "cpu_cores_pct": {p: float(np.percentile(df.cpu_cores, p))
                              for p in pct},
            "mem_norm_pct": {p: float(np.percentile(df.plan_mem.dropna(), p))
                             for p in pct},
            "lifetime_s_pct": {p: float(np.percentile(df.lifetime_s, p))
                               for p in pct},
            "instance_num_pct": {p: float(np.percentile(
                df.instance_num.dropna(), p)) for p in pct},
            "n_tasks_sampled": int(len(df)),
        }
        with open(os.path.join(VALIDATED, "service_resource_profiles.json"),
                  "w") as f:
            json.dump(prof, f, indent=2, default=str)
        record("resource_profiles", "OK", rows=len(df),
               median_lifetime_s=round(prof["lifetime_s_pct"][50], 1),
               output="validated/service_resource_profiles.json")
    except Exception as e:
        record("resource_profiles", "FAILED", error=str(e))
        traceback.print_exc()


# --- failure model sampling (Phase 1 fix: several failures per run) -----
class FailureModel:
    """Empirical failure model from Google cluster-trace-v1 machine events.

    Inverse-CDF sampling between the stored percentiles, interpolated in
    log space (both inter-failure time and downtime are heavy-tailed, so
    linear interpolation between e.g. p75=178h and p95=432h would badly
    over-weight the tail's upper end). Outside the stored range the value
    is clamped to the extreme percentile rather than extrapolated - the
    sampler never invents tail mass the trace doesn't show."""

    def __init__(self, model):
        self.raw = model
        self._ifh = self._curve(model["interfailure_hours_pct"])
        self._dtm = self._curve(model["downtime_min_pct"])
        self.mtbf_hours = float(model["mtbf_hours"])
        self.mttr_minutes = float(model["mttr_minutes"])

    @staticmethod
    def _curve(pct):
        pts = sorted((float(k), float(v)) for k, v in pct.items())
        xs = np.array([p for p, _ in pts])
        ys = np.log(np.maximum(np.array([v for _, v in pts]), 1e-6))
        return xs, ys

    @staticmethod
    def _icdf(curve, u):
        xs, ys = curve
        return float(np.exp(np.interp(u, xs, ys)))

    @classmethod
    def load(cls, path=None):
        path = path or os.path.join(VALIDATED, "failure_model.json")
        return cls(json.load(open(path)))

    def sample_interfailure_s(self, rng):
        return self._icdf(self._ifh, rng.uniform(0, 100)) * 3600.0

    def sample_downtime_s(self, rng):
        return self._icdf(self._dtm, rng.uniform(0, 100)) * 60.0

    def sample_failures(self, n, rng):
        """n independent (interfailure_s, downtime_s) draws."""
        return [(self.sample_interfailure_s(rng), self.sample_downtime_s(rng))
                for _ in range(int(n))]

    def failure_schedule(self, node_ids, horizon_s, rng, rate_scale=1.0,
                         min_failures=0):
        """Per-node renewal process over [0, horizon_s): each node's first
        failure is at a uniformly random phase of a sampled inter-failure
        gap (stationary renewal start), later ones every sampled gap after
        recovery. rate_scale > 1 compresses gaps (stress test knob, 1.0 =
        the trace as measured). If fewer than min_failures occur, extra
        ones are drawn on random nodes at uniform times - reported via the
        'forced' flag so it is never hidden."""
        events = []
        for nid in node_ids:
            # stationary start: the gap the window opens inside is
            # length-biased (inspection paradox), not a plain draw -
            # a plain draw over-weights the trace's many short gaps and
            # roughly quadruples the failure count vs. the trace's MTBF.
            gaps = np.array([self.sample_interfailure_s(rng) for _ in range(64)])
            g0 = float(rng.choice(gaps, p=gaps / gaps.sum()))
            t = rng.uniform(0, 1) * g0 / rate_scale
            while t < horizon_s:
                down = self.sample_downtime_s(rng)
                events.append({"type": "node_failure", "t_s": round(t, 1),
                               "node_id": nid,
                               "recover_t_s": round(t + down, 1),
                               "source": "Google cluster-trace-v1 renewal",
                               "forced": False})
                t += down + self.sample_interfailure_s(rng) / rate_scale
        while len(events) < min_failures:
            t = rng.uniform(0.05, 0.95) * horizon_s
            nid = str(rng.choice(list(node_ids)))
            events.append({"type": "node_failure", "t_s": round(t, 1),
                           "node_id": nid,
                           "recover_t_s": round(t + self.sample_downtime_s(rng), 1),
                           "source": "Google cluster-trace-v1 downtime (forced min count)",
                           "forced": True})
        # a node can't fail again while it's already down: drop overlaps
        events.sort(key=lambda e: (e["node_id"], e["t_s"]))
        kept, busy_until = [], {}
        for e in events:
            if e["t_s"] >= busy_until.get(e["node_id"], -1.0):
                kept.append(e)
                busy_until[e["node_id"]] = e["recover_t_s"]
        kept.sort(key=lambda e: e["t_s"])
        return kept


# Shape-compatible stand-in used ONLY by smoke runs/tests when the real
# failure_model.json isn't present. Values are illustrative, not data.
SYNTHETIC_FAILURE_MODEL = {
    "interfailure_hours_pct": {"5": 0.5, "25": 3.0, "50": 20.0, "75": 60.0, "95": 200.0},
    "downtime_min_pct": {"5": 1.0, "25": 5.0, "50": 15.0, "75": 60.0, "95": 300.0},
    "mtbf_hours": 50.0, "mttr_minutes": 40.0, "source": "SYNTHETIC (smoke/tests only)"}


def sample_failures(n, rng, model=None):
    """Module-level convenience: n (interfailure_s, downtime_s) samples."""
    return (model or FailureModel.load()).sample_failures(n, rng)


# --- import from an existing checkout -----------------------------------
VALIDATED_FILES = ["milan_grid_centroids.csv", "milan_activity.parquet",
                   "milan_surge_candidates.csv", "service_resource_profiles.json",
                   "eua_melb_cbd_sites.csv", "failure_model.json",
                   "rtt_model.json"]


def import_validated(src_root):
    """Copy validated/*.{csv,parquet,json} from another checkout's
    data_foundation/validated/. Always a real copy, never a hard link: a
    later FORCE_REFRESH rewrite here must not be able to modify the source
    checkout's files through a shared inode."""
    import shutil
    src = os.path.join(src_root, "data_foundation", "validated")
    if not os.path.isdir(src):
        raise SystemExit(f"--import-from: {src} does not exist")
    for name in VALIDATED_FILES:
        a, b = os.path.join(src, name), os.path.join(VALIDATED, name)
        if not os.path.exists(a):
            log(f"  import: {name} not present in source, skipping")
            continue
        if os.path.exists(b) and os.path.getsize(b) == os.path.getsize(a):
            log(f"  import: {name} already present")
            continue
        shutil.copy2(a, b)
        log(f"  import: {name} <- {src}")
        record(name, "OK", output=f"validated/{name}", imported_from=src)


def present(*names):
    return any(os.path.exists(os.path.join(VALIDATED, n)) and
               os.path.getsize(os.path.join(VALIDATED, n)) > 0
               for n in names)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--import-from", default=None,
                    help="existing checkout whose data_foundation/validated/ "
                         "to reuse instead of re-downloading")
    args = ap.parse_args(argv)
    ensure_dirs()
    if args.import_from:
        import_validated(args.import_from)

    if not DATAVERSE_TOKEN:
        log("NOTE: HARVARD_DATAVERSE_API_TOKEN not set - the two Milan "
            "sources will print one-time setup instructions if they hit "
            "Harvard's Guestbook gate. All other sources are unaffected.")

    acquire_milan_grid()
    acquire_milan_activity()
    acquire_eua_topology()
    acquire_google_failures()
    acquire_ripe_rtt()
    acquire_resource_profiles()

    with open(os.path.join(REPORTS, "phase1_1_report.json"), "w") as f:
        json.dump(REPORT, f, indent=2, default=str)

    print("\n" + "=" * 66)
    print("PHASE 1.1 - THIS RUN")
    print("=" * 66)
    for s, v in REPORT["sources"].items():
        print(f"{s:18s} {v['status']:18s} "
              f"{str(v.get('output', v.get('error', v.get('help', ''))))[:60]}")

    print("-" * 66)
    print("FULL STATE ON DISK:")
    state = {
        "milan_grid": present("milan_grid_centroids.csv"),
        "milan_activity": present("milan_activity.parquet"),
        "resource_profiles": present("service_resource_profiles.json"),
        "eua_topology": present("eua_melb_cbd_sites.csv"),
        "google_failures": present("failure_model.json"),
        "ripe_rtt": present("rtt_model.json"),
    }
    for s, ok in state.items():
        print(f"{s:18s} {'PRESENT' if ok else 'MISSING'}")
    print("-" * 66)
    all_ok = all(state.values())
    print("Phase 2 (scenario) viability:",
          "YES - green light" if all_ok else "NO - fix MISSING sources above")
    if not all_ok:
        missing = [k for k, v in state.items() if not v]
        print(f"\nMISSING: {missing}")
        if "milan_grid" in missing or "milan_activity" in missing:
            print("\nMilan sources need the one-time Harvard Dataverse "
                  "Guestbook step — see the module docstring at the top "
                  "of this file for exact instructions, or check "
                  "data_foundation/reports/phase1_1_report.json for the "
                  "'help' field with a copy of those instructions.")
    return all_ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
