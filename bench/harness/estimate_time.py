#!/usr/bin/env python3
"""Projects how long a study will take on THIS machine, per model and per high-level mode.

  python3 harness/estimate_time.py --config configs/<study>.json            # after calibration
  python3 harness/estimate_time.py --config configs/<study>.json --from-reference
  python3 harness/estimate_time.py --study <study_id> --backtest            # check the method

Method (protocol section "Time"): for each model and workload, batch time at an unmeasured
level is the straight line through the two highest measured levels; the same for p95 time to
first token and median per-user speed, which decide where stop mode ends. Each level adds the
measured per-run overhead (server restart, load, warm-up). On the Gemma 4 GB10 study the
actual time was 0.64x to 1.23x of this projection one level ahead, and 0.7x to 2x from the
calibration levels, so estimates are shown as a range: projection x1 to x2.

--from-reference guesses before any calibration by scaling the GB10 Gemma reference by the
gigabytes of weights read per generated token (weights file size x active/total parameters;
run fill_digests.py first so the size is known). It is rough: use it only to decide whether to
calibrate today."""
import argparse
import math

import common
import protocol
import study_data

REFERENCE_STUDY = "gemma4_qat-q4_0_promaxgb10-2c9c_20261115"
# GB of weights read per token in the reference study: weights file x active/total params.
REFERENCE_GB_PER_TOKEN = {"e4b": 5.15, "12b": 6.98, "26b-a4b": 14.44 * 3.8 / 25.2, "31b": 17.65}
HIGH_FACTOR = 2.0
DEFAULT_OVERHEAD_S = 90


def _line(points, n):
    """Straight line through the two highest measured levels, evaluated at n."""
    pts = sorted(points.items())
    if not pts:
        return None
    if len(pts) == 1:
        (n1, y1), = pts
        return y1 * n / n1
    (n1, y1), (n2, y2) = pts[-2], pts[-1]
    return y2 + (y2 - y1) / (n2 - n1) * (n - n2)


def project(lv, model, workload, n):
    """(batch_s, ttft_p95_s, decode_median, measured?) for one level."""
    v = lv.get((model, n, workload))
    if v and v["complete"]:
        return v["batch_s"], v["ttft_p95_s"], v["decode_median"], True
    meas = {k[1]: x for k, x in lv.items() if k[0] == model and k[2] == workload and x["complete"]}
    if not meas:
        return None, None, None, False
    b = _line({k: x["batch_s"] for k, x in meas.items()}, n)
    t = _line({k: x["ttft_p95_s"] for k, x in meas.items() if x["ttft_p95_s"]}, n)
    d = _line({k: x["decode_median"] for k, x in meas.items() if x["decode_median"]}, n)
    return max(b or 0, 0.1), t, (max(d, 0.1) if d is not None else None), False


def gb_per_token(m):
    gb, tot, act = m.get("model_blob_gb"), m.get("params_b"), m.get("active_params_b")
    if not gb:
        return None
    return gb * (act / tot) if act and tot else gb


def reference_levels(cfg, models):
    """Pretend calibration data from the GB10 reference, scaled by GB read per token."""
    ref = study_data.levels(REFERENCE_STUDY)
    out = {}
    for m in models:
        g = gb_per_token(m)
        if g is None:
            print(f"  {m['key']}: weights size unknown (run fill_digests.py); using the 12B reference")
        rk = min(REFERENCE_GB_PER_TOKEN, key=lambda k: abs(REFERENCE_GB_PER_TOKEN[k] - g)) if g else "12b"
        f = g / REFERENCE_GB_PER_TOKEN[rk] if g else 1.0
        for (mk, n, w), v in ref.items():
            if mk == rk and n in protocol.CALIBRATION_LEVELS:
                out[(m["key"], n, w)] = dict(v, batch_s=v["batch_s"] * f,
                                             decode_median=(v["decode_median"] or 0) / f,
                                             ttft_p95_s=(v["ttft_p95_s"] or 0) * f)
    return out


def plan(lv, models, levels, mode, level_cap_min=None, skip_done=True):
    """Per model: list of levels with projected seconds, and where stop mode ends each workload."""
    relaxed = protocol.scenario(protocol.STOP_SCENARIO)
    out = {}
    for m in models:
        mk = m["key"]
        overheads = [v["run_overhead_s"] for k, v in lv.items()
                     if k[0] == mk and v.get("run_overhead_s") is not None]
        overhead = sum(overheads) / len(overheads) if overheads else DEFAULT_OVERHEAD_S
        stopped, rows, unknown = {}, [], False
        for n in levels:
            secs, active, all_measured = 0.0, [], True
            for w in protocol.WORKLOADS:
                if w in stopped:
                    continue
                b, t, d, measured = project(lv, mk, w, n)
                if b is None:
                    unknown = True
                    continue
                active.append(w)
                if not (measured and skip_done):
                    all_measured = False
                    s = protocol.reps_for(n) * b
                    if mode == "cap" and level_cap_min:
                        s = min(s, level_cap_min * 60 + b)
                    secs += s
                if mode == "stop":
                    fail = (t is not None and t > relaxed["p95_ttft_s_max"]) or \
                           (d is not None and d < relaxed["decode_tps_min"])
                    if fail:
                        stopped[w] = n
            if not active:
                break
            if not all_measured:
                secs += overhead
                rows.append({"users": n, "workloads": active, "seconds": secs})
        out[mk] = {"levels": rows, "stops_after": stopped, "unknown": unknown,
                   "seconds": sum(r["seconds"] for r in rows)}
    return out


def remaining(study, cfg, models, levels, mode, level_cap_min=None):
    """{'low': min, 'high': min} still to run, for the status file."""
    lv = study_data.levels(study)
    p = plan(lv, models, levels, "stop" if mode == "calibrate" else mode, level_cap_min)
    low = sum(x["seconds"] for x in p.values()) / 60
    return {"low": round(low), "high": round(low * HIGH_FACTOR)}


def fmt_range(sec):
    lo = sec / 60
    hi = lo * HIGH_FACTOR

    def f(x):
        return f"{x:.0f} min" if x < 90 else f"{x / 60:.1f} h"
    return f"{f(lo)} to {f(hi)}"


def report(cfg, lv, models, modes, source, include_measured=False):
    print(f"Time estimate for study {cfg['study']['id']} on {cfg['study'].get('machine')} "
          f"(from {source}; range = projection x1 to x{HIGH_FACTOR:g})\n")
    result = {}
    for mode in modes:
        name, cap = (mode.split(":")[0], float(mode.split(":")[1])) if ":" in mode else (mode, None)
        p = plan(lv, models, protocol.LEVELS, name, cap, skip_done=not include_measured)
        total = sum(x["seconds"] for x in p.values())
        result[mode] = {"total_s": round(total), "models": p}
        print(f"Mode {mode}: total {fmt_range(total)}")
        for mk, x in p.items():
            stops = ", ".join(f"{w} stops after {n} users" for w, n in x["stops_after"].items())
            last = x["levels"][-1]["users"] if x["levels"] else "-"
            note = " (no calibration data: run --calibrate first)" if x["unknown"] else ""
            print(f"  {mk:<22} {fmt_range(x['seconds']):<22} up to {last} users"
                  f"{'; ' + stops if stops else ''}{note}")
        print()
    return result


def backtest(study, upto=4):
    """Hide levels above `upto`, project them, compare with what was measured."""
    lv = study_data.levels(study)
    ratios = []
    for (mk, n, w), v in sorted(lv.items()):
        if n <= upto or not v["complete"]:
            continue
        cal = {k: x for k, x in lv.items() if k[0] == mk and k[1] <= upto}
        b, *_ = project(cal, mk, w, n)
        if b:
            ratios.append(v["batch_s"] / b)
            print(f"{mk:10} {w:13} {n:>3} users: projected {b:7.1f} s, actual {v['batch_s']:7.1f} s, "
                  f"actual/projected {v['batch_s'] / b:4.2f}")
    if ratios:
        print(f"\nactual/projected: min {min(ratios):.2f}, median "
              f"{sorted(ratios)[len(ratios) // 2]:.2f}, max {max(ratios):.2f} "
              f"(the range shown to members is x1 to x{HIGH_FACTOR:g})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--study", help="study id (for --backtest)")
    ap.add_argument("--modes", nargs="+", default=["stop", "all", "cap:10"])
    ap.add_argument("--from-reference", action="store_true")
    ap.add_argument("--backtest", action="store_true")
    a = ap.parse_args()
    if a.backtest:
        backtest(a.study or common.load_json(a.config)["study"]["id"])
        return
    cfg = common.load_json(a.config)
    if "study" not in cfg:
        raise SystemExit("this config has no study section (pre-protocol); use --study ... --backtest")
    models = cfg["models"]
    if a.from_reference:
        lv, source = reference_levels(cfg, models), "the GB10 Gemma reference, scaled (ROUGH)"
    else:
        lv = study_data.levels(cfg["study"]["id"])
        source = "this machine's calibration runs"
    res = report(cfg, lv, models, a.modes, source, include_measured=a.from_reference)
    if not a.from_reference:
        out = common.RESULTS / cfg["study"]["id"]
        out.mkdir(parents=True, exist_ok=True)
        common.write_json(out / "time_estimate.json", {"source": source, "high_factor": HIGH_FACTOR,
                                                       "modes": res})
        print(f"saved {out / 'time_estimate.json'}")
    if any(math.isnan(x["total_s"]) for x in res.values()):
        print("warning: some projections are undefined")


if __name__ == "__main__":
    main()
