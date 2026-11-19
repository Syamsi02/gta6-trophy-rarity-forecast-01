"""Trophy Dilution Forecaster.

Forecasts each GTA VI PS5 trophy's Exophase earn rate at the case target
as five quantiles, from the house's recorded reads of the trophy page.

Model (per case, deterministic, stdlib only):
  * Exophase refreshes its counts in batches, not continuously.  Between
    refreshes the page is frozen; at a refresh a wave of newly tracked
    players arrives who hold far fewer trophies per player than the
    launch-night cohort (~1.03 vs ~1.48 trophies/player at the first
    refresh), so most rates fall even while existing players keep earning.
  * So rates are forecast as earned / tracked, with the two counts
    modelled separately:
      - players_tracked at the target: log-time (power-law) fit of N
        over the history, blended with a refresh-step projection
        (median jump factor / median cadence between distinct N
        plateaus).  If the history is perfectly flat (no refresh seen
        yet) a prior assumes the pending refresh: N grows roughly in
        proportion to the accumulation window since the first read.
      - per-trophy earned counts: at each refresh step the implied
        holdings of the newly tracked players, h_i = dE_i / dN
        (progression-adjusted), shrunk toward a prior that scales with
        the trophy's last rate (common early trophies are held by more
        newcomers; rare late ones by fewer).  Existing players'
        progression is estimated from within-plateau earned growth.
        If no refresh was ever seen, h_i comes from the prior alone.
      - the two-component median is blended (geometric mean) with a
        log-time trend extrapolation of the rate itself, but only when
        the history actually varies; on flat history the trend is
        persistence, which the refresh proves wrong, so it is not used.
  * Spread on the logit scale: a base term, a random-walk term growing
    with horizon, and a per-refresh jump term (practice jumps were
    ~0.3-0.5 logit units), so the 90% band is wide enough to cover
    refresh-sized moves over a multi-day horizon without paying for
    excess width at every trophy.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)


def parse_time(s):
    """Parse an ISO-8601 timestamp to epoch seconds (float)."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    txt = str(s).strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(txt)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_pct(r):
    p = clamp(r, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(l):
    if l >= 0:
        e = math.exp(-l) if l < 700 else 0.0
        p = 1.0 / (1.0 + e)
    else:
        e = math.exp(l) if l > -700 else 0.0
        p = e / (1.0 + e)
    return clamp(p * 100.0, 0.01, 99.99)


def median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def linfit(xs, ys):
    """Least squares y = a + b*x. Returns (a, b)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return my - b * mx, b


def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_min = float(case.get("horizon_minutes") or 0)
    horizon_h = horizon_min / 60.0
    horizon_d = horizon_min / 1440.0

    game = case.get("game") or {}
    release_s = parse_time(game.get("release_date"))
    if release_s is None:
        release_s = parse_time(history[0]["at"]) if history else 0.0

    # --- Last rates and earned counts -------------------------------------
    n_tr = len(trophies)
    last_rates = [None] * n_tr
    if history:
        last = history[-1]
        rates = last.get("rates") or []
        for i in range(n_tr):
            if i < len(rates) and rates[i] is not None:
                last_rates[i] = float(rates[i])
        # Fill gaps from earlier reads (newest first).
        for read in reversed(history[:-1]):
            rates = read.get("rates") or []
            for i in range(n_tr):
                if last_rates[i] is None and i < len(rates) \
                        and rates[i] is not None:
                    last_rates[i] = float(rates[i])
    for i in range(n_tr):
        if last_rates[i] is None:
            last_rates[i] = 0.0

    n_last = None
    t_last = None
    if history:
        n_last = history[-1].get("players_tracked")
        t_last = parse_time(history[-1].get("at"))
    if not n_last or n_last <= 0:
        n_last = 638.0
    n_last = float(n_last)
    t_target = parse_time(case.get("target"))
    if t_target is None:
        t_target = (t_last or 0.0) + horizon_min * 60.0
    if t_last is None:
        t_last = t_target - horizon_min * 60.0

    t_ref = release_s if release_s else t_last
    th_last = max((t_last - t_ref) / 3600.0, 0.0)
    th_target = max((t_target - t_ref) / 3600.0, th_last)
    logt_last = math.log1p(th_last)
    logt_target = math.log1p(th_target)

    # --- Build the read series --------------------------------------------
    reads = []
    for read in history:
        at = parse_time(read.get("at"))
        n = read.get("players_tracked")
        rates = read.get("rates") or []
        if at is None or not n or n <= 0:
            continue
        th = max((at - t_ref) / 3600.0, 0.0)
        vals = []
        for i in range(n_tr):
            v = None
            if i < len(rates) and rates[i] is not None:
                v = float(rates[i])
            vals.append(v)
        reads.append({"t": at, "th": th, "n": float(n), "rates": vals})
    reads.sort(key=lambda r: r["t"])

    # Distinct N plateaus (consecutive reads with equal players_tracked).
    plateaus = []
    for r in reads:
        if plateaus and plateaus[-1]["n"] == r["n"]:
            plateaus[-1]["reads"].append(r)
        else:
            plateaus.append({"n": r["n"], "reads": [r]})
    n_plateaus = len(plateaus)
    flat_history = n_plateaus <= 1

    # --- Forecast players_tracked at target -------------------------------
    if flat_history:
        # No refresh seen yet: the pending refresh covers the whole
        # accumulation window since the first read.  Launch night: the
        # first refresh multiplied tracked players ~6x over a ~4.7h
        # window (638 -> 3868).  Scale linearly with the window, capped.
        first_t = reads[0]["t"] if reads else t_last
        window_h = max((t_target - first_t) / 3600.0, 0.0)
        n_factor = 1.0 + 1.1 * window_h
        n_target = n_last * clamp(n_factor, 1.0, 8.0)
        expected_refreshes = max(1.0, window_h / 4.7)
    else:
        # Log-time (power-law) fit of N.
        xs = [math.log1p(r["th"]) for r in reads]
        ys = [math.log(r["n"]) for r in reads]
        a_n, b_n = linfit(xs, ys)
        n_fit = math.exp(a_n + b_n * logt_target) if b_n > 0 else n_last
        # Refresh-step projection from plateau jumps.
        jumps = []
        gaps = []
        for k in range(1, n_plateaus):
            prev = plateaus[k - 1]["reads"][-1]
            cur = plateaus[k]["reads"][0]
            if prev["n"] > 0 and cur["n"] > prev["n"]:
                jumps.append(cur["n"] / prev["n"])
                gaps.append((cur["t"] - prev["t"]) / 3600.0)
        n_step = None
        expected_refreshes = 0.0
        if jumps and gaps:
            j_med = median(jumps)
            gap_med = median([g for g in gaps if g > 0]) if any(
                g > 0 for g in gaps) else 0.0
            if gap_med > 0:
                steps = horizon_h / gap_med
                expected_refreshes = steps
                j_damped = 1.0 + (j_med - 1.0) * 0.6
                n_step = n_last * (j_damped ** min(steps, 40.0))
        if n_step is not None:
            n_target = math.sqrt(max(n_fit, n_last) * max(n_step, n_last))
        else:
            n_target = max(n_fit, n_last)
            expected_refreshes = horizon_h / 5.0
    n_target = clamp(n_target, n_last, n_last * 60.0)
    n_new = max(n_target - n_last, 0.0)

    # --- Per-trophy earned counts and progression -------------------------
    # Within-plateau progression: fractional earned growth per hour while
    # N is frozen (pure existing-player progression).
    prog = [0.0] * n_tr
    for i in range(n_tr):
        vals = []
        for pl in plateaus:
            rs = pl["reads"]
            if len(rs) < 2:
                continue
            first = rs[0]["rates"][i]
            last = rs[-1]["rates"][i]
            dt = (rs[-1]["t"] - rs[0]["t"]) / 3600.0
            if first is None or last is None or dt <= 0 or first <= 0:
                continue
            vals.append(((last - first) / first) / dt)
        if vals:
            prog[i] = clamp(median(vals), 0.0, 0.05)

    # Implied holdings of newly tracked players, per refresh step.
    # A newcomer holds only a fraction f of the mature rate: they have
    # barely played.  f grows with the newcomer's mean age at the
    # target, f = 1 - exp(-age / 34h).  Launch night pins the scale:
    # the first refresh (players ~2.3h old on average) left the
    # non-100% trophies at ~0.22x their rates, i.e. f ~= 0.067, while
    # the 100% trophy (earned instantly) stayed at 100%.
    if flat_history:
        first_t = reads[0]["t"] if reads else t_last
        age_new_h = max((t_target - first_t) / 3600.0, 0.0) / 2.0
    else:
        age_new_h = horizon_h / 2.0
    f_new = 1.0 - math.exp(-max(age_new_h, 0.0) / 34.0)
    h_prior = []
    for i in range(n_tr):
        r = last_rates[i]
        h_prior.append(f_new * r)  # in percent
    h_est = list(h_prior)
    if n_plateaus >= 2:
        # Weighted mean of step-implied h_i, shrunk toward the prior.
        for i in range(n_tr):
            num = 0.0
            den = 0.0
            for k in range(1, n_plateaus):
                prev = plateaus[k - 1]["reads"][-1]
                cur = plateaus[k]["reads"][0]
                dn = cur["n"] - prev["n"]
                if dn <= 0:
                    continue
                rp = prev["rates"][i]
                rc = cur["rates"][i]
                if rp is None or rc is None:
                    continue
                dt = (cur["t"] - prev["t"]) / 3600.0
                e_prev = rp * prev["n"] / 100.0
                e_cur = rc * cur["n"] / 100.0
                # Remove existing players' progression over the gap.
                e_prev_grown = e_prev * (1.0 + prog[i] * dt)
                h_step = (e_cur - e_prev_grown) / dn * 100.0
                h_step = clamp(h_step, 0.0, max(last_rates[i], 0.01))
                num += h_step * dn
                den += dn
            if den > 0:
                h_raw = num / den
                w = den / (den + 2000.0)
                h_est[i] = w * h_raw + (1.0 - w) * h_prior[i]

    # --- Median rate per trophy -------------------------------------------
    # Log-time trend of the rate itself (only meaningful if history moves).
    d_trend = [0.0] * n_tr
    if not flat_history:
        xs = [math.log1p(r["th"]) for r in reads]
        _, b_n_all = linfit(xs, [math.log(r["n"]) for r in reads])
        for i in range(n_tr):
            ex, ey = [], []
            for r in reads:
                if r["rates"][i] is not None and r["rates"][i] > 0:
                    ex.append(math.log1p(r["th"]))
                    ey.append(math.log(r["rates"][i]))
            if len(ex) >= 3:
                _, b_i = linfit(ex, ey)
                d_trend[i] = clamp(0.5 * (b_i - b_n_all), -1.5, 1.5)

    rows = []
    for i in range(n_tr):
        tid = trophies[i].get("id") if i < len(trophies) else i + 1
        r_last = last_rates[i]
        if r_last >= 99.995:
            rows.append({"id": int(tid), "q": [100.0] * 5})
            continue
        e_last = r_last * n_last / 100.0
        # Two-component forecast.
        e_prog = e_last * (1.0 + prog[i] * horizon_h)
        e_new = n_new * (h_est[i] / 100.0) * (1.0 + prog[i] * horizon_h / 2.0)
        e_target = max(e_prog + e_new, 0.0)
        r_two = clamp(100.0 * e_target / n_target, 0.01, 99.99)
        if flat_history or d_trend[i] == 0.0:
            r_med = r_two
        else:
            r_trend = r_last * math.exp(
                d_trend[i] * (logt_target - logt_last))
            r_trend = clamp(r_trend, 0.01, 99.99)
            r_med = math.sqrt(r_two * r_trend)
        r_med = clamp(r_med, 0.01, 99.99)

        # --- Spread on the logit scale ------------------------------------
        base = 0.45
        walk = 0.35 * math.sqrt(max(horizon_d, 0.0))
        jump = 0.25 * math.sqrt(min(max(expected_refreshes, 0.0), 12.0))
        sigma = math.sqrt(base * base + walk * walk + jump * jump)
        if flat_history:
            sigma *= 1.15
        # Rare trophies rest on few earned counts: a little extra width.
        sigma *= 1.0 + 0.25 / math.sqrt(max(e_last, 1.0))

        lm = logit_pct(r_med)
        qs = []
        for z in Z:
            qs.append(round(inv_logit_pct(lm + z * sigma), 4))
        # Enforce non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": int(tid), "q": qs})

    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(
        f for f in os.listdir(in_dir) if f.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(in_dir, name), "r",
                  encoding="utf-8") as fh:
            case = json.load(fh)
        out = forecast_case(case)
        text = json.dumps(out, separators=(",", ":"))
        with open(os.path.join(out_dir, case_id + ".out"), "w",
                  encoding="utf-8") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
