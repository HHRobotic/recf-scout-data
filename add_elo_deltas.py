#!/usr/bin/env python3
"""
add_elo_deltas.py - OPTIONAL step that runs right after recf_elo.py. It does not change recf_elo.py or any rating.

recf_elo.py publishes every match but not how many Elo points each team won or lost in it. This script replays the
same matches through the same Elo engine (recf_elo.Elo), records each team's rating before and after every match,
and adds three things to the JSON files recf_elo.py already wrote in out/v1/:

    events/<id>.json   every team in every match gets   "elo_delta" and "elo_after"
    teams/<code>.json  every match gets                  "elo_delta", "elo_after", and "deltas" {team: change}
    teams/<code>.json  gets an "upcoming_events" list: current and upcoming events the team is registered for

The RECF Scouting site shows these as the Elo change next to each match and team, and draws the Elo graph from them.
If this script is not run, or is skipped by its safety check, the site still works and simply leaves those out.

Run it with the SAME options you gave recf_elo.py (the workflow uses the defaults):
    python recf_elo.py       --out out --cache cache
    python add_elo_deltas.py --out out --cache cache

Safety check: the replay must reproduce the ratings recf_elo.py just published (out/v1/teams.json). If it does not
(different options, or new matches arrived between the two runs) nothing is changed and the script exits normally.
"""
import argparse, glob, json, os, re, sys
from datetime import timedelta
from types import SimpleNamespace

import recf_elo as R


class TracingElo(R.Elo):
    """Same ratings as recf_elo.Elo. Also remembers each team's rating before and after every match."""

    def __init__(self, seeds=None):
        super().__init__(seeds)
        self.trace = {}  # (event id, match id) -> {team: (before, after)}

    def play(self, m, weight, eid):
        sides = list(m["red"]) + list(m["blue"])
        for k, n, _ in sides:
            self.team(k, n)
        before = {k: self.teams[k]["rating"] for k, _, _ in sides}
        super().play(m, weight, eid)
        if m.get("id") is not None:
            self.trace[(eid, m["id"])] = {k: (before[k], self.teams[k]["rating"]) for k, _, _ in sides}


def r1(x):
    return round(float(x), 1)


def read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_if_changed(path, obj):
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    try:
        with open(path, encoding="utf-8") as f:
            if f.read() == data:
                return False
    except OSError:
        pass
    R.write_json(path, obj)
    return True


def ratings_match(elo, v1_dir, tolerance=0.06):
    """Does the replay give the ratings recf_elo.py published? -> (ok, checked, mismatched)"""
    teams = read(os.path.join(v1_dir, "teams.json"))["teams"]
    checked = bad = 0
    for t in teams:
        k, published = t["team"], t.get("rating")
        if published is None or k not in elo.teams:
            continue
        checked += 1
        if abs(elo.teams[k]["rating"] - published) > tolerance:
            bad += 1
    return (checked > 0 and bad <= max(0, checked // 100)), checked, bad  # allow 1% slack for rounding quirks


def patch_events(v1_dir, trace):
    """Adds elo_delta / elo_after to every team in every scored match. -> {match key: {team: delta}} for the team files"""
    lookup, changed = {}, 0
    for path in sorted(glob.glob(os.path.join(v1_dir, "events", "*.json"))):
        ev = read(path)
        eid = ev["id"]
        for m in ev.get("matches", []):
            tr = trace.get((eid, m.get("id")))
            if not tr:
                continue
            deltas = {}
            for side in ("red", "blue"):
                for t in m[side]["teams"]:
                    if t["code"] in tr:
                        b, a = tr[t["code"]]
                        t["elo_delta"], t["elo_after"] = r1(a - b), r1(a)
                        deltas[t["code"]] = r1(a - b)
            codes = frozenset(deltas)
            lookup[(eid, m["round"], m["instance"], m["matchnum"], codes)] = (deltas, {c: r1(tr[c][1]) for c in deltas})
        changed += write_if_changed(path, ev)
    return lookup, changed


def patch_teams(v1_dir, lookup):
    changed = 0
    for path in sorted(glob.glob(os.path.join(v1_dir, "teams", "*.json"))):
        tj = read(path)
        code = tj["team"]
        for m in tj.get("matches", []):
            key = (m["event_id"], m["round"], m["instance"], m["matchnum"], frozenset([code, *m["partners"], *m["opponents"]]))
            hit = lookup.get(key)
            if hit and code in hit[0]:
                deltas, after = hit
                m["elo_delta"], m["elo_after"], m["deltas"] = deltas[code], after[code], deltas
        changed += write_if_changed(path, tj)
    return changed


def patch_upcoming(v1_dir, now=None):
    """Adds "upcoming_events" to every team file: events that are live or have not started, where the team is registered.
    Needs no Elo replay and no API access, only the event files recf_elo.py already wrote. -> number of team files changed"""
    now = now or R._now()
    live = []
    for path in sorted(glob.glob(os.path.join(v1_dir, "events", "*.json"))):
        ev = read(path)
        if ev.get("is_test"):
            continue
        start = R.parse_iso(ev.get("start"))
        end = R.parse_iso(ev.get("end")) or start
        if not start or (end + timedelta(days=1) < now and not ev.get("live")):
            continue  # already finished
        codes = {t["code"] for t in ev.get("teams", [])}
        codes |= {t["code"] for m in ev.get("matches", []) for side in ("red", "blue") for t in m[side]["teams"]}
        live.append((ev, codes))
    changed = 0
    for path in sorted(glob.glob(os.path.join(v1_dir, "teams", "*.json"))):
        tj = read(path)
        ups = sorted(({"event_id": ev["id"], "name": ev.get("name"), "start": ev.get("start"), "end": ev.get("end"), "city": ev.get("city"),
                       "region": ev.get("region"), "live": bool(ev.get("live"))} for ev, codes in live if tj["team"] in codes),
                     key=lambda e: (e["start"] or "", str(e["event_id"])))
        tj["upcoming_events"] = ups
        changed += write_if_changed(path, tj)
    return changed


def run_patch(elo, out_dir):
    v1 = os.path.join(out_dir, "v1")
    if not os.path.exists(os.path.join(v1, "teams.json")):
        return "no v1/teams.json found (did recf_elo.py run with JSON output?). Nothing to do.", 0, 0
    ok, checked, bad = ratings_match(elo, v1)
    if not ok:
        return f"replay does not match the published ratings ({bad} of {checked} teams differ). Nothing was changed.", 0, 0
    lookup, ce = patch_events(v1, elo.trace)
    ct = patch_teams(v1, lookup)
    return f"patched {len(lookup)} matches ({ce} event files and {ct} team files changed)", ce, ct


def default_args(a):
    """The same defaults as recf_elo.py's command line."""
    return SimpleNamespace(
        base=R.API_DEFAULT,
        token=os.environ.get("RECF_TOKEN"),
        out=a.out,
        cache=a.cache,
        refresh=False,
        delay=0.6,
        retries=5,
        breaker=3,
        cooldown=120.0,
        max_cooldowns=4,
        recent_days=3,
        recent_ttl_hours=6,
        empty_ttl_hours=24,
        event_ids=a.event_ids,
        program_prefix="ACH",
        program_type_id=2,
        max_pages=60,
        include_playoffs=a.include_playoffs,
        skills_seed=a.skills_seed,
        skip_surrogates=a.skip_surrogates,
        min_matches=5,
        upcoming_days=30,
        time_budget_min=None,
        json_out=None,
        no_json=False,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--event-ids")
    ap.add_argument("--no-playoffs", dest="include_playoffs", action="store_false")
    ap.add_argument("--no-skills-seed", dest="skills_seed", action="store_false")
    ap.add_argument("--skip-surrogates", action="store_true")
    ap.add_argument("--region-group", action="append", default=[])
    a = ap.parse_args()
    for g in a.region_group:
        R.REGION_GROUPS.append(tuple(R._region_name(x) for x in re.split(r"\s*[/&+]\s*", g) if x.strip()))
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    v1 = os.path.join(a.out, "v1")
    if os.path.isdir(os.path.join(v1, "events")):
        print(f"add_elo_deltas: upcoming events added to {patch_upcoming(v1)} team files", file=sys.stderr)
    events, world, _report, _client = R.fetch_all(default_args(a))
    if not events:
        print("add_elo_deltas: no events with scored matches; nothing to do.", file=sys.stderr)
        return
    seeds = R.build_seeds(R.best_skills(world["by_event"])) if a.skills_seed else {}
    elo = TracingElo(seeds)
    elo.run(events)
    msg, _, _ = run_patch(elo, a.out)
    print(f"add_elo_deltas: {msg}", file=sys.stderr)


if __name__ == "__main__":
    main()
