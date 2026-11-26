#!/usr/bin/env python3
"""
recf_elo.py - Elo ratings for RECF alliance matches (RECF Events API v2).

Pulls every scored qualification match, replays them in date order, and writes
ratings.csv. See the design notes below for how fairness across regions is handled.

Usage
  python recf_elo.py                         # fetch everything, write ./out/ratings.csv
  python recf_elo.py --no-skills-seed        # skip solo-skills seeding (on by default)
  python recf_elo.py --no-playoffs           # qualification matches only (playoffs on by default)
  python recf_elo.py --skip-surrogates       # surrogate/asterisk appearances don't update ratings
  python recf_elo.py --event-ids ids.txt     # skip discovery, use these event IDs
  python recf_elo.py --demo                  # synthetic league, no network needed

Also writes out/solo_scores.csv (worlds-style ranking of solo match scores, all events incl. solo-only).

Also adds (ratings.csv): value = an overall 0-100 whole number combining Elo, TrueSkill, CCWM and the solo match
score with fixed curves (a team's value depends only on its own stats). See the "overall value" section.

Also computed (same matches, same order):
  * TrueSkill (built in, no packages needed): mu, sigma, and a conservative rating mu - 3*sigma.
    Win/loss only; surrogate appearances count. Seeded from skills (default).
  * Average OPR / DPR / CCWM, solved per event by least squares on alliance rows, then averaged across
    events weighted by matches played. OPR: alliance score = sum of its teams' OPRs. DPR: the same
    regression on the OPPONENTS' score (lower = opponents score less against you). CCWM = OPR - DPR
    (contribution to winning margin). Qualification matches only (playoff partners are fixed pairs a
    regression can't separate); surrogate appearances count. These are relative to each event's field,
    so they ignore opponent strength across events.

Re-running is safe: finished events/divisions are read from ./cache; only missing, failed or stale
ones hit the API. A failed request is never treated as 'no matches'.

Design (each point is one simple rule)
  1. Result = win-weighted score: a win is worth >= 0.65, a shutout 1.0, tie 0.5.
  2. Alliance rating = mean of teammates; each teammate moves by its own K.
  3. K decays with experience: 40 (rookie) -> 16 (30+ matches).
  4. Surrogate ("sitting"/asterisk) appearances count like any other match (--skip-surrogates to disable).
  5. Event weight on K: scrimmage/workshop 0.5, signature/championship 1.25, playoffs x0.75.
  6. Optional skills seed: solo skills use the same scoring everywhere, so they anchor
     regions that never play each other.
  7. Connected components: ratings are only comparable inside a component; flagged in output.
  8. Ranking uses a confidence-shrunk rating so tiny samples can't top the list.
  9. Deterministic order (event date, id, schedule) and per-match change clamped to +/-32.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from statistics import NormalDist
from datetime import datetime, timezone

# ----------------------------- tunables -----------------------------------
BASE_RATING = 1500.0
K_MAX, K_MIN, K_DECAY_MATCHES = 40.0, 16.0, 30
WIN_FLOOR = 0.30          # a win scores 0.5 + 0.5*(0.30 + 0.70*margin) >= 0.65
MAX_DELTA = 32.0          # clamp on any single-match rating change
SHRINK_N = 12             # confidence = n / (n + SHRINK_N)
SEED_SCALE = 80.0         # rating points per skills z-score
SEED_CLAMP = 2.5
QUAL_ROUND = 2            # RobotEvents/VEX convention: 1 practice, 2 qual, 3 QF, 4 SF, 5 F, 6 R16
PLAYOFF_ROUNDS = {3, 4, 5, 6}
PLAYOFF_WEIGHT = 0.75
TS_MU, TS_SIGMA = 25.0, 25.0 / 3
TS_BETA, TS_TAU = TS_SIGMA / 2, TS_SIGMA / 100
TS_DRAW_P = 0.05          # TrueSkill: chance of a tie (ties are rare in Achieve)
TS_SEED_SCALE = 4.0       # TrueSkill mu shift per skills z-score (sigma0 = 8.33)
CCWM_RIDGE = 1e-4         # tiny regularisation so disconnected/sparse events still solve (matches numpy lstsq to ~0.002)
API_DEFAULT = "https://api.recf.org/api/v2"
LISTING_URL = "https://recfevents.org/Events/ApprovedEvents?programTypeId={pt}&view=past&page={page}"


# ----------------------------- regions ------------------------------------
US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
    "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky",
    "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
    "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota",
    "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "PR": "Puerto Rico"}
PROVINCES = {"AB": "Alberta", "BC": "British Columbia", "MB": "Manitoba", "NB": "New Brunswick",
             "NL": "Newfoundland and Labrador", "NS": "Nova Scotia", "NT": "Northwest Territories", "NU": "Nunavut",
             "ON": "Ontario", "PE": "Prince Edward Island", "QC": "Quebec", "SK": "Saskatchewan", "YT": "Yukon"}
_ABBR = {**US_STATES, **PROVINCES}
_CANON = {v.lower(): v for v in _ABBR.values()}
_CANON.update({k: "District of Columbia" for k in ("washington dc", "washington d.c.", "washington, dc", "washington, d.c.", "dc")})
# RECF regions made by combining states. Members are matched after abbreviations are expanded, so "NH", "VT",
# "Vermont", "NH/VT" and "New Hampshire / Vermont" all become "New Hampshire / Vermont".
# Add more with e.g.  --region-group "Massachusetts / Rhode Island"
REGION_GROUPS = [("New Hampshire", "Vermont")]


def _region_name(p):
    p = p.strip()
    if len(p) == 2 and p.upper() in _ABBR:
        return _ABBR[p.upper()]
    return _CANON.get(p.lower(), p)


def canon_region(raw):
    """One canonical label per region: expands state abbreviations, tidies stray punctuation (e.g. "CT`"),
    and folds the members of a combined region into that region's label."""
    s = re.sub(r"[`'\"\u2018\u2019\u201c\u201d]+", "", str(raw or ""))
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    names = [_region_name(p) for p in re.split(r"\s*[/&+]\s*", s) if p.strip()]
    if not names:
        return ""
    for members in REGION_GROUPS:
        if all(n in members for n in names):
            return " / ".join(members)
    return " / ".join(names)


# ----------------------------- HTTP client --------------------------------
class FetchError(Exception):
    """A request failed (after retries, or permanently). This NEVER means 'no data'."""
    def __init__(self, msg, status=None, transient=True):
        super().__init__(msg)
        self.status, self.transient = status, transient


class NotFound(FetchError):
    def __init__(self, msg="HTTP 404"):
        super().__init__(msg, 404, False)


class Abort(Exception):
    """Server kept failing even after cool-downs; stop and let the user re-run later."""


class Client:
    """
    Polite HTTP client:
      * adaptive pacing: delay doubles on 429/403 and decays after a run of successes
      * retries with exponential backoff + jitter, honours Retry-After
      * a failed request raises FetchError (never silently becomes 'empty')
      * circuit breaker: after `breaker` consecutive exhausted requests, cool down, then give up
      * on-disk cache (cache/<key>.json) so re-runs only fetch what is missing or stale
    """

    def __init__(self, base, token=None, cache_dir="cache", delay=0.6, retries=5, breaker=3,
                 cooldown=120.0, max_cooldowns=4, refresh=False):
        import requests  # imported lazily so --demo works without it
        self.requests = requests
        self.base = base.rstrip("/")
        self.base_delay = self.delay = delay
        self.retries, self.breaker = retries, breaker
        self.cooldown, self.max_cooldowns = cooldown, max_cooldowns
        self.refresh = refresh
        self.cache_dir = cache_dir
        self.last_req = 0.0
        self.ok_streak = self.fail_streak = self.cooldowns = 0
        self.stats = Counter()
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "recf-elo/1.1", "Accept": "application/json"})
        if token:
            self.s.headers["Authorization"] = f"Bearer {token}"
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    # -- pacing / health
    def _pace(self):
        wait = self.delay - (time.time() - self.last_req)
        if wait > 0:
            time.sleep(wait)
        self.last_req = time.time()

    def _ok(self, val):
        self.fail_streak = self.cooldowns = 0
        self.ok_streak += 1
        if self.ok_streak >= 15 and self.delay > self.base_delay:
            self.delay = max(self.base_delay, self.delay * 0.8)
            self.ok_streak = 0
        return val

    def cool_if_needed(self):
        if self.fail_streak < self.breaker:
            return
        if self.cooldowns >= self.max_cooldowns:
            raise Abort(f"{self.fail_streak} consecutive failures after {self.cooldowns} cool-downs")
        wait = min(self.cooldown * 2 ** self.cooldowns, 900)
        print(f"  !! {self.fail_streak} requests in a row failed (likely rate limiting). "
              f"Cooling down {wait:.0f}s ...", file=sys.stderr)
        time.sleep(wait)
        self.cooldowns += 1
        self.fail_streak = 0
        self.delay = max(self.delay, 2.0)

    # -- one request with retries
    def _request(self, url, params=None, want="json"):
        last, status = "no attempt", None
        for attempt in range(self.retries):
            self._pace()
            retry_after = 0.0
            try:
                r = self.s.get(url, params=params, timeout=30)
                status = r.status_code
            except self.requests.RequestException as e:
                self.stats["network error"] += 1
                last = f"network error: {type(e).__name__}"
            else:
                self.stats[f"HTTP {status}"] += 1
                if status == 200:
                    if want == "text":
                        return self._ok(r.text)
                    try:
                        return self._ok(r.json())
                    except ValueError:
                        self.stats["200 non-JSON"] += 1
                        last = "HTTP 200 but body was not JSON (block/challenge page?)"
                elif status == 404:
                    raise NotFound(f"HTTP 404 {url}")
                elif status in (400, 401, 422):
                    raise FetchError(f"HTTP {status}", status, transient=False)
                else:  # 403, 408, 429, 5xx ...
                    last = f"HTTP {status}"
                    try:
                        retry_after = float(r.headers.get("Retry-After", 0))
                    except ValueError:
                        retry_after = 0.0
                    if status in (429, 403):
                        self.delay = min(self.delay * 2, 20.0)
            self.ok_streak = 0
            if attempt < self.retries - 1:
                back = max(retry_after, min(2 ** attempt, 30)) * (1 + random.random() * 0.25)
                time.sleep(min(back, 120))
        self.fail_streak += 1
        raise FetchError(last, status)

    def get_text(self, url):
        return self._request(url, want="text")

    def get_obj(self, path):
        d = self._request(self.base + path)
        if isinstance(d, dict) and "data" in d and "id" not in d:
            d = d["data"]
        if not isinstance(d, dict) or "id" not in d:
            self.fail_streak += 1
            raise FetchError("unexpected response shape (no id)")
        return d

    def paged(self, path, params=None):
        """All rows of a Laravel-style paginated endpoint. Raises if ANY page fails (no partial results)."""
        rows, page, use_pp = [], 1, True
        while page <= 200:
            p = dict(params or {})
            p["page"] = page
            if use_pp:
                p["per_page"] = 250
            try:
                d = self._request(self.base + path, p)
            except FetchError as e:
                if e.status in (400, 422) and use_pp and page == 1:
                    use_pp = False  # server rejected per_page; use its default
                    continue
                raise
            data = d.get("data") if isinstance(d, dict) else None
            if not isinstance(data, list):
                self.fail_streak += 1
                raise FetchError("response had no 'data' list (error body?)")
            rows += data
            meta = d.get("meta") or {}
            if meta.get("next_page_url") or page < (meta.get("last_page") or 1):
                page += 1
            else:
                return rows
        raise FetchError("pagination did not terminate")

    # -- cache (v2 files carry a timestamp; legacy files are raw JSON and use mtime)
    def _cpath(self, key):
        return os.path.join(self.cache_dir, re.sub(r"[^A-Za-z0-9_.-]", "_", key) + ".json")

    def cache_read(self, key):
        if not self.cache_dir:
            return None
        p = self._cpath(key)
        try:
            with open(p, encoding="utf-8") as f:
                raw = json.load(f)
            mtime = os.path.getmtime(p)
        except (OSError, ValueError):
            return None
        if isinstance(raw, dict) and raw.get("v") == 2 and "data" in raw:
            return raw["data"], raw.get("fetched_at", mtime), False
        return raw, mtime, True

    def cache_write(self, key, data):
        if not self.cache_dir:
            return
        p = self._cpath(key)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"v": 2, "fetched_at": time.time(), "data": data}, f)
        os.replace(tmp, p)  # atomic: Ctrl-C can't leave a corrupt file

    def cached(self, key, fetch, ttl_for=None):
        """Return cached data if fresh, else fetch() and store it. ttl_for(data) -> hours or None (forever)."""
        if not self.refresh:
            rec = self.cache_read(key)
            if rec:
                data, ts, legacy = rec
                ttl = ttl_for(data) if ttl_for else None
                # old-version files with empty data may be rate-limit blanks: always re-verify them once
                stale = (legacy and not data) or (ttl is not None and time.time() - ts > ttl * 3600)
                if not stale:
                    self.stats["cache hit"] += 1
                    return data
        data = fetch()
        self.cache_write(key, data)
        return data


# ----------------------------- data loading -------------------------------
def _now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def parse_iso(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z", "")[:19])
    except ValueError:
        return None


def is_recent(ev, days):
    end = parse_iso(ev.get("end") or ev.get("start"))
    return bool(end) and (_now() - end).total_seconds() < days * 86400


def list_events_api(client, prefix="ACH"):
    rows = client.cached("events_list", lambda: client.paged("/events"), lambda d: 1)
    ids = []
    for e in rows:
        code = ((e.get("program") or {}).get("code") or "").upper()
        if code == prefix or str(e.get("sku", "")).upper().startswith(prefix):
            ids.append(int(e["id"]))
    return ids


def list_events_scrape(client, program_type_id, max_pages):
    ids, seen = [], set()
    for page in range(1, max_pages + 1):
        def fetch(page=page):
            html = client.get_text(LISTING_URL.format(pt=program_type_id, page=page))
            return list(dict.fromkeys(int(i) for i in re.findall(r"/Events/Details/(\d+)", html)))
        try:
            found = client.cached(f"listing_{program_type_id}_{page}", fetch, lambda d: 1)
        except FetchError as e:
            print(f"  listing page {page} failed ({e}); event list may be incomplete", file=sys.stderr)
            break
        new = [x for x in found if x not in seen]
        if not new:
            break
        seen.update(new)
        ids += new
    return ids


def norm_match(m, include_playoffs, skip_surrogates=False):
    rnd = m.get("round")
    if not m.get("scored") or rnd is None:
        return None
    playoff = rnd in PLAYOFF_ROUNDS
    if rnd != QUAL_ROUND and not (playoff and include_playoffs):
        return None
    sides = {}
    for a in m.get("alliances") or []:
        teams = []
        for at in a.get("teams") or []:
            t = at.get("team") or {}
            key = t.get("code") or (str(t["id"]) if t.get("id") is not None else None)
            if key:
                teams.append((key, t.get("name") or "", bool(at.get("sitting")) and skip_surrogates))
        if teams and a.get("color") in ("red", "blue"):
            sides[a["color"]] = (teams, int(a.get("score") or 0))
    if set(sides) != {"red", "blue"}:
        return None
    (red, rs), (blue, bs) = sides["red"], sides["blue"]
    if (rs == 0 and bs == 0) or {t[0] for t in red} & {t[0] for t in blue}:
        return None
    return {"key": (m.get("scheduled") or "9999", rnd, m.get("instance") or 0, m.get("matchnum") or 0),
            "red": red, "blue": blue, "rs": rs, "bs": bs, "playoff": playoff}


def load_event(client, eid, a):
    """-> dict(status, detail, name, event). status: ok | empty | partial | failed | notfound | future"""
    res = {"status": "failed", "detail": "", "name": "", "event": None, "matches": 0}
    def event_ttl(d):
        # divisions/dates can appear or change until the event has run, so don't trust old copies
        start = parse_iso(d.get("start"))
        if not d.get("divisions"):
            return a.recent_ttl_hours
        if start and start > _now():
            return a.empty_ttl_hours
        return a.recent_ttl_hours if is_recent(d, a.recent_days) else None

    try:
        ev = client.cached(f"event_{eid}", lambda: client.get_obj(f"/events/{eid}"), event_ttl)
    except NotFound:
        res.update(status="notfound", detail="event 404")
        return res
    except FetchError as e:
        res.update(detail=f"event: {e}")
        return res
    res["name"] = ev.get("name", "")
    start = parse_iso(ev.get("start"))
    if start and start > _now():
        res.update(status="future")
        return res
    recent = is_recent(ev, a.recent_days)
    ttl_for = lambda d: a.empty_ttl_hours if not d else (a.recent_ttl_hours if recent else None)
    matches, failed = [], []
    for d in ev.get("divisions") or []:
        try:
            rows = client.cached(f"matches_{eid}_{d['id']}",
                                 lambda d=d: client.paged(f"/events/{eid}/divisions/{d['id']}/matches"), ttl_for)
        except FetchError as e:
            failed.append(f"division {d['id']}: {e}")
            continue
        for m in rows:
            nm = norm_match(m, a.include_playoffs, a.skip_surrogates)
            if nm:
                matches.append(nm)
    res["matches"] = len(matches)
    res["detail"] = "; ".join(failed)
    if matches:
        res["event"] = {"id": eid, "name": ev.get("name", ""), "start": ev.get("start") or "",
                        "level": ev.get("level"), "event_type": ev.get("event_type"), "matches": matches}
    res["status"] = ("partial" if matches else "failed") if failed else ("ok" if matches else "empty")
    return res


def _ttl_fn(a, recent):
    return lambda d: a.empty_ttl_hours if not d else (a.recent_ttl_hours if recent else None)


def event_info(client, eid):
    """Name / date / region of an event from its cached /events/{id} record."""
    rec = client.cache_read(f"event_{eid}")
    ev = rec[0] if rec and isinstance(rec[0], dict) else {}
    loc = ev.get("location") or {}
    info = {"name": ev.get("name", ""), "start": ev.get("start") or "",
            "region": canon_region(loc.get("region") or loc.get("city") or loc.get("country") or "")}
    return info, ev


def load_skills(client, eids, a):
    """Solo skills for every given event (including solo-only events that have no alliance matches).
    -> ({eid: {team: {"driver", "programming", "name"}}}, {eid: event info})"""
    by_event, info, failed = {}, {}, []
    for eid in eids:
        client.cool_if_needed()
        info[eid], ev = event_info(client, eid)
        try:
            rows = client.cached(f"skills_{eid}", lambda eid=eid: client.paged(f"/events/{eid}/skills"),
                                 _ttl_fn(a, bool(ev) and is_recent(ev, a.recent_days)))
        except FetchError:
            failed.append(eid)
            continue
        teams = {}
        for r in rows:
            tm = r.get("team") or {}
            code, typ = tm.get("code"), r.get("type")
            if code and typ in ("driver", "programming"):
                cur = teams.setdefault(code, {"driver": 0, "programming": 0, "name": ""})
                cur[typ] = max(cur[typ], int(r.get("score") or 0))
                cur["name"] = tm.get("name") or cur["name"]
        if teams:
            by_event[eid] = teams
    if failed:
        print(f"  skills fetch failed for {len(failed)} events (re-run to retry): {failed[:10]}", file=sys.stderr)
    return by_event, info


def load_team_meta(client, eids, info, a):
    """Grade level / home region from /events/{id}/teams (falls back to /teams?event[]=).
    -> ({team: {"grade", "team_region", "organization"}}, available)"""
    meta, failed, misses, got = {}, [], 0, 0

    def fetch(eid):
        try:
            return client.paged(f"/events/{eid}/teams")
        except NotFound:
            return client.paged("/teams", {"event[]": eid})

    for eid in sorted(eids, key=lambda e: (info.get(e, {}).get("start", ""), e)):   # oldest first: newest value wins
        client.cool_if_needed()
        rec = client.cache_read(f"event_{eid}")
        recent = bool(rec) and isinstance(rec[0], dict) and is_recent(rec[0], a.recent_days)
        try:
            rows = client.cached(f"eventteams_{eid}", lambda eid=eid: fetch(eid), _ttl_fn(a, recent))
        except NotFound:
            misses += 1
            if misses >= 3 and not got:
                print("  team-list routes look unavailable on this API: grade level will be blank.", file=sys.stderr)
                return meta, False
            continue
        except FetchError:
            failed.append(eid)
            continue
        misses, got = 0, got + 1
        for tm in rows:
            key = tm.get("number") or tm.get("code") or (tm.get("team") or {}).get("code")
            if not key:
                continue
            m = meta.setdefault(str(key), {})
            if (tm.get("grade") or "").strip():
                m["grade"] = tm["grade"].strip()
            reg = canon_region((tm.get("location") or {}).get("region"))
            if reg:
                m["team_region"] = reg
            if (tm.get("organization") or "").strip():
                m["organization"] = tm["organization"].strip()
    if failed:
        print(f"  team lists failed for {len(failed)} events (re-run to retry): {failed[:10]}", file=sys.stderr)
    return meta, bool(got)


def best_skills(by_event):
    """Best driver / programming per team across all events (used for the solo-score seed)."""
    best = {}
    for teams in by_event.values():
        for code, s in teams.items():
            cur = best.setdefault(code, {"driver": 0, "programming": 0})
            cur["driver"] = max(cur["driver"], s["driver"])
            cur["programming"] = max(cur["programming"], s["programming"])
    return best


def build_solo_scores(by_event, info, meta):
    """Worlds-style ranking of solo match scores.
    score = best (driving + coding) at ONE event; coding / driving = the two parts of that score;
    best_coding / best_driving = each one's best across all events.
    Tiebreakers (RECF worlds rules), in order:
      1. highest coding score   2. highest driving score
      3. coding stop time       4. driving stop time      <- NOT in the API, so skipped
      then repeat 1-4 with each team's 2nd-highest result, then its 3rd-highest.
    The API exposes only each team's best result per event, so "2nd/3rd-highest" means the 2nd/3rd-best EVENT result
    (individual attempts within an event are not available). Missing results count as 0. Only teams tied on every
    key share a rank (1, 2, 2, 4 ...)."""
    teams = {}
    for eid, tdict in by_event.items():
        start = info.get(eid, {}).get("start", "")
        for code, s in tdict.items():
            r = teams.setdefault(code, {"name": "", "cs": [], "ds": [], "top": None, "n": 0})
            r["name"] = s["name"] or r["name"]
            r["cs"].append(s["programming"])
            r["ds"].append(s["driver"])
            r["n"] += 1
            key = (s["driver"] + s["programming"], start, eid)
            if r["top"] is None or key > r["top"][0]:
                r["top"] = (key, s["driver"], s["programming"], eid)
    rows = []
    for code, r in teams.items():
        (score, _, _), d, p, eid = r["top"]
        if score <= 0:
            continue
        cs = sorted(r["cs"], reverse=True) + [0, 0, 0]
        ds = sorted(r["ds"], reverse=True) + [0, 0, 0]
        ei, m = info.get(eid, {}), meta.get(code, {})
        rows.append({"team": code, "name": r["name"], "event_region": ei.get("region", ""),
                     "grade": m.get("grade", ""), "score": score, "coding": p, "driving": d,
                     "best_coding": cs[0], "best_driving": ds[0], "event": ei.get("name", ""), "events": r["n"],
                     "coding_2nd": cs[1], "driving_2nd": ds[1], "coding_3rd": cs[2], "driving_3rd": ds[2],
                     "_tb": (score, cs[0], ds[0], cs[1], ds[1], cs[2], ds[2])})
    rows.sort(key=lambda x: (tuple(-v for v in x["_tb"]), x["team"]))
    for i, x in enumerate(rows, 1):
        x["rank"] = rows[i - 2]["rank"] if i > 1 and x["_tb"] == rows[i - 2]["_tb"] else i
    return rows


def write_solo_scores(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "solo_scores.csv")
    cols = ["rank", "team", "name", "event_region", "grade", "score", "coding", "driving",
            "best_coding", "best_driving", "event", "events",
            "coding_2nd", "driving_2nd", "coding_3rd", "driving_3rd"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def fetch_all(a):
    client = Client(a.base, a.token, a.cache, a.delay, a.retries, a.breaker, a.cooldown, a.max_cooldowns, a.refresh)
    if a.event_ids:
        with open(a.event_ids, encoding="utf-8-sig") as f:
            ids = [int(x) for x in re.findall(r"\d+", f.read())]
    else:
        ids = []
        try:
            ids = list_events_api(client, a.program_prefix)
        except FetchError as e:
            print(f"API /events unavailable ({e}); scraping recfevents.org listing instead.", file=sys.stderr)
        if not ids:
            ids = list_events_scrape(client, a.program_type_id, a.max_pages)
    ids = list(dict.fromkeys(ids))
    print(f"{len(ids)} candidate events; cache dir: {a.cache or '(disabled)'}", file=sys.stderr)

    events, report, tally = [], {}, Counter()
    i = 0
    try:
        for i, eid in enumerate(ids, 1):
            client.cool_if_needed()
            res = load_event(client, eid, a)
            tally[res["status"]] += 1
            report[eid] = {k: res[k] for k in ("status", "name", "matches", "detail")}
            if res["event"]:
                events.append(res["event"])
            if res["status"] in ("failed", "partial", "notfound"):
                print(f"  event {eid} {res['status']}: {res['detail']}", file=sys.stderr)
            if i % 10 == 0 or i == len(ids):
                print(f"  [{i}/{len(ids)}] " + " ".join(f"{k}={v}" for k, v in sorted(tally.items())) +
                      f" | delay={client.delay:.1f}s | {dict(client.stats)}", file=sys.stderr)
    except Abort as e:
        print(f"\nSTOPPED: {e}. Progress is saved; re-run later to finish.", file=sys.stderr)
        for eid in ids[i - 1:]:
            report.setdefault(eid, {"status": "pending", "name": "", "matches": 0, "detail": "not reached"})
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved; re-run to continue.", file=sys.stderr)
        sys.exit(130)
    world = {"by_event": {}, "info": {}, "meta": {}, "teams_ok": False}
    sk_ids = [eid for eid, r in report.items() if r["status"] in ("ok", "empty", "partial")]
    print(f"Loading solo skills for {len(sk_ids)} started events (includes solo-only events) ...", file=sys.stderr)
    try:
        world["by_event"], world["info"] = load_skills(client, sk_ids, a)
        world["meta"], world["teams_ok"] = load_team_meta(client, sk_ids, world["info"], a)
        cache_awards(client, sk_ids, a)
    except Abort as e:
        print(f"\nSTOPPED during skills: {e}. Re-run later.", file=sys.stderr)
    return events, world, report, client


def print_fetch_summary(report, client):
    c = Counter(r["status"] for r in report.values())
    print("\nFetch summary: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))
    print("Requests: " + ", ".join(f"{k}={v}" for k, v in sorted(client.stats.items())))
    bad = [(e, r) for e, r in report.items() if r["status"] in ("failed", "partial", "pending")]
    if bad:
        print(f"\n{len(bad)} events incomplete (likely rate limited or blocked). Re-run the same command; "
              f"finished events/divisions come from cache and only these are retried:")
        for e, r in bad[:15]:
            print(f"  {e} {r['status']:<8} {r['name'][:40]:<40} {r['detail'][:60]}")
        if len(bad) > 15:
            print(f"  ... and {len(bad) - 15} more (see fetch_report.json)")
    print("Note: 'empty' = server answered 200 with no scored matches (e.g. awaiting results); re-checked every "
          "24h.\n")


# ----------------------------- Elo engine ---------------------------------
def event_weight(ev):
    name = (ev.get("name") or "").lower()
    if ev.get("event_type") == "workshop" or re.search(r"\b(scrimmage|workshop|practice|kickoff)\b", name):
        return 0.5
    if ev.get("level") in ("World", "National", "Regional", "State", "Signature") or \
            re.search(r"\b(signature|championship|nationals?|worlds?)\b", name):
        return 1.25
    return 1.0


def k_factor(n):
    return K_MIN + (K_MAX - K_MIN) * max(0.0, 1.0 - n / K_DECAY_MATCHES)


def build_seeds(skills):
    totals = {t: s["driver"] + s["programming"] for t, s in skills.items()}
    vals = [v for v in totals.values() if v > 0]
    if len(vals) < 5:
        return {}
    mu = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals)) or 1.0
    return {t: BASE_RATING + SEED_SCALE * max(-SEED_CLAMP, min(SEED_CLAMP, (v - mu) / sd))
            for t, v in totals.items() if v > 0}


def iter_matches(events):
    """Deterministic replay order shared by every rating system: event date, event id, schedule."""
    for ev in sorted(events, key=lambda e: (e["start"], str(e["id"]))):
        for m in sorted(ev["matches"], key=lambda x: x["key"]):
            yield ev, m


class Elo:
    def __init__(self, seeds=None):
        self.seeds = seeds or {}
        self.teams = {}
        self.parent = {}

    def team(self, key, name=""):
        t = self.teams.get(key)
        if t is None:
            r = self.seeds.get(key, BASE_RATING)
            t = self.teams[key] = {"rating": r, "seed": r, "n": 0, "w": 0, "l": 0, "t": 0,
                                   "events": set(), "name": ""}
            self.parent[key] = key
        if name:
            t["name"] = name
        return t

    def _find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def _union(self, a, b):
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self.parent[ra] = rb

    def play(self, m, weight, eid):
        red = [(self.team(k, n), k, sit) for k, n, sit in m["red"]]
        blue = [(self.team(k, n), k, sit) for k, n, sit in m["blue"]]
        rr = sum(t["rating"] for t, _, _ in red) / len(red)
        rb = sum(t["rating"] for t, _, _ in blue) / len(blue)
        exp_red = 1.0 / (1.0 + 10 ** ((rb - rr) / 400.0))
        diff = m["rs"] - m["bs"]
        sign = (diff > 0) - (diff < 0)
        margin = abs(diff) / max(m["rs"] + m["bs"], 1)
        s_red = 0.5 + 0.5 * sign * (WIN_FLOOR + (1 - WIN_FLOOR) * margin)
        w = weight * (PLAYOFF_WEIGHT if m["playoff"] else 1.0)
        for side, score, exp in ((red, s_red, exp_red), (blue, 1 - s_red, 1 - exp_red)):
            for t, _, sitting in side:
                if sitting:
                    continue
                delta = k_factor(t["n"]) * w * (score - exp)
                t["rating"] += max(-MAX_DELTA, min(MAX_DELTA, delta))
                t["n"] += 1
                t["events"].add(eid)
                if score > 0.5:
                    t["w"] += 1
                elif score < 0.5:
                    t["l"] += 1
                else:
                    t["t"] += 1
        keys = [k for _, k, _ in red + blue]
        for k in keys[1:]:
            self._union(keys[0], k)

    def run(self, events):
        for ev, m in iter_matches(events):
            self.play(m, event_weight(ev), ev["id"])

    def results(self, min_matches=6):
        sizes = {}
        for k in self.teams:
            r = self._find(k)
            sizes[r] = sizes.get(r, 0) + 1
        main = max(sizes, key=sizes.get) if sizes else None
        comp_ids = {r: i + 1 for i, (r, _) in enumerate(sorted(sizes.items(), key=lambda kv: -kv[1]))}
        rows = []
        for k, t in self.teams.items():
            conf = t["n"] / (t["n"] + SHRINK_N)
            adj = t["seed"] + conf * (t["rating"] - t["seed"])
            r = self._find(k)
            rows.append({"team": k, "name": t["name"], "rating": round(t["rating"], 1), "adj_rating": round(adj, 1),
                         "matches": t["n"], "wins": t["w"], "losses": t["l"], "ties": t["t"],
                         "events": len(t["events"]), "seed": round(t["seed"], 1),
                         "component": comp_ids[r], "main_component": r == main, "ranked": t["n"] >= min_matches})
        rows.sort(key=lambda x: (-x["ranked"], -x["adj_rating"]))
        n = 0
        for row in rows:
            n += 1 if row["ranked"] else 0
            row["rank"] = n if row["ranked"] else ""
        return rows, sizes


_N01 = NormalDist()


def _v_w(diff, margin, draw):
    """TrueSkill v and w truncation functions (Herbrich et al. 2006), same edge cases as the reference lib."""
    if draw:
        a, b = margin - abs(diff), -margin - abs(diff)
        denom = _N01.cdf(a) - _N01.cdf(b)
        v = ((_N01.pdf(b) - _N01.pdf(a)) / denom) if denom else a
        w = v * v + ((a * _N01.pdf(a) - b * _N01.pdf(b)) / denom if denom else 0.0)
        return v * (-1 if diff < 0 else 1), w
    x = diff - margin
    denom = _N01.cdf(x)
    v = (_N01.pdf(x) / denom) if denom else -x
    w = v * (v + x) if denom else (1.0 if diff < 0 else 0.0)
    return v, w


def ts_update(red, blue, outcome):
    """Two-team TrueSkill update, exact for 2 teams. red/blue: lists of [mu, sigma, update?].
    outcome: +1 red wins, -1 blue wins, 0 draw. Mutates the lists. Players with update?=False
    still contribute to their team's strength but their own rating is left alone."""
    for p in red + blue:
        if p[2]:
            p[1] = math.sqrt(p[1] ** 2 + TS_TAU ** 2)          # skill drift between matches
    n = len(red) + len(blue)
    c = math.sqrt(sum(p[1] ** 2 + TS_BETA ** 2 for p in red + blue))
    margin = _N01.inv_cdf((TS_DRAW_P + 1) / 2) * math.sqrt(n) * TS_BETA / c
    mu_r, mu_b = sum(p[0] for p in red), sum(p[0] for p in blue)
    win, lose = (blue, red) if outcome < 0 else (red, blue)
    diff = ((mu_b - mu_r) if outcome < 0 else (mu_r - mu_b)) / c
    v, w = _v_w(diff, margin, outcome == 0)
    for team, sign in ((win, 1), (lose, -1)):
        for p in team:
            if p[2]:
                var = p[1] ** 2
                p[0] += sign * (var / c) * v
                p[1] = math.sqrt(var * (1 - (var / c ** 2) * w))


def run_trueskill(events, seeds=None):
    """-> {team: {"mu","sigma","ts_rating"}}; ts_rating = mu - 3*sigma (conservative)."""
    seeds = seeds or {}
    ratings = {}

    def get(k):
        if k not in ratings:
            mu = TS_MU + TS_SEED_SCALE * (seeds[k] - BASE_RATING) / SEED_SCALE if k in seeds else TS_MU
            ratings[k] = [mu, TS_SIGMA]
        return ratings[k]

    for _, m in iter_matches(events):
        red = [[*get(k), not sit] for k, _, sit in m["red"]]
        blue = [[*get(k), not sit] for k, _, sit in m["blue"]]
        outcome = 1 if m["rs"] > m["bs"] else -1 if m["rs"] < m["bs"] else 0
        ts_update(red, blue, outcome)
        for (k, _, sit), p in zip(m["red"] + m["blue"], red + blue):
            if not sit:
                ratings[k] = [p[0], p[1]]
    return {k: {"mu": mu, "sigma": sg, "ts_rating": mu - 3 * sg} for k, (mu, sg) in ratings.items()}


def _solve(a, b):
    """Gaussian elimination with partial pivoting (small dense systems; no numpy needed)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[piv] = m[piv], m[c]
        d = m[c][c]
        if abs(d) < 1e-12:
            continue
        for r in range(c + 1, n):
            f = m[r][c] / d
            if f:
                for k in range(c, n + 1):
                    m[r][k] -= f * m[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = m[r][n] - sum(m[r][k] * x[k] for k in range(r + 1, n))
        x[r] = s / m[r][r] if abs(m[r][r]) > 1e-12 else 0.0
    return x


def run_ccwm(events):
    """Per-event OPR, DPR and CCWM by least squares on alliance rows, averaged across events (match-weighted).
    OPR: own alliance score ~ sum of teammates' OPR.  DPR: opponents' score ~ sum of teammates' DPR.
    CCWM = OPR - DPR (least squares is linear, so this equals regressing the margin directly).
    -> {team: {"opr_avg", "dpr_avg", "ccwm_avg", "ccwm_events", "ccwm_matches"}}"""
    acc = {}
    for ev in events:
        ms = [m for m in ev["matches"] if not m["playoff"]]   # qualification only (no alliance-selection bias)
        if len(ms) < 4:
            continue
        idx, played = {}, {}

        def var(k, sitting):
            key = f"{k}#S" if sitting else k   # one throwaway variable absorbs a surrogate's contribution
            if key not in idx:
                idx[key] = len(idx)
            return idx[key]

        rows = []   # one row per ALLIANCE: (member columns, own score, opponent score)
        for m in ms:
            rows.append(([var(k, s) for k, _, s in m["red"]], m["rs"], m["bs"]))
            rows.append(([var(k, s) for k, _, s in m["blue"]], m["bs"], m["rs"]))
            for k, _, s in m["red"] + m["blue"]:
                if not s:
                    played[k] = played.get(k, 0) + 1
        n = len(idx)
        ata = [[0.0] * n for _ in range(n)]
        rhs_o, rhs_d = [0.0] * n, [0.0] * n
        for cols, own, opp in rows:
            for i in cols:
                rhs_o[i] += own
                rhs_d[i] += opp
                for j in cols:
                    ata[i][j] += 1.0
        for i in range(n):
            ata[i][i] += CCWM_RIDGE
        opr, dpr = _solve(ata, rhs_o), _solve(ata, rhs_d)
        for k, cnt in played.items():
            cur = acc.setdefault(k, [0.0, 0.0, 0, 0])
            cur[0] += opr[idx[k]] * cnt
            cur[1] += dpr[idx[k]] * cnt
            cur[2] += cnt
            cur[3] += 1
    return {k: {"opr_avg": o / n, "dpr_avg": d / n, "ccwm_avg": (o - d) / n, "ccwm_events": e, "ccwm_matches": n}
            for k, (o, d, n, e) in acc.items()}


def add_extras(rows, ts, ccwm):
    """Attach TrueSkill / CCWM columns and their ranks (ranked teams only) to the Elo rows."""
    for r in rows:
        t = (ts or {}).get(r["team"])
        c = ccwm.get(r["team"])
        r["ts_mu"] = round(t["mu"], 2) if t else ""
        r["ts_sigma"] = round(t["sigma"], 2) if t else ""
        r["ts_rating"] = round(t["ts_rating"], 2) if t else ""
        r["ccwm_avg"] = round(c["ccwm_avg"], 2) if c else ""
        r["opr_avg"] = round(c["opr_avg"], 2) if c else ""
        r["dpr_avg"] = round(c["dpr_avg"], 2) if c else ""
        r["ccwm_events"] = c["ccwm_events"] if c else ""
    for col, rank_col in (("ts_rating", "ts_rank"), ("ccwm_avg", "ccwm_rank")):
        ranked = sorted((r for r in rows if r["ranked"] and r[col] != ""), key=lambda r: -r[col])
        for i, r in enumerate(ranked, 1):
            r[rank_col] = i
        for r in rows:
            r.setdefault(rank_col, "")
    return rows


def team_regions_from_matches(events, info):
    """Fallback home region: the region of the events where the team played the most matches."""
    cnt = {}
    for ev in events:
        reg = info.get(ev["id"], {}).get("region", "")
        if not reg:
            continue
        for m in ev["matches"]:
            for k, _, _ in m["red"] + m["blue"]:
                cnt.setdefault(k, Counter())[reg] += 1
    return {k: c.most_common(1)[0][0] for k, c in cnt.items()}


def add_profile(rows, meta, match_regions, ws_rows):
    """region / grade (for filtering) and the three solo match numbers: score (best driving + coding at one event),
    coding and driving (the two parts of that score)."""
    ws = {r["team"]: r for r in ws_rows}
    for r in rows:
        k = r["team"]
        m, w = meta.get(k, {}), ws.get(k)
        r["region"] = m.get("team_region") or match_regions.get(k) or (w["event_region"] if w else "")
        r["grade"] = m.get("grade", "")
        r["score"] = w["score"] if w else ""
        r["coding"] = w["coding"] if w else ""
        r["driving"] = w["driving"] if w else ""
    return rows


# ----------------------------- overall value ------------------------------
# Overall value (0-100). Each part is an ABSOLUTE 0-100 score from the team's own displayed stats using fixed curves
# (never percentiles or anything relative to other teams), so unchanged stats always give an unchanged value.
#   curve = 100 / (1 + exp(-(x - center) / scale))
# Elo and TrueSkill both measure match results (two models), so together they carry 55%.
VALUE_WEIGHTS = {"elo": 30, "ts": 25, "ccwm": 15, "solo": 30}
ELO_CENTER, ELO_SCALE = 1500.0, 87.0      # adj_rating: 1500 -> 50, 1700 -> ~91, 1300 -> ~9
TS_CENTER, TS_SCALE = 12.0, 5.0           # ts_rating (mu - 3 sigma): 0 -> ~8, 12 -> 50, 22 -> ~88
CCWM_CENTER, CCWM_SCALE = 0.0, 15.0       # ccwm_avg (points of margin): +30 -> ~88
CCWM_SHRINK_N = 12                        # CCWM pulled toward 50 when few matches: n / (n + 12)
SOLO_CENTER, SOLO_SCALE = 45.0, 20.0      # solo score (best driving + coding at one event): 76 -> ~82, 104 -> ~95


def _curve(x, center, scale):
    return 100.0 / (1.0 + math.exp(-(x - center) / scale))


def _num(v):
    return None if v in ("", None) else float(v)


def team_value(adj_rating, ts_rating, ccwm_avg, matches, solo_score):
    """Overall 0-100 whole number from the team's own displayed stats. A missing stat (None) is left out and the
    remaining weights are rescaled. -> (value, {part: 0-100 score})"""
    parts = {}
    if adj_rating is not None:
        parts["elo"] = _curve(adj_rating, ELO_CENTER, ELO_SCALE)
    if ts_rating is not None:
        parts["ts"] = _curve(ts_rating, TS_CENTER, TS_SCALE)
    if ccwm_avg is not None:
        conf = (matches or 0) / ((matches or 0) + CCWM_SHRINK_N)
        parts["ccwm"] = 50.0 + conf * (_curve(ccwm_avg, CCWM_CENTER, CCWM_SCALE) - 50.0)
    if solo_score is not None:
        parts["solo"] = _curve(solo_score, SOLO_CENTER, SOLO_SCALE)
    if not parts:
        return None, parts
    total = sum(VALUE_WEIGHTS[k] * v for k, v in parts.items()) / sum(VALUE_WEIGHTS[k] for k in parts)
    return int(max(0.0, min(100.0, total)) + 0.5), parts


def add_value(rows):
    """Add the overall value. Uses the ROUNDED numbers that appear in the CSV, so the same visible stats always give
    the same value."""
    for r in rows:
        value, parts = team_value(_num(r.get("adj_rating")), _num(r.get("ts_rating")), _num(r.get("ccwm_avg")),
                                  _num(r.get("matches")), _num(r.get("score")))
        r["value"] = value if value is not None else ""
        for k in ("elo", "ts", "ccwm", "solo"):
            r[f"value_{k}"] = int(parts[k] + 0.5) if k in parts else ""
    return rows


def cache_awards(client, eids, a):
    """Best effort: keep each started event's /awards response in the cache for later. Nothing reads it yet, it is
    silent, and anything that fails is simply retried on the next run."""
    misses = got = 0
    for eid in eids:
        client.cool_if_needed()
        rec = client.cache_read(f"event_{eid}")
        ev = rec[0] if rec and isinstance(rec[0], dict) else {}
        try:
            client.cached(f"awards_{eid}", lambda eid=eid: client.paged(f"/events/{eid}/awards"),
                          _ttl_fn(a, bool(ev) and is_recent(ev, a.recent_days)))
        except NotFound:
            misses += 1
            if misses >= 3 and not got:
                return                      # this API has no such route; stop asking
        except FetchError:
            pass
        else:
            misses, got = 0, got + 1


def write_outputs(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "ratings.csv")
    cols = ["rank", "team", "name", "region", "grade", "value", "rating", "adj_rating", "matches", "wins", "losses", "ties",
            "events", "seed", "component", "main_component",
            "ts_rank", "ts_rating", "ts_mu", "ts_sigma", "ccwm_rank", "ccwm_avg", "opr_avg", "dpr_avg", "ccwm_events",
            "score", "coding", "driving", "value_elo", "value_ts", "value_ccwm", "value_solo"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # BOM so Excel reads names correctly
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


# ----------------------------- synthetic demo -----------------------------
def _ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def make_demo(seed=7):
    rng = random.Random(seed)
    offsets = {"North": 120, "Central": 0, "South": -120}   # South never plays the others
    truth, region = {}, {}
    for ri, (rn, off) in enumerate(offsets.items()):
        for i in range(24):
            code = f"{(ri + 1) * 1000 + i}A"
            truth[code], region[code] = rng.gauss(off, 100), rn
    by_region = {rn: [c for c in truth if region[c] == rn] for rn in offsets}
    events, eid = [], 0

    def make_event(name, codes, date, rounds=7):
        nonlocal eid
        eid += 1
        matches, num = [], 0
        for rd in range(rounds):
            pool = codes[:]
            rng.shuffle(pool)
            for g in range(0, len(pool) - 3, 4):
                a, b = pool[g:g + 2], pool[g + 2:g + 4]
                sa = sum(truth[c] for c in a) / 2
                sb = sum(truth[c] for c in b) / 2
                rs = max(0, round(rng.gauss(45 + 0.12 * sa, 18)))
                bs = max(0, round(rng.gauss(45 + 0.12 * sb, 18)))
                if rs == 0 and bs == 0:
                    continue
                num += 1
                events_key = (f"{date}T10:{num:03d}:00", QUAL_ROUND, 1, num)
                matches.append({"key": events_key, "red": [(c, c, False) for c in a],
                                "blue": [(c, c, False) for c in b], "rs": rs, "bs": bs, "playoff": False})
        events.append({"id": eid, "name": name, "start": date, "level": None, "event_type": "tournament",
                       "matches": matches})

    for rn, codes in by_region.items():
        for d in range(4):
            make_event(f"{rn} Qualifier {d + 1}", rng.sample(codes, 12), f"2026-10-{5 + d * 7:02d}")
    make_event("Interregional Signature", rng.sample(by_region["North"], 8) + rng.sample(by_region["Central"], 8),
               "2026-11-15")
    skills = {}
    for c, tv in truth.items():
        if rng.random() < 0.8:
            tot = max(0, rng.gauss(50 + 0.15 * tv, 12))
            skills[c] = {"driver": tot * 0.8, "programming": tot * 0.2}
    return events, skills, truth, region


def run_demo():
    events, skills, truth, region = make_demo()
    print(f"Synthetic league: {len(truth)} teams, {len(events)} events, "
          f"{sum(len(e['matches']) for e in events)} matches (South region never plays the others)\n")
    for label, seeds in (("no skills seed", {}), ("skills seed", build_seeds(skills))):
        elo = Elo(seeds)
        elo.run(events)
        rows, sizes = elo.results(min_matches=1)
        by = {r["team"]: r for r in rows}
        codes = [c for c in truth if c in by]
        def corr(sel):
            cs = [c for c in codes if sel(c)]
            return spearman([by[c]["adj_rating"] for c in cs], [truth[c] for c in cs])
        mean = lambda rn: sum(by[c]["rating"] for c in codes if region[c] == rn) / sum(1 for c in codes if region[c] == rn)
        print(f"[{label}]  components={len(sizes)} sizes={sorted(sizes.values(), reverse=True)}")
        print(f"  Spearman vs truth  all={corr(lambda c: True):.2f}  "
              f"North+Central={corr(lambda c: region[c] != 'South'):.2f}  South only={corr(lambda c: region[c] == 'South'):.2f}")
        ts, cc = run_trueskill(events, seeds), run_ccwm(events)
        if ts:
            tc = [c for c in codes if c in ts]
            print(f"  TrueSkill Spearman all={spearman([ts[c]['ts_rating'] for c in tc], [truth[c] for c in tc]):.2f}", end="")
        cs = [c for c in codes if c in cc]
        sp = lambda key: spearman([cc[c][key] for c in cs], [truth[c] for c in cs])
        print(f"  | OPR={sp('opr_avg'):.2f} DPR={sp('dpr_avg'):.2f} CCWM={sp('ccwm_avg'):.2f}")
        print("  mean rating by region  " + "  ".join(f"{rn}={mean(rn):.0f}" for rn in ("North", "Central", "South")))
        print("  (true region offsets   North=+120  Central=0  South=-120)\n")


# ----------------------------- CLI ----------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=API_DEFAULT)
    ap.add_argument("--token", default=os.environ.get("RECF_TOKEN"), help="optional bearer token")
    ap.add_argument("--out", default="out")
    ap.add_argument("--cache", default="cache", help="cache dir ('' to disable). Safe to reuse across runs.")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache and re-fetch everything")
    ap.add_argument("--delay", type=float, default=0.6, help="base seconds between requests (adapts upward on 429)")
    ap.add_argument("--retries", type=int, default=5, help="attempts per request before it counts as failed")
    ap.add_argument("--breaker", type=int, default=3, help="consecutive failed requests that trigger a cool-down")
    ap.add_argument("--cooldown", type=float, default=120.0, help="first cool-down seconds (doubles each time)")
    ap.add_argument("--max-cooldowns", type=int, default=4, help="give up (progress saved) after this many")
    ap.add_argument("--recent-days", type=float, default=3, help="events this recent get re-fetched periodically")
    ap.add_argument("--recent-ttl-hours", type=float, default=6)
    ap.add_argument("--empty-ttl-hours", type=float, default=24, help="re-check 'no matches' results after this long")
    ap.add_argument("--max-age-hours", type=float, help=argparse.SUPPRESS)  # old flag, ignored
    ap.add_argument("--event-ids", help="text file of event IDs (skips discovery)")
    ap.add_argument("--program-prefix", default="ACH", help="program code/SKU prefix for /events filter")
    ap.add_argument("--program-type-id", type=int, default=2, help="recfevents.org programTypeId (2 = Achieve)")
    ap.add_argument("--max-pages", type=int, default=60)
    ap.add_argument("--no-playoffs", dest="include_playoffs", action="store_false",
                    help="qualification matches only (default: playoffs included, Elo weight 0.75)")
    ap.add_argument("--no-skills-seed", dest="skills_seed", action="store_false",
                    help="don't seed starting ratings from solo skills (default: seeded)")
    ap.add_argument("--skip-surrogates", action="store_true",
                    help="don't update ratings for surrogate (asterisk) appearances (default: they count)")
    ap.add_argument("--include-playoffs", dest="include_playoffs", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--skills-seed", dest="skills_seed", action="store_true", help=argparse.SUPPRESS)
    ap.set_defaults(include_playoffs=True, skills_seed=True)
    ap.add_argument("--min-matches", type=int, default=5,
                    help="matches needed to be ranked (a qualifier has at least 5 per team)")
    ap.add_argument("--region-group", action="append", default=[], metavar="'A / B'",
                    help="treat these regions as one combined region (repeatable). Built in: New Hampshire / Vermont")
    ap.add_argument("--demo", action="store_true", help="synthetic data, no network")
    a = ap.parse_args()
    for g in a.region_group:
        REGION_GROUPS.append(tuple(_region_name(x) for x in re.split(r"\s*[/&+]\s*", g) if x.strip()))
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    if a.demo:
        run_demo()
        return

    events, world, report, client = fetch_all(a)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "fetch_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print_fetch_summary(report, client)
    ws_rows = build_solo_scores(world["by_event"], world["info"], world["meta"])
    ws_path = write_solo_scores(ws_rows, a.out)
    graded = sum(1 for r in ws_rows if r["grade"])
    print(f"Solo match scores: {len(ws_rows)} teams from {len(world['by_event'])} events -> {ws_path}"
          f" ({graded} with grade level{'' if world['teams_ok'] else '; team lists unavailable'})\n")
    if not events:
        sys.exit("No events with scored matches were loaded (solo_scores.csv was still written).")
    seeds = build_seeds(best_skills(world["by_event"])) if a.skills_seed else {}
    elo = Elo(seeds)
    elo.run(events)
    rows, sizes = elo.results(a.min_matches)
    ts = run_trueskill(events, seeds)
    add_extras(rows, ts, run_ccwm(events))
    add_profile(rows, world["meta"], team_regions_from_matches(events, world["info"]), ws_rows)
    add_value(rows)
    path = write_outputs(rows, a.out)
    incomplete = sum(1 for r in report.values() if r["status"] in ("failed", "partial", "pending"))
    if incomplete:
        print(f"WARNING: ratings use {len(events)} events; {incomplete} more are incomplete. Re-run to include them.\n")
    main_size = max(sizes.values())
    print(f"{sum(len(e['matches']) for e in events)} matches from {len(events)} events, {len(rows)} teams")
    print(f"Connected components: {len(sizes)} (largest = {main_size} teams). "
          f"Ratings are only comparable within a component.")
    print(f"Wrote {path}\n\nTop 20:")
    for r in rows[:20]:
        print(f"{str(r['rank']):>4}  {r['team']:<8} {r['name'][:26]:<26} Elo {r['adj_rating']:>7}  "
              f"TS {str(r['ts_rating']):>6}  CCWM {str(r['ccwm_avg']):>6} (O {str(r['opr_avg']):>5} / D {str(r['dpr_avg']):>5})  "
              f"({r['wins']}-{r['losses']}-{r['ties']}, {r['events']} ev)")


if __name__ == "__main__":
    main()
