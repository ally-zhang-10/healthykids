"""Content Influence Index engine.

Asks BehaviorGPT: after a simulated youth sees a given piece of social media
content, how much of what they are predicted to look at next is vaping,
alcohol or cannabis products? Population-level, simulated profiles only.

Measurements (all = % of the top-100 predicted next products):
  direct        - right after one exposure to the content
  breakthrough  - in a normal mixed feed with the content shown 1-3 times
  prevention    - risky post followed by a prevention post, vs risky post alone

The API key is read from the UNBOXAI_API_KEY environment variable.
Run `python influence.py` once to precompute and cache all results.
"""
from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from behaviorgpt import ExternalItem, UnboxAIClient, User, View

CATALOG = "retail_catalog_500k"  # Unbox AI's public sample retail catalog
TOP_N = 100  # predictions read per call (API maximum)
CACHE = Path(__file__).with_name("results_cache.json")

# ---------------------------------------------------------------------------
# 1. Social media content library
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Content:
    key: str
    label: str        # shown in the dashboard
    theme: str        # grouping for the policy view
    description: str  # what BehaviorGPT reads
    categories: str

    def item(self) -> ExternalItem:
        return ExternalItem(self.description, categories=self.categories)


CONTENT = [
    # Vaping / nicotine
    Content("vape_unboxing", "Influencer unboxing a fruit-flavored vape", "Vaping & nicotine",
            "TikTok video: popular influencer unboxing a fruit flavored disposable vape, rating the flavors",
            "Social Media Video, Influencer, Vape"),
    Content("vape_tricks", "Vape cloud-tricks challenge", "Vaping & nicotine",
            "Viral video challenge: teens doing vape cloud tricks with a vape pen",
            "Social Media Video, Challenge, Vape"),
    Content("vape_ad_candy", "Sponsored ad: candy-colored vape device", "Vaping & nicotine",
            "Sponsored Instagram ad: colorful vape device with candy and mango flavors",
            "Social Media Ad, Vape, E-cigarette"),
    Content("vape_friends", "Friends vaping at a party (story)", "Vaping & nicotine",
            "Instagram story: group of friends vaping together at a house party",
            "Social Media Story, Friends, Vape"),
    Content("hookah_night", "Hookah lounge night-out reel", "Vaping & nicotine",
            "Instagram reel: night out at a hookah lounge with flavored shisha",
            "Social Media Video, Hookah, Nightlife"),
    # Alcohol
    Content("celeb_seltzer", "Celebrity promoting a hard-seltzer brand", "Alcohol",
            "Instagram post: celebrity promoting their hard seltzer brand at the beach",
            "Social Media Post, Celebrity, Alcohol"),
    Content("beer_pong", "College party beer-pong highlights", "Alcohol",
            "TikTok video: college party beer pong highlights with friends cheering",
            "Social Media Video, Party, Beer"),
    Content("shots_spring_break", "Spring-break shots video", "Alcohol",
            "Viral video: spring break crowd taking tequila shots on the beach",
            "Social Media Video, Party, Alcohol"),
    Content("music_video_drinking", "Music video with artists drinking", "Alcohol",
            "Music video: rappers drinking vodka and champagne in a club",
            "Music Video, Nightlife, Alcohol"),
    Content("drunk_meme", "Meme glamorizing getting drunk", "Alcohol",
            "Funny meme glamorizing getting drunk with friends on the weekend",
            "Social Media Meme, Humor, Alcohol"),
    # Cannabis
    Content("stoner_meme", "Stoner comedy meme", "Cannabis",
            "Funny stoner meme about rolling joints and getting high",
            "Social Media Meme, Humor, Cannabis"),
    # Stress & coping framing
    Content("vape_stress", "\"Vaping helps my anxiety\" post", "Stress & coping",
            "TikTok video: teen says vaping helps them cope with stress and anxiety",
            "Social Media Video, Mental Health, Vape"),
    Content("stress_wine", "\"Need a drink after this week\" post", "Stress & coping",
            "Instagram post: 'I need a drink after this stressful week' with a glass of wine",
            "Social Media Post, Stress, Alcohol"),
    # Prevention
    Content("psa_vape", "Anti-vaping PSA (real cost of addiction)", "Prevention",
            "Public health ad: the real cost of nicotine addiction from vaping, teens sharing how they quit",
            "Public Service Announcement, Health, Quit Vaping"),
    Content("athlete_clean", "Teen athlete on staying substance-free", "Prevention",
            "Video: teen athlete explains why they stay substance free and focus on training",
            "Social Media Video, Sports, Healthy Lifestyle"),
    Content("quit_program", "Quit-vaping support program ad", "Prevention",
            "Ad for a free text-message program that helps teens quit vaping",
            "Health Ad, Quit Vaping, Support"),
]
CONTENT_BY_KEY = {c.key: c for c in CONTENT}
RISKY = [c for c in CONTENT if c.theme != "Prevention"]
PREVENTION = [c for c in CONTENT if c.theme == "Prevention"]
THEMES = ["Vaping & nicotine", "Alcohol", "Cannabis", "Stress & coping", "Prevention"]

# Everyday posts used as the neutral baseline and as filler in a mixed feed
NEUTRAL = {
    "n_sports": ExternalItem("TikTok video: NBA basketball highlights, best dunks of the week",
                             categories="Social Media Video, Sports"),
    "n_gaming": ExternalItem("Video game livestream clip: Fortnite victory royale",
                             categories="Social Media Video, Gaming"),
    "n_pets": ExternalItem("Funny cat video compilation", categories="Social Media Video, Pets, Humor"),
    "n_music": ExternalItem("Pop music video from a chart-topping singer", categories="Music Video, Pop"),
    "n_school": ExternalItem("Study tips for final exams, back to school",
                             categories="Social Media Video, School"),
}
NEUTRAL_LABELS = {"n_sports": "Basketball highlights", "n_gaming": "Gaming livestream clip",
                  "n_pets": "Funny cat video", "n_music": "Pop music video", "n_school": "Exam study tips"}


def item_for(key: str) -> ExternalItem:
    return NEUTRAL[key] if key in NEUTRAL else CONTENT_BY_KEY[key].item()


def label_for(key: str) -> str:
    return NEUTRAL_LABELS.get(key) or CONTENT_BY_KEY[key].label


def mixed_feed(key: str, dose: int) -> tuple[str, ...]:
    """A normal feed (sports, music, ...) with `key` shown `dose` times."""
    filler = ["n_pets", "n_school", "n_gaming"]
    feed = ["n_sports", "n_music"]
    for d in range(dose):
        feed.append(key)
        if d < dose - 1:
            feed.append(filler[d % len(filler)])
    return tuple(feed)


# ---------------------------------------------------------------------------
# 2. Simulated youth profiles (no real people)
# ---------------------------------------------------------------------------

def _user(gender: str, age_group: str) -> User:
    return User(gender=gender, age_group=age_group, market="US", timezone="America/New_York")

PROFILES = {
    "Teen girl (<18)": _user("female", "<18"),
    "Teen boy (<18)": _user("male", "<18"),
    "Young woman (18-25)": _user("female", "18-25"),
    "Young man (18-25)": _user("male", "18-25"),
}

# ---------------------------------------------------------------------------
# 3. Sort predicted products into outcome buckets
# ---------------------------------------------------------------------------

def _rx(*words: str) -> re.Pattern:
    return re.compile(r"\b(" + "|".join(words) + r")", re.IGNORECASE)

# Order matters: quit aids are checked before nicotine products.
BUCKETS = [
    ("Quit aids", _rx(r"quit", r"stop smoking", r"nicorette", r"nicotine (gum|patch|lozenge)",
                      r"nicotine polacrilex", r"nicotinell", r"smoking cessation")),
    ("Vaping & nicotine", _rx(r"vap(e|es|ing|or)", r"e-?cig", r"e-?liquid", r"elf ?bar", r"geekvape", r"juul",
                              r"puff bar", r"hookah", r"shisha", r"cigar", r"tobacco", r"nicotine", r"zyn",
                              r"atomi[sz]er", r"coils?\b.*(ohm|mesh)", r"pod (kit|system|cartridge)s?",
                              r"nord\b", r"smok\b", r"voopoo", r"uwell", r"vaporesso", r"wick ?'?n'? ?vape",
                              r"dynavap", r"totally wicked")),
    ("Cannabis & smoking gear", _rx(r"rolling paper", r"pre-?rolled", r"cones?\b", r"cannabis", r"weed",
                                    r"thc\b", r"cbd\b", r"cannabidiol", r"bong", r"herb grinder", r"toker",
                                    r"stoner", r"raw classic", r"smoke odor")),
    ("Alcohol & drinking gear", _rx(r"beer", r"wine", r"vodka", r"whisk(e)?y", r"bourbon", r"scotch", r"tequila",
                                    r"rum\b", r"gin\b", r"hard seltzer", r"white claw", r"truly\b", r"smirnoff",
                                    r"liqueur", r"champagne", r"prosecco", r"cocktail", r"shot glass", r"abv\b",
                                    r"spirits\b", r"kegerator", r"brewing", r"margarita", r"twisted tea",
                                    r"high noon", r"hangover", r"bartend", r"ale\b", r"lager", r"cider")),
    ("Energy drinks", _rx(r"energy drink", r"red bull", r"monster energy", r"celsius")),
]
RISK_BUCKETS = ["Vaping & nicotine", "Cannabis & smoking gear", "Alcohol & drinking gear"]
ALL_BUCKETS = [b for b, _ in BUCKETS]


def classify(name: str | None, categories: str | None) -> str | None:
    text = f"{name or ''} | {categories or ''}"
    for bucket, pattern in BUCKETS:
        if pattern.search(text):
            return bucket
    return None


# ---------------------------------------------------------------------------
# 4. Ask BehaviorGPT (cached on disk so the demo also works offline)
# ---------------------------------------------------------------------------

_client: UnboxAIClient | None = None
_lock = threading.Lock()


def client() -> UnboxAIClient:
    global _client
    if _client is None:
        if not os.environ.get("UNBOXAI_API_KEY"):
            raise RuntimeError("UNBOXAI_API_KEY is not set")
        _client = UnboxAIClient(market="us", default_catalog_id=CATALOG)
    return _client


try:
    _cache: dict = json.loads(CACHE.read_text(encoding="utf-8"))
except Exception:
    _cache = {}


def _ck(feed: tuple[str, ...], profile: str) -> str:
    return f"{profile}|{'>'.join(feed)}"


def predict(feed: tuple[str, ...], profile: str) -> list[dict]:
    """Top-N predicted next products after a feed of content keys."""
    ck = _ck(feed, profile)
    if ck in _cache:
        return _cache[ck]
    history = [View(item_for(k)) for k in feed]
    res = client().complete(history, limit=TOP_N, user=PROFILES[profile], register_event=False)
    rows = []
    for rank, it in enumerate(res.items, 1):
        name, cats = it.data.get("name"), it.data.get("categories")
        rows.append({"rank": rank, "name": name, "categories": cats, "image": it.data.get("image_url"),
                     "bucket": classify(name, cats)})
    with _lock:
        _cache[ck] = rows
    return rows


def save_cache() -> None:
    with _lock:
        CACHE.write_text(json.dumps(_cache), encoding="utf-8")


def run_many(jobs) -> None:
    todo = list({j for j in jobs if _ck(*j) not in _cache})
    if not todo:
        return
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda j: predict(*j), todo))
    save_cache()


def shares(rows: list[dict]) -> dict[str, float]:
    """% of the top-N predictions in each bucket, plus total risk."""
    n = max(len(rows), 1)
    out = {b: 100 * sum(r["bucket"] == b for r in rows) / n for b in ALL_BUCKETS}
    out["Risk"] = sum(out[b] for b in RISK_BUCKETS)
    return out


def _avg(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


# ---------------------------------------------------------------------------
# 5. Analyses used by the dashboard
# ---------------------------------------------------------------------------

def direct_table(profiles: list[str]) -> list[dict]:
    """Influence of each content type right after one exposure."""
    run_many([((k,), p) for k in list(NEUTRAL) + list(CONTENT_BY_KEY) for p in profiles])
    baseline = _avg(shares(predict((k,), p))["Risk"] for k in NEUTRAL for p in profiles)
    rows = []
    for c in CONTENT:
        s = {p: shares(predict((c.key,), p)) for p in profiles}
        row = {"key": c.key, "Content": c.label, "Theme": c.theme,
               "Influence (pts vs neutral)": _avg(s[p]["Risk"] for p in profiles) - baseline}
        for b in ALL_BUCKETS:
            row[b] = _avg(s[p][b] for p in profiles)
        for p in profiles:
            row[p] = s[p]["Risk"] - baseline
        rows.append(row)
    return sorted(rows, key=lambda r: r["Influence (pts vs neutral)"], reverse=True)


def breakthrough_table(profiles: list[str], doses=(1, 2, 3)) -> dict[str, dict[int, float]]:
    """Risk in a normal mixed feed with the content shown 1-3 times, minus the plain feed."""
    plain = ("n_sports", "n_music", "n_pets")
    jobs = [(plain, p) for p in profiles] + [(mixed_feed(c.key, d), p) for c in CONTENT for d in doses for p in profiles]
    run_many(jobs)
    base = _avg(shares(predict(plain, p))["Risk"] for p in profiles)
    return {c.key: {d: _avg(shares(predict(mixed_feed(c.key, d), p))["Risk"] for p in profiles) - base
                    for d in doses} for c in CONTENT}


def prevention_table(profiles: list[str]) -> list[dict]:
    """Risky post then a prevention post, vs the risky post alone."""
    jobs = [((r.key,), p) for r in RISKY for p in profiles]
    jobs += [((r.key, v.key), p) for r in RISKY for v in PREVENTION for p in profiles]
    run_many(jobs)
    rows = []
    for v in PREVENTION:
        d_risk, d_quit = [], []
        for r in RISKY:
            for p in profiles:
                before, after = shares(predict((r.key,), p)), shares(predict((r.key, v.key), p))
                d_risk.append(after["Risk"] - before["Risk"])
                d_quit.append(after["Quit aids"] - before["Quit aids"])
        rows.append({"key": v.key, "Prevention message": v.label,
                     "Change in risk (pts)": _avg(d_risk), "Change in quit aids (pts)": _avg(d_quit)})
    return rows


def prevention_pairs(prevention_key: str, profiles: list[str]) -> list[dict]:
    out = []
    for r in RISKY:
        before = _avg(shares(predict((r.key,), p))["Risk"] for p in profiles)
        after = _avg(shares(predict((r.key, prevention_key), p))["Risk"] for p in profiles)
        out.append({"Risky post": r.label, "Risk alone": before, "Risk after prevention": after,
                    "Change": after - before})
    return out


def precompute_all() -> None:
    ps = list(PROFILES)
    direct_table(ps)
    breakthrough_table(ps)
    prevention_table(ps)


if __name__ == "__main__":
    import time
    t = time.time()
    precompute_all()
    ps = list(PROFILES)
    print(f"Computed/cached {len(_cache)} predictions in {time.time() - t:.0f}s\n")
    bt = breakthrough_table(ps)
    print(f"{'Content':<44}{'Direct':>8}{'Mixed x1/x2/x3':>22}")
    for r in direct_table(ps):
        b = bt[r["key"]]
        print(f"{r['Content'][:43]:<44}{r['Influence (pts vs neutral)']:>+8.1f}"
              f"{b[1]:>+8.1f}{b[2]:>+7.1f}{b[3]:>+7.1f}")
    print("\nPrevention (after a risky post):")
    for r in prevention_table(ps):
        print(f"  {r['Prevention message']:<44} risk {r['Change in risk (pts)']:+.1f}  "
              f"quit aids {r['Change in quit aids (pts)']:+.1f}")
