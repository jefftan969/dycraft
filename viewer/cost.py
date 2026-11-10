"""USD cost for a trajectory's token usage.

Billing is not one token count — it is four separately-priced buckets, and the
cheap-looking one is not the cheap one. For a measured object run
(`b6_grinder`, 13 turns): cache reads were 87% of the tokens but 24% of the
cost, while output was 3% of the tokens and 45% of the cost.

Rates live in `prices.json` next to this module (override with
`$DYCRAFT_PRICES_JSON`), keyed by model id, longest matching prefix wins so a
family entry covers its variants. An unpriced model yields `None` rather than a
guess — the token counts are exact either way.

Three modifiers come off the recorded usage rather than being assumed, because
the transcript carries them per turn:

* ``cache_creation.ephemeral_5m/1h_input_tokens`` — the write tier. This is not
  cosmetic: 1 h writes cost 1.6x the 5 m rate, and every dycraft run measured so
  far writes **entirely** at the 1 h tier, so pricing cache writes at the 5 m
  rate understates a run by ~20%.
* ``inference_geo`` — ``"us"`` bills at 1.1x.
* ``service_tier`` / ``speed`` — ``"batch"`` at 0.5x, ``"fast"`` at 2x.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_DEFAULT_PRICES_PATH = Path(__file__).resolve().parent / "prices.json"
_PER_MILLION = 1_000_000.0

# usage key -> price key. A bare `cache_write` (a caller that does not split the
# write tiers) is priced at the 5 m rate.
_BUCKETS = {
    "input": "input",
    "cache_read": "cache_read",
    "cache_write_5m": "cache_write_5m",
    "cache_write_1h": "cache_write_1h",
    "output": "output",
}

_GEO_MULTIPLIER = {"us": 1.1}
_TIER_MULTIPLIER = {"batch": 0.5}
_SPEED_MULTIPLIER = {"fast": 2.0}

_cache: dict | None = None
_cache_key: tuple | None = None


def prices_path() -> Path:
    configured = os.getenv("DYCRAFT_PRICES_JSON")
    return Path(configured).expanduser() if configured else _DEFAULT_PRICES_PATH


def load_prices() -> dict:
    """The price table, reloaded when the file changes on disk."""
    global _cache, _cache_key
    path = prices_path()
    try:
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        _cache, _cache_key = {}, None
        return {}
    if _cache is not None and _cache_key == key:
        return _cache
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        raw = {}
    table = {
        str(model).lower(): rates
        for model, rates in raw.items()
        if isinstance(rates, dict) and not str(model).startswith("_")
    }
    _cache, _cache_key = table, key
    return table


def rates_for(model: str | None) -> dict | None:
    """The rate card for `model`, or None when it is not in the table."""
    if not model:
        return None
    table = load_prices()
    if not table:
        return None
    needle = model.lower()
    best: tuple[int, dict] | None = None
    for key, rates in table.items():
        if needle.startswith(key) and (best is None or len(key) > best[0]):
            best = (len(key), rates)
    return best[1] if best else None


def multiplier(*, inference_geo: str | None = None,
               service_tier: str | None = None,
               speed: str | None = None) -> float:
    """Combined rate multiplier from the per-turn billing mode.

    Applied uniformly to every bucket. Anthropic publishes the geo surcharge
    against the standard input/output rates only; whether cache rates scale too
    is unstated, so treating it as uniform is an assumption — it moves a
    US-inference run by at most 10%.
    """
    factor = 1.0
    factor *= _GEO_MULTIPLIER.get((inference_geo or "").lower(), 1.0)
    factor *= _TIER_MULTIPLIER.get((service_tier or "").lower(), 1.0)
    factor *= _SPEED_MULTIPLIER.get((speed or "").lower(), 1.0)
    return factor


def price_usage(usage: dict, model: str | None, *,
                inference_geo: str | None = None,
                service_tier: str | None = None,
                speed: str | None = None) -> dict | None:
    """Per-bucket USD for one turn, or None when `model` has no rate card.

    Returns ``{"total": float, "buckets": {bucket: usd}, "multiplier": float}``.
    """
    rates = rates_for(model)
    if rates is None:
        return None
    factor = multiplier(inference_geo=inference_geo, service_tier=service_tier,
                        speed=speed)
    buckets: dict[str, float] = {}
    for usage_key, rate_key in _BUCKETS.items():
        tokens = usage.get(usage_key) or 0
        if not tokens:
            continue
        rate = rates.get(rate_key)
        if rate is None and rate_key.startswith("cache_write"):
            rate = rates.get("cache_write")     # single-tier rate card
        try:
            rate = float(rate or 0.0)
        except (TypeError, ValueError):
            rate = 0.0
        buckets[usage_key] = tokens * rate * factor / _PER_MILLION
    # A caller that never split the write tiers still gets it priced.
    unsplit = usage.get("cache_write") or 0
    if unsplit and not (usage.get("cache_write_5m") or usage.get("cache_write_1h")):
        rate = rates.get("cache_write_5m", rates.get("cache_write", 0.0))
        buckets["cache_write"] = unsplit * float(rate or 0.0) * factor / _PER_MILLION
    return {
        "total": round(sum(buckets.values()), 6),
        "buckets": {k: round(v, 6) for k, v in buckets.items()},
        "multiplier": factor,
    }


def unpriced_models(models) -> list[str]:
    """Which of `models` have no rate card — for telling the user what is missing."""
    return sorted({m for m in models if m and rates_for(m) is None})
