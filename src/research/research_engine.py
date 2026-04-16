"""Free research layer that gathers world events affecting trading.

Runs BEFORE the quant signal engine and LLM call. Produces a structured
research briefing from 100% free sources — no paid APIs.

Free data sources used:
1. RSS feeds — CoinDesk, CoinTelegraph, Decrypt, The Block, Google News
2. CoinGecko — trending coins, global market data, price changes (free tier)
3. Fear & Greed Index — alternative.me (free)
4. Reddit — public .json endpoints (no API key)
5. DeFi Llama — TVL data, stablecoin flows (free)
6. Hyperliquid on-chain — funding rate spikes, OI changes (already available)

Cross-validation strategy:
- Events must appear in 2+ independent sources to be flagged high-impact
- On-chain data (funding/OI) must align with news sentiment
- Keyword-based sentiment as primary (no LLM dependency)
- Optional Ollama for deeper analysis if available locally
"""

from __future__ import annotations
import asyncio
import json
import logging
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional
from xml.etree import ElementTree

import aiohttp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NewsItem:
    title: str
    source: str
    url: str = ""
    published: Optional[datetime] = None
    summary: str = ""
    credibility: float = 0.5
    is_fresh: bool = False  # Within last 24h


@dataclass
class ResearchBriefing:
    """Structured output consumed by the quant engine and LLM."""
    timestamp: str
    market_sentiment: dict          # score, label, fear_greed
    key_events: list[dict]          # Cross-validated events
    asset_signals: dict[str, dict]  # Per-asset sentiment + catalysts
    macro_context: dict             # Global crypto market, stablecoin flows
    risk_alerts: list[str]          # High-priority warnings
    data_quality: dict              # Freshness, source count, confidence
    on_chain_context: dict          # Funding/OI anomalies from Hyperliquid


# ---------------------------------------------------------------------------
# Source credibility weights
# ---------------------------------------------------------------------------

SOURCE_CREDIBILITY = {
    "coindesk": 0.95,
    "the_block": 0.90,
    "cointelegraph": 0.85,
    "decrypt": 0.85,
    "bitcoin_magazine": 0.80,
    "google_news": 0.75,
    "newsbtc": 0.60,
    "cryptopotato": 0.55,
    "coingecko_trending": 0.70,
    "reddit_cryptocurrency": 0.45,
    "reddit_bitcoin": 0.40,
    "reddit_other": 0.35,
}

# ---------------------------------------------------------------------------
# Keyword sentiment lexicon (no LLM needed)
# ---------------------------------------------------------------------------

BULLISH_KEYWORDS = {
    # Strong bullish
    "etf approved": 3, "etf approval": 3, "institutional adoption": 3,
    "all-time high": 3, "ath": 2, "record high": 3,
    "partnership": 2, "integration": 2, "mainstream adoption": 3,
    "accumulation": 2, "whale buying": 2, "whale accumulation": 2,

    # Moderate bullish
    "rally": 1.5, "surge": 1.5, "breakout": 1.5, "bull run": 2,
    "buy signal": 1.5, "bullish": 1.5, "moon": 1, "pump": 1,
    "upgrade": 1.5, "launch": 1, "new listing": 1.5,
    "inflow": 1.5, "net inflows": 2, "stablecoin inflow": 2,
    "halving": 1.5, "supply shock": 2,
    "rate cut": 2, "dovish": 1.5, "fed pivot": 2,
}

BEARISH_KEYWORDS = {
    # Strong bearish
    "hack": -3, "exploit": -3, "rug pull": -3, "ponzi": -3,
    "sec lawsuit": -3, "regulatory crackdown": -3, "ban": -2.5,
    "insolvency": -3, "bankruptcy": -3, "fraud": -3,

    # Moderate bearish
    "crash": -2, "dump": -2, "sell-off": -2, "liquidation": -2,
    "bear market": -2, "bearish": -1.5, "fud": -1,
    "investigation": -1.5, "subpoena": -2, "enforcement": -2,
    "outflow": -1.5, "net outflows": -2, "stablecoin outflow": -2,
    "rate hike": -2, "hawkish": -1.5, "recession": -2,
    "delist": -2, "vulnerability": -2, "security breach": -2.5,
}

ASSET_ALIASES = {
    "BTC": ["bitcoin", "btc", "₿"],
    "ETH": ["ethereum", "eth", "ether"],
    "SOL": ["solana", "sol"],
    "AVAX": ["avalanche", "avax"],
    "DOGE": ["dogecoin", "doge"],
    "LINK": ["chainlink", "link"],
    "DOT": ["polkadot", "dot"],
    "MATIC": ["polygon", "matic"],
    "ARB": ["arbitrum", "arb"],
    "OP": ["optimism"],
}


# ---------------------------------------------------------------------------
# HTTP session helper
# ---------------------------------------------------------------------------

async def _fetch(session: aiohttp.ClientSession, url: str, timeout: int = 12) -> Optional[str]:
    """Fetch URL with timeout and error handling."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status == 200:
                return await resp.text()
    except Exception as e:
        logger.debug("Fetch failed for %s: %s", url, e)
    return None


async def _fetch_json(session: aiohttp.ClientSession, url: str, timeout: int = 12) -> Optional[dict | list]:
    """Fetch and parse JSON."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status == 200:
                return await resp.json(content_type=None)
    except Exception as e:
        logger.debug("JSON fetch failed for %s: %s", url, e)
    return None


# ---------------------------------------------------------------------------
# RSS parser (no feedparser dependency needed)
# ---------------------------------------------------------------------------

def _parse_rss(xml_text: str, source_name: str) -> list[NewsItem]:
    """Parse RSS/Atom XML into NewsItem list without feedparser."""
    items = []
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return items

    # RSS 2.0 format
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date_str = item.findtext("pubDate") or item.findtext("published") or ""
        summary = (item.findtext("description") or "")[:300].strip()
        # Strip HTML from summary
        summary = re.sub(r"<[^>]+>", "", summary).strip()

        pub_date = _parse_date(pub_date_str)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        is_fresh = pub_date >= cutoff if pub_date else False

        if title:
            items.append(NewsItem(
                title=title,
                source=source_name,
                url=link,
                published=pub_date,
                summary=summary,
                credibility=SOURCE_CREDIBILITY.get(source_name, 0.5),
                is_fresh=is_fresh,
            ))

    # Atom format
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
        title = (entry.findtext("{http://www.w3.org/2005/Atom}title") or "").strip()
        link_el = entry.find("{http://www.w3.org/2005/Atom}link")
        link = link_el.get("href", "") if link_el is not None else ""
        updated = entry.findtext("{http://www.w3.org/2005/Atom}updated") or ""
        summary_el = entry.findtext("{http://www.w3.org/2005/Atom}summary") or ""
        summary = re.sub(r"<[^>]+>", "", summary_el)[:300].strip()

        pub_date = _parse_date(updated)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        is_fresh = pub_date >= cutoff if pub_date else False

        if title:
            items.append(NewsItem(
                title=title,
                source=source_name,
                url=link,
                published=pub_date,
                summary=summary,
                credibility=SOURCE_CREDIBILITY.get(source_name, 0.5),
                is_fresh=is_fresh,
            ))

    return items


def _parse_date(date_str: str) -> Optional[datetime]:
    """Parse common date formats from RSS feeds."""
    if not date_str or date_str.strip() == "N/A":
        return None
    date_str = date_str.strip()
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%d %b %Y %H:%M:%S %z",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Individual source fetchers
# ---------------------------------------------------------------------------

async def fetch_rss_news(session: aiohttp.ClientSession) -> list[NewsItem]:
    """Fetch news from multiple RSS feeds in parallel."""
    feeds = {
        "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "cointelegraph": "https://cointelegraph.com/rss",
        "decrypt": "https://decrypt.co/feed",
        "bitcoin_magazine": "https://bitcoinmagazine.com/feed",
        "the_block": "https://www.theblock.co/rss.xml",
        "newsbtc": "https://www.newsbtc.com/feed/",
        "cryptopotato": "https://cryptopotato.com/feed/",
    }

    tasks = {name: _fetch(session, url) for name, url in feeds.items()}
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    all_items = []
    for name, result in zip(tasks.keys(), results):
        if isinstance(result, str) and result:
            items = _parse_rss(result, name)
            all_items.extend(items)
            logger.debug("RSS %s: %d items", name, len(items))
        else:
            logger.debug("RSS %s: failed", name)

    return all_items


async def fetch_google_news(session: aiohttp.ClientSession, query: str = "crypto+market") -> list[NewsItem]:
    """Fetch Google News RSS for broad market context."""
    url = f"https://news.google.com/rss/search?q={query}+when:1d&hl=en-US&gl=US&ceid=US:en"
    xml = await _fetch(session, url)
    if xml:
        return _parse_rss(xml, "google_news")
    return []


async def fetch_reddit(session: aiohttp.ClientSession, assets: list[str]) -> list[NewsItem]:
    """Fetch Reddit posts from crypto subreddits (public .json endpoints, no API key)."""
    subreddits = [
        ("CryptoCurrency", "reddit_cryptocurrency"),
        ("Bitcoin", "reddit_bitcoin"),
        ("ethtrader", "reddit_other"),
        ("CryptoMarkets", "reddit_other"),
        ("solana", "reddit_other"),
    ]

    all_items = []
    asset_lower = {a.lower() for a in assets}
    asset_aliases_flat = {}
    for sym, aliases in ASSET_ALIASES.items():
        if sym in assets:
            for alias in aliases:
                asset_aliases_flat[alias] = sym

    for subreddit, source_name in subreddits:
        url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=25"
        data = await _fetch_json(session, url)
        if not data or "data" not in data:
            continue

        for post in data["data"].get("children", []):
            pd = post.get("data", {})
            title = pd.get("title", "")
            score = pd.get("ups", 0) + pd.get("num_comments", 0) * 2

            # Only include posts with decent engagement
            if score < 15:
                continue

            # Check relevance to our assets
            title_lower = title.lower()
            relevant = any(a in title_lower for a in asset_lower)
            if not relevant:
                relevant = any(alias in title_lower for alias in asset_aliases_flat)

            if relevant:
                created = pd.get("created_utc")
                pub_date = datetime.fromtimestamp(created, tz=timezone.utc) if created else None
                cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
                is_fresh = pub_date >= cutoff if pub_date else False

                all_items.append(NewsItem(
                    title=title,
                    source=source_name,
                    url=f"https://reddit.com{pd.get('permalink', '')}",
                    published=pub_date,
                    summary=f"Score: {score}, Comments: {pd.get('num_comments', 0)}",
                    credibility=SOURCE_CREDIBILITY.get(source_name, 0.35),
                    is_fresh=is_fresh,
                ))

    return all_items


async def fetch_fear_greed(session: aiohttp.ClientSession) -> dict:
    """Fetch Crypto Fear & Greed Index (alternative.me, free)."""
    data = await _fetch_json(session, "https://api.alternative.me/fng/?limit=7")
    if data and "data" in data:
        current = data["data"][0]
        # Also get 7-day trend
        values = [int(d["value"]) for d in data["data"]]
        trend = "rising" if values[0] > values[-1] else "falling" if values[0] < values[-1] else "stable"
        return {
            "value": int(current["value"]),
            "label": current.get("value_classification", "Neutral"),
            "trend_7d": trend,
            "history": values,
        }
    return {"value": 50, "label": "Neutral", "trend_7d": "stable", "history": [50]}


async def fetch_coingecko_global(session: aiohttp.ClientSession) -> dict:
    """Fetch global crypto market data from CoinGecko (free, no key)."""
    data = await _fetch_json(session, "https://api.coingecko.com/api/v3/global")
    if data and "data" in data:
        g = data["data"]
        return {
            "total_market_cap_usd": g.get("total_market_cap", {}).get("usd", 0),
            "total_volume_24h_usd": g.get("total_volume", {}).get("usd", 0),
            "btc_dominance": round(g.get("market_cap_percentage", {}).get("btc", 0), 2),
            "eth_dominance": round(g.get("market_cap_percentage", {}).get("eth", 0), 2),
            "market_cap_change_24h_pct": round(g.get("market_cap_change_percentage_24h_usd", 0), 2),
            "active_cryptocurrencies": g.get("active_cryptocurrencies", 0),
        }
    return {}


async def fetch_coingecko_trending(session: aiohttp.ClientSession) -> list[dict]:
    """Fetch trending coins from CoinGecko (free)."""
    data = await _fetch_json(session, "https://api.coingecko.com/api/v3/search/trending")
    if data and "coins" in data:
        return [
            {
                "name": c["item"]["name"],
                "symbol": c["item"]["symbol"].upper(),
                "market_cap_rank": c["item"].get("market_cap_rank"),
                "score": c["item"].get("score", 0),
            }
            for c in data["coins"][:10]
        ]
    return []


async def fetch_defi_llama_stablecoins(session: aiohttp.ClientSession) -> dict:
    """Fetch stablecoin market cap changes from DeFi Llama (free).

    Stablecoin flows are a leading indicator:
    - Rising stablecoin supply = new money entering crypto (bullish)
    - Falling stablecoin supply = money leaving crypto (bearish)
    """
    data = await _fetch_json(session, "https://stablecoins.llama.fi/stablecoins?includePrices=false")
    if data and "peggedAssets" in data:
        total_mcap = 0
        top_stables = []
        for s in data["peggedAssets"][:5]:  # Top 5 stablecoins
            mcap = s.get("circulating", {}).get("peggedUSD", 0) or 0
            total_mcap += mcap
            top_stables.append({
                "name": s.get("name", ""),
                "symbol": s.get("symbol", ""),
                "mcap_usd": round(mcap, 0),
            })
        return {
            "total_stablecoin_mcap": round(total_mcap, 0),
            "top_stablecoins": top_stables,
        }
    return {}


# ---------------------------------------------------------------------------
# Keyword-based sentiment engine (no LLM needed)
# ---------------------------------------------------------------------------

def compute_keyword_sentiment(items: list[NewsItem], asset: str) -> dict:
    """Compute sentiment purely from keyword analysis.

    This is the PRIMARY sentiment method — no external LLM dependency.
    Uses credibility-weighted scoring.
    """
    if not items:
        return {"score": 0, "confidence": 0, "label": "neutral", "keyword_hits": []}

    # Build asset-specific search terms
    search_terms = [asset.lower()]
    if asset in ASSET_ALIASES:
        search_terms.extend(ASSET_ALIASES[asset])

    total_score = 0
    total_weight = 0
    keyword_hits = []
    relevant_count = 0

    for item in items:
        text = (item.title + " " + item.summary).lower()

        # Check if this item is relevant to our asset
        is_relevant = any(term in text for term in search_terms)
        # Also count general crypto/market items
        is_general = any(w in text for w in ["crypto", "market", "bitcoin", "defi", "trading"])

        if not is_relevant and not is_general:
            continue

        relevance_mult = 1.0 if is_relevant else 0.3  # General items get less weight
        freshness_mult = 1.5 if item.is_fresh else 0.5  # Fresh items matter more

        item_score = 0
        for keyword, weight in BULLISH_KEYWORDS.items():
            if keyword in text:
                item_score += weight
                keyword_hits.append({"keyword": keyword, "source": item.source, "direction": "bullish"})

        for keyword, weight in BEARISH_KEYWORDS.items():
            if keyword in text:
                item_score += weight  # weight is already negative
                keyword_hits.append({"keyword": keyword, "source": item.source, "direction": "bearish"})

        # Weight by credibility * relevance * freshness
        item_weight = item.credibility * relevance_mult * freshness_mult
        total_score += item_score * item_weight
        total_weight += item_weight

        if is_relevant:
            relevant_count += 1

    if total_weight == 0:
        return {"score": 0, "confidence": 0, "label": "neutral", "keyword_hits": []}

    # Normalize score to -1..+1 range
    raw_score = total_score / total_weight
    normalized_score = math.tanh(raw_score / 3)  # Scale so +-3 raw maps to ~+-0.95

    # Confidence based on data volume and diversity
    sources = set(h["source"] for h in keyword_hits)
    volume_confidence = min(len(keyword_hits) / 10, 1.0)
    diversity_confidence = min(len(sources) / 4, 1.0)
    confidence = (volume_confidence * 0.5 + diversity_confidence * 0.5) * min(relevant_count / 3, 1.0)

    # Label
    if normalized_score > 0.3:
        label = "bullish"
    elif normalized_score > 0.1:
        label = "slightly_bullish"
    elif normalized_score < -0.3:
        label = "bearish"
    elif normalized_score < -0.1:
        label = "slightly_bearish"
    else:
        label = "neutral"

    return {
        "score": round(normalized_score, 4),
        "confidence": round(confidence, 4),
        "label": label,
        "keyword_hits": keyword_hits[:10],  # Top 10 for context
        "relevant_articles": relevant_count,
    }


# ---------------------------------------------------------------------------
# Cross-validation engine
# ---------------------------------------------------------------------------

def cross_validate_events(items: list[NewsItem], min_sources: int = 2) -> list[dict]:
    """Identify events confirmed by multiple independent sources.

    From IMC Prosperity: only act on signals confirmed by multiple data points.
    An event must appear in 2+ independent sources to be flagged as high-impact.
    """
    # Extract key phrases from titles
    event_clusters: dict[str, list[NewsItem]] = defaultdict(list)

    for item in items:
        if not item.is_fresh:
            continue

        title_lower = item.title.lower()

        # Extract significant n-grams (3-5 words)
        words = re.findall(r'\b[a-z]+\b', title_lower)
        for n in (4, 3):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i + n])
                # Skip generic phrases
                if any(skip in ngram for skip in ["the market", "this week", "price analysis", "what you need"]):
                    continue
                event_clusters[ngram].append(item)

    # Filter to events with multiple independent sources
    validated_events = []
    seen_items = set()

    for phrase, cluster_items in sorted(event_clusters.items(), key=lambda x: len(x[1]), reverse=True):
        sources = set(item.source for item in cluster_items)
        if len(sources) >= min_sources:
            # Deduplicate
            key = frozenset(sources)
            if key in seen_items:
                continue
            seen_items.add(key)

            avg_cred = sum(item.credibility for item in cluster_items) / len(cluster_items)
            validated_events.append({
                "phrase": phrase,
                "sources": list(sources),
                "source_count": len(sources),
                "credibility": round(avg_cred, 2),
                "sample_title": cluster_items[0].title,
                "impact": "high" if len(sources) >= 3 else "medium",
            })

    return validated_events[:10]  # Top 10 events


# ---------------------------------------------------------------------------
# On-chain signal analysis (from Hyperliquid data)
# ---------------------------------------------------------------------------

def analyze_onchain_signals(
    funding_rates: dict[str, float],
    open_interests: dict[str, float],
    prev_funding_rates: dict[str, float] | None = None,
    prev_open_interests: dict[str, float] | None = None,
) -> dict:
    """Analyze Hyperliquid on-chain data for sentiment signals.

    Key signals:
    - Funding rate spike: extreme positive = overleveraged longs (bearish contrarian)
    - Funding rate deeply negative: overleveraged shorts (bullish contrarian)
    - OI surge + funding spike: potential liquidation cascade risk
    - OI dropping: deleveraging (may indicate bottom forming)
    """
    signals = {}

    for asset in funding_rates:
        rate = funding_rates[asset]
        oi = open_interests.get(asset, 0)

        annualized = rate * 24 * 365 * 100 if rate else 0

        # Funding rate analysis
        if annualized > 50:
            funding_signal = "extreme_positive"
            funding_bias = "bearish_contrarian"  # Too many longs
        elif annualized > 20:
            funding_signal = "elevated_positive"
            funding_bias = "slightly_bearish"
        elif annualized < -50:
            funding_signal = "extreme_negative"
            funding_bias = "bullish_contrarian"  # Too many shorts
        elif annualized < -20:
            funding_signal = "elevated_negative"
            funding_bias = "slightly_bullish"
        else:
            funding_signal = "normal"
            funding_bias = "neutral"

        # OI change analysis
        oi_change_pct = 0
        if prev_open_interests and asset in prev_open_interests and prev_open_interests[asset]:
            oi_change_pct = ((oi - prev_open_interests[asset]) / prev_open_interests[asset]) * 100

        if oi_change_pct > 10:
            oi_signal = "surging"
        elif oi_change_pct > 3:
            oi_signal = "rising"
        elif oi_change_pct < -10:
            oi_signal = "plunging"
        elif oi_change_pct < -3:
            oi_signal = "declining"
        else:
            oi_signal = "stable"

        # Combined risk assessment
        liquidation_risk = "high" if (funding_signal.startswith("extreme") and oi_signal in ("surging", "rising")) else "low"

        signals[asset] = {
            "funding_annualized_pct": round(annualized, 2),
            "funding_signal": funding_signal,
            "funding_bias": funding_bias,
            "oi": oi,
            "oi_change_pct": round(oi_change_pct, 2),
            "oi_signal": oi_signal,
            "liquidation_risk": liquidation_risk,
        }

    return signals


# ---------------------------------------------------------------------------
# Optional: Ollama local LLM synthesis
# ---------------------------------------------------------------------------

async def ollama_synthesize(items: list[NewsItem], asset: str, macro: dict) -> Optional[dict]:
    """Use local Ollama model for deeper analysis if available.

    This is OPTIONAL — the system works fine without it using keyword sentiment.
    Returns None if Ollama is not available.
    """
    try:
        import ollama as _ollama
        client = _ollama.Client(host="http://localhost:11434")
        # Quick check if Ollama is running
        client.list()
    except Exception:
        return None

    # Build concise context from top items
    fresh_items = [i for i in items if i.is_fresh]
    fresh_items.sort(key=lambda x: x.credibility, reverse=True)

    headlines = []
    for item in fresh_items[:15]:
        cred_label = "HIGH" if item.credibility >= 0.8 else "MED" if item.credibility >= 0.5 else "LOW"
        headlines.append(f"[{cred_label}] {item.title}")

    if not headlines:
        return None

    prompt = f"""Analyze these {asset} headlines for a quantitative trading system.

Headlines (sorted by source credibility):
{chr(10).join(headlines)}

Market: Fear/Greed={macro.get('fear_greed', {}).get('value', 50)}, BTC dominance={macro.get('global', {}).get('btc_dominance', 0)}%

Return ONLY valid JSON (no markdown, no explanation):
{{"sentiment": <float -1 to 1>, "urgency": "<immediate/today/week/none>", "key_event": "<most impactful event in 1 sentence or null>", "risk_alert": "<critical risk or null>", "bias": "<bullish/bearish/neutral>"}}"""

    try:
        # Try fast model first, fall back to any available
        models = client.list()
        model_names = [m.get("name", "") if isinstance(m, dict) else str(m) for m in models.get("models", [])]

        model = None
        for preferred in ["qwen3-nothink:latest", "qwen3:latest", "llama3:latest", "mistral:latest", "gemma:latest"]:
            if any(preferred in n for n in model_names):
                model = preferred
                break

        if not model and model_names:
            model = model_names[0]

        if not model:
            return None

        response = client.generate(
            model=model,
            prompt=prompt,
            options={"temperature": 0.1, "num_ctx": 4096, "num_predict": 300},
        )

        text = response.get("response", "").strip()
        # Clean markdown fences
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        return json.loads(text)
    except Exception as e:
        logger.debug("Ollama synthesis failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Master research orchestrator
# ---------------------------------------------------------------------------

class ResearchEngine:
    """Orchestrates all research sources and produces a structured briefing.

    Runs before the quant signal engine. All sources are free.
    Cross-validates events and computes sentiment without paid APIs.
    """

    def __init__(self, assets: list[str], use_ollama: bool = True):
        self.assets = assets
        self.use_ollama = use_ollama
        self._prev_funding: dict[str, float] = {}
        self._prev_oi: dict[str, float] = {}
        self._cache: Optional[ResearchBriefing] = None
        self._cache_time: float = 0
        self._cache_ttl: float = 300  # 5 min cache to avoid hammering free APIs

    async def get_briefing(
        self,
        funding_rates: dict[str, float] | None = None,
        open_interests: dict[str, float] | None = None,
    ) -> ResearchBriefing:
        """Produce a complete research briefing.

        Args:
            funding_rates: Current funding rates per asset (from Hyperliquid)
            open_interests: Current OI per asset (from Hyperliquid)

        Returns:
            ResearchBriefing with all findings, ready for quant engine consumption.
        """
        # Check cache
        now = time.time()
        if self._cache and (now - self._cache_time) < self._cache_ttl:
            # Update on-chain data even with cached news
            if funding_rates:
                self._cache.on_chain_context = analyze_onchain_signals(
                    funding_rates, open_interests or {},
                    self._prev_funding, self._prev_oi
                )
            return self._cache

        logger.info("Research engine: fetching fresh data from free sources...")

        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        async with aiohttp.ClientSession(headers=headers) as session:
            # Fetch all sources in parallel
            tasks = {
                "rss": fetch_rss_news(session),
                "google": fetch_google_news(session, "cryptocurrency+market"),
                "reddit": fetch_reddit(session, self.assets),
                "fear_greed": fetch_fear_greed(session),
                "global": fetch_coingecko_global(session),
                "trending": fetch_coingecko_trending(session),
                "stablecoins": fetch_defi_llama_stablecoins(session),
            }

            results = {}
            gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for key, result in zip(tasks.keys(), gathered):
                if isinstance(result, Exception):
                    logger.warning("Research source '%s' failed: %s", key, result)
                    results[key] = [] if key in ("rss", "google", "reddit", "trending") else {}
                else:
                    results[key] = result

            # Combine all news items
            all_items: list[NewsItem] = []
            all_items.extend(results.get("rss", []))
            all_items.extend(results.get("google", []))
            all_items.extend(results.get("reddit", []))

            # Compute per-asset sentiment
            asset_signals = {}
            for asset in self.assets:
                sentiment = compute_keyword_sentiment(all_items, asset)

                # Optional: enhance with Ollama
                ollama_result = None
                if self.use_ollama and sentiment.get("relevant_articles", 0) >= 3:
                    ollama_result = await ollama_synthesize(
                        all_items, asset,
                        {"fear_greed": results.get("fear_greed", {}),
                         "global": results.get("global", {})}
                    )

                asset_signals[asset] = {
                    "keyword_sentiment": sentiment,
                    "ollama_sentiment": ollama_result,
                    # Combine: keyword is primary, Ollama is supplementary
                    "final_sentiment": _merge_sentiment(sentiment, ollama_result),
                }

            # Cross-validate events
            key_events = cross_validate_events(all_items, min_sources=2)

            # On-chain analysis
            on_chain = {}
            if funding_rates:
                on_chain = analyze_onchain_signals(
                    funding_rates, open_interests or {},
                    self._prev_funding, self._prev_oi
                )
                self._prev_funding = dict(funding_rates)
                self._prev_oi = dict(open_interests or {})

            # Market sentiment aggregate
            fear_greed = results.get("fear_greed", {})
            fg_value = fear_greed.get("value", 50)
            asset_scores = [s["final_sentiment"]["score"] for s in asset_signals.values()]
            avg_sentiment = sum(asset_scores) / len(asset_scores) if asset_scores else 0

            market_sentiment = {
                "aggregate_score": round(avg_sentiment, 4),
                "fear_greed_index": fg_value,
                "fear_greed_label": fear_greed.get("label", "Neutral"),
                "fear_greed_trend": fear_greed.get("trend_7d", "stable"),
                "market_label": _sentiment_label(avg_sentiment, fg_value),
            }

            # Macro context
            global_data = results.get("global", {})
            trending = results.get("trending", [])
            stablecoins = results.get("stablecoins", {})

            macro_context = {
                "global_market": global_data,
                "trending_coins": [t["symbol"] for t in trending[:5]],
                "stablecoin_mcap": stablecoins.get("total_stablecoin_mcap", 0),
                "trending_in_our_assets": [
                    t["symbol"] for t in trending if t["symbol"] in self.assets
                ],
            }

            # Risk alerts
            risk_alerts = _compute_risk_alerts(
                fear_greed, on_chain, key_events, market_sentiment
            )

            # Data quality assessment
            fresh_count = sum(1 for i in all_items if i.is_fresh)
            sources_used = len(set(i.source for i in all_items))
            data_quality = {
                "total_items": len(all_items),
                "fresh_items_24h": fresh_count,
                "freshness_ratio": round(fresh_count / len(all_items), 2) if all_items else 0,
                "sources_count": sources_used,
                "sources_list": list(set(i.source for i in all_items)),
                "cross_validated_events": len(key_events),
                "ollama_available": any(s.get("ollama_sentiment") for s in asset_signals.values()),
            }

            briefing = ResearchBriefing(
                timestamp=datetime.now(timezone.utc).isoformat(),
                market_sentiment=market_sentiment,
                key_events=key_events,
                asset_signals=asset_signals,
                macro_context=macro_context,
                risk_alerts=risk_alerts,
                data_quality=data_quality,
                on_chain_context=on_chain,
            )

            # Cache it
            self._cache = briefing
            self._cache_time = now

            logger.info(
                "Research complete: %d items, %d fresh, %d events, %d alerts, %d sources",
                len(all_items), fresh_count, len(key_events), len(risk_alerts), sources_used
            )

            return briefing


def _merge_sentiment(keyword: dict, ollama: Optional[dict]) -> dict:
    """Merge keyword and Ollama sentiment. Keyword is primary."""
    score = keyword.get("score", 0)
    confidence = keyword.get("confidence", 0)

    if ollama and isinstance(ollama, dict):
        ollama_score = ollama.get("sentiment", 0)
        # Weighted average: 60% keyword, 40% Ollama
        score = score * 0.6 + ollama_score * 0.4
        # Boost confidence if both agree on direction
        if (score > 0 and ollama_score > 0) or (score < 0 and ollama_score < 0):
            confidence = min(confidence + 0.15, 1.0)

    label = keyword.get("label", "neutral")
    if score > 0.3:
        label = "bullish"
    elif score < -0.3:
        label = "bearish"
    elif score > 0.1:
        label = "slightly_bullish"
    elif score < -0.1:
        label = "slightly_bearish"

    result = {
        "score": round(score, 4),
        "confidence": round(confidence, 4),
        "label": label,
    }

    # Add Ollama extras if available
    if ollama and isinstance(ollama, dict):
        if ollama.get("key_event"):
            result["key_event"] = ollama["key_event"]
        if ollama.get("risk_alert"):
            result["risk_alert"] = ollama["risk_alert"]
        if ollama.get("urgency"):
            result["urgency"] = ollama["urgency"]

    return result


def _sentiment_label(avg_score: float, fear_greed: int) -> str:
    """Generate overall market sentiment label."""
    if fear_greed <= 20:
        return "extreme_fear"
    elif fear_greed <= 35:
        return "fear"
    elif fear_greed >= 80:
        return "extreme_greed"
    elif fear_greed >= 65:
        return "greed"
    elif avg_score > 0.2:
        return "cautiously_bullish"
    elif avg_score < -0.2:
        return "cautiously_bearish"
    return "neutral"


def _compute_risk_alerts(
    fear_greed: dict,
    on_chain: dict,
    events: list[dict],
    sentiment: dict,
) -> list[str]:
    """Generate priority risk alerts from all research data."""
    alerts = []

    # Fear & Greed extremes
    fg = fear_greed.get("value", 50)
    if fg <= 15:
        alerts.append(f"EXTREME FEAR ({fg}/100) — potential capitulation or contrarian buy zone")
    elif fg >= 85:
        alerts.append(f"EXTREME GREED ({fg}/100) — elevated risk of correction")

    # On-chain liquidation risk
    for asset, signals in on_chain.items():
        if signals.get("liquidation_risk") == "high":
            alerts.append(
                f"LIQUIDATION RISK: {asset} — extreme funding ({signals['funding_annualized_pct']}% ann.) "
                f"with {signals['oi_signal']} OI"
            )

    # Cross-validated high-impact events
    for event in events:
        if event.get("impact") == "high":
            alerts.append(f"HIGH-IMPACT EVENT ({event['source_count']} sources): {event['sample_title'][:80]}")

    return alerts[:5]  # Cap at 5 most important
