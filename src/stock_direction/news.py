"""Headline normalization and causal session-level news features.

Historical public archives lack first-seen timestamps and article revisions.
Their publication times are a research proxy for availability, not proof of a
point-in-time feed. Date-only mode deliberately delays intraday information.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import validate_dates

NEWS_FEATURE_COLUMNS = (
    "news_count_log",
    "news_present",
    "news_sentiment_mean",
    "news_sentiment_std",
    "news_positive_share",
    "news_negative_share",
    "news_sentiment_ma7",
    "news_count_ma7",
)
TICKER_ALIASES = {"GOOG": "GOOGL"}


@dataclass
class NewsData:
    articles: pd.DataFrame
    audit: dict


def normalize_news(
    frame: pd.DataFrame,
    *,
    source_timezone: str = "UTC",
    date_only: bool = True,
    publication_delay_hours: float = 0,
    source_name: str = "user_csv",
    require_url_dates: bool = False,
) -> NewsData:
    """Normalize FNSPID/Benzinga/canonical rows, keeping an audit of rejected data.

    date_only=True uses the END of the stated publication day in New York, even
    if a source supplies a time. Otherwise naive times require source_timezone;
    missing or midnight times still use the conservative date rule. No guessed
    epoch timestamps and no neutral sentiment substituted for invalid articles.
    """
    if not np.isfinite(publication_delay_hours) or publication_delay_hours < 0:
        raise ValueError("publication_delay_hours must be finite and nonnegative")
    aliases = {
        "Ticker": ("Ticker", "Stock_symbol", "stock", "ticker"),
        "Headline": ("Headline", "Article_title", "headline", "title"),
        "Published_At": ("Published_At", "Date", "DATE", "date"),
        "URL": ("URL", "Url", "url"),
    }
    columns = {}
    for canonical, candidates in aliases.items():
        existing = [name for name in candidates if name in frame]
        if not existing:
            if canonical == "URL":
                columns[canonical] = pd.Series("", index=frame.index)
                continue
            raise ValueError(f"Missing news field {canonical}; accepted names: {candidates}")
        columns[canonical] = frame[existing[0]]
    work = pd.DataFrame(columns).reset_index(drop=True)
    work["Ticker"] = (
        work["Ticker"].fillna("").astype(str).str.strip().str.upper().replace(TICKER_ALIASES)
    )
    work["Headline"] = (
        work["Headline"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    )
    work["URL"] = work["URL"].fillna("").astype(str).str.strip()
    work["Source_Published_At"] = work["Published_At"].astype(str)
    if require_url_dates:
        nasdaq = work["URL"].str.match(r"^https?://(?:www\.)?nasdaq\.com/", case=False)
        encoded = work["URL"].str.extract(r"-(20\d{2}-\d{2}-\d{2})(?:$|[?#])", expand=False)
        work["Published_At"] = encoded.where(nasdaq)
        unsupported_url_rows = int(work["Published_At"].isna().sum())
        differing_url_rows = int(
            (
                work["Published_At"].notna()
                & work["Published_At"].ne(work["Source_Published_At"].str[:10])
            ).sum()
        )
    dates, precision = [], []
    for raw in work["Published_At"]:
        try:
            text = str(raw).strip()
            if not text or text.isdigit():
                raise ValueError("Numeric timestamps require explicit conversion by the importer")
            # FNSPID uses a literal ' UTC' suffix; pandas accepts ISO +00:00 reliably.
            timestamp = pd.Timestamp(re.sub(r"\s+UTC$", "+00:00", text))
            if pd.isna(timestamp) or timestamp.year < 1990 or timestamp.year > 2100:
                raise ValueError("Invalid news date")
            uncertain_time = (
                require_url_dates
                or date_only
                or timestamp.time() == pd.Timestamp("00:00:00").time()
            )
            if uncertain_time:
                # Keep the source's written date; do not move it backward by converting UTC.
                timestamp = (pd.Timestamp(timestamp.date()) + pd.Timedelta(days=1)).tz_localize(
                    "America/New_York"
                )
                precision.append("date")
            else:
                if timestamp.tzinfo is None:
                    timestamp = timestamp.tz_localize(
                        source_timezone, ambiguous="raise", nonexistent="raise"
                    )
                precision.append("timestamp")
            dates.append(timestamp.tz_convert("UTC"))
        except (ValueError, TypeError, OverflowError):
            dates.append(pd.NaT)
            precision.append("invalid")
    work["Published_At"] = pd.to_datetime(dates, utc=True)
    work["Timestamp_Precision"] = precision
    invalid = work["Published_At"].isna() | work["Headline"].eq("") | work["Ticker"].eq("")
    audit = {"input_rows": len(work), "invalid_rows": int(invalid.sum()), "source": source_name}
    if require_url_dates:
        audit["rows_without_supported_url_date"] = unsupported_url_rows
        audit["url_date_disagrees_with_source_date"] = differing_url_rows
    work = work.loc[~invalid].copy()
    work["Available_At"] = work["Published_At"] + pd.to_timedelta(publication_delay_hours, unit="h")
    work["Source"] = source_name
    work["_headline"] = work["Headline"].str.casefold()
    work["_day"] = work["Published_At"].dt.tz_convert("America/New_York").dt.date
    work = work.sort_values("Available_At", kind="stable")
    before = len(work)
    work = work.drop_duplicates(["Ticker", "_day", "_headline"], keep="first")
    audit["duplicates_removed"] = before - len(work)
    work = work.drop(columns=["_headline", "_day"]).reset_index(drop=True)
    audit.update(
        {
            "accepted_rows": len(work),
            "date_precision_rows": int(work["Timestamp_Precision"].eq("date").sum()),
            "date_only_policy": date_only,
            "source_timezone": source_timezone,
            "publication_delay_hours": publication_delay_hours,
            "availability_is_publication_proxy": True,
            "require_url_dates": require_url_dates,
        }
    )
    return NewsData(work, audit)


def score_news(news: NewsData) -> NewsData:
    """Frozen VADER lexicon baseline; no fitting on price targets or future articles."""
    try:
        from vaderSentiment import vaderSentiment
    except ImportError as exc:
        raise ImportError("Install stock-direction[news] to score headlines") from exc
    analyzer = vaderSentiment.SentimentIntensityAnalyzer()
    frame = news.articles.copy()
    unique = frame["Headline"].unique()
    scores = {headline: analyzer.polarity_scores(headline)["compound"] for headline in unique}
    frame["Sentiment"] = frame["Headline"].map(scores).astype(float)
    lexicon = Path(vaderSentiment.__file__).parent / "vader_lexicon.txt"
    return NewsData(
        frame,
        {
            **news.audit,
            "sentiment_backend": "vader",
            "sentiment_is_financial_domain_model": False,
            "lexicon_sha256": hashlib.sha256(lexicon.read_bytes()).hexdigest(),
        },
    )


def load_scored_news(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"Ticker", "Headline", "Published_At", "Available_At", "Sentiment"}
    missing = required.difference(frame)
    if missing:
        raise ValueError(
            f"Missing prepared news columns: {sorted(missing)}; run stock-news prepare"
        )
    for column in ("Published_At", "Available_At"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
        if frame[column].isna().any():
            raise ValueError(f"Missing news timestamp in {column}")
    if (frame["Available_At"] < frame["Published_At"]).any():
        raise ValueError("News cannot be available before publication")
    scores = frame["Sentiment"].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or (np.abs(scores) > 1).any():
        raise ValueError("Sentiment scores must be finite and within [-1, 1]")
    return frame


def session_closes(dates) -> pd.DatetimeIndex:
    """US regular-session closes, including early closes and daylight saving.

    exchange_calendars aliases NASDAQ to XNYS; these share regular US sessions.
    Reject an unrecognized price date instead of silently inventing a cutoff.
    """
    try:
        import exchange_calendars as xcals
    except ImportError as exc:
        raise ImportError("Install stock-direction[news] for the exchange calendar") from exc
    days = pd.DatetimeIndex(dates).normalize()
    if days.empty:
        return pd.DatetimeIndex([], tz="UTC")
    if days.tz is not None or not days.is_unique or not days.is_monotonic_increasing:
        raise ValueError("Price session dates must be naive, unique and ordered")
    calendar = xcals.get_calendar(
        "XNYS", start=days.min() - pd.Timedelta(days=7), end=days.max() + pd.Timedelta(days=7)
    )
    schedule = calendar.schedule.reindex(days)
    if schedule["close"].isna().any():
        bad = days[schedule["close"].isna()]
        raise ValueError(f"Price dates outside the US session calendar: {bad.tolist()[:5]}")
    return pd.DatetimeIndex(schedule["close"].array).tz_convert("UTC")


def add_news_features(
    prices: pd.DataFrame,
    articles: pd.DataFrame,
    ticker: str,
    *,
    latency_minutes: float = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate each article at its first eligible close, without filling later news backward.

    Only articles later than the preceding session's close enter the current bin.
    At the start of the price calendar, cap history at seven days so extending
    an archive far into the past cannot inflate the first observation's count.
    """
    if not np.isfinite(latency_minutes) or latency_minutes < 0:
        raise ValueError("latency_minutes must be finite and nonnegative")
    frame = validate_dates(prices)
    closes = session_closes(frame["DATE"])
    news = articles.loc[articles["Ticker"].eq(ticker)].copy()
    available = pd.to_datetime(news["Available_At"], utc=True) + pd.to_timedelta(
        latency_minutes, unit="m"
    )
    if len(closes) == 0:
        raise ValueError("Empty price calendar")
    keep = (available > closes[0] - pd.Timedelta(days=7)) & (available < closes[-1])
    news, available = news.loc[keep].copy(), available.loc[keep]
    positions = closes.searchsorted(pd.DatetimeIndex(available.array), side="right")
    news["Session_Date"] = frame["DATE"].to_numpy()[positions]
    news["Decision_At"] = closes.take(positions)
    news["Effective_Available_At"] = pd.DatetimeIndex(available.array)
    if not (
        pd.DatetimeIndex(news["Effective_Available_At"].array)
        < pd.DatetimeIndex(news["Decision_At"].array)
    ).all():
        raise ValueError("A future article entered a session")
    news["_positive"] = (news["Sentiment"] > 0.05).astype(float)
    news["_negative"] = (news["Sentiment"] < -0.05).astype(float)
    grouped = (
        news.groupby("Session_Date")
        .agg(
            news_count=("Sentiment", "size"),
            news_sentiment_mean=("Sentiment", "mean"),
            news_sentiment_std=("Sentiment", "std"),
            news_positive_share=("_positive", "mean"),
            news_negative_share=("_negative", "mean"),
        )
        .reindex(pd.DatetimeIndex(frame["DATE"]))
        .fillna(0)
    )
    grouped["news_count_log"] = np.log1p(grouped["news_count"])
    grouped["news_present"] = (grouped["news_count"] > 0).astype(float)
    grouped["news_sentiment_ma7"] = grouped["news_sentiment_mean"].rolling(7, min_periods=1).mean()
    grouped["news_count_ma7"] = grouped["news_count"].rolling(7, min_periods=1).mean()
    for name in NEWS_FEATURE_COLUMNS:
        frame[name] = grouped[name].to_numpy()
    coverage = pd.DataFrame(
        {
            "DATE": frame["DATE"],
            "Decision_At": closes,
            "News_Count": grouped["news_count"].to_numpy().astype(int),
        }
    )
    coverage.attrs["articles_assigned"] = len(news)
    return frame, coverage
