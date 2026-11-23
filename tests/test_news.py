import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from stock_direction.cli import main
from stock_direction.news import (
    NEWS_FEATURE_COLUMNS,
    add_news_features,
    load_scored_news,
    normalize_news,
    score_news,
    session_closes,
)
from stock_direction.news_cli import main as news_main

pytest.importorskip("exchange_calendars")
pytest.importorskip("vaderSentiment")


def articles(times, ticker="AAPL", scores=None):
    times = pd.to_datetime(times, utc=True)
    return pd.DataFrame(
        {
            "Ticker": ticker,
            "Headline": [f"Headline {i}" for i in range(len(times))],
            "Published_At": times,
            "Available_At": times,
            "Sentiment": scores if scores is not None else np.ones(len(times)) * 0.5,
        }
    )


def test_normalization_rejects_bad_legacy_data_and_deduplicates():
    source = pd.DataFrame(
        {
            "DATE": ["1970-01-01", "invalid", "2020-01-02", "2020-01-02", "2020-01-03"],
            "Headline": ["missing date", "bad date", "  Positive  news ", "positive news", ""],
            "Ticker": ["AAPL", "MSFT", "GOOG", "GOOGL", "AAPL"],
        }
    )
    result = normalize_news(source)
    assert result.audit["invalid_rows"] == 3
    assert result.audit["duplicates_removed"] == 1
    assert result.articles["Ticker"].tolist() == ["GOOGL"]
    assert result.articles["Available_At"].iloc[0] == pd.Timestamp("2020-01-03 05:00Z")


def test_timestamp_mode_respects_offsets_and_source_timezone():
    source = pd.DataFrame(
        {
            "Date": ["2020-07-02 10:00:00-04:00", "2020-07-02 11:00:00", "2020-07-02 00:00:00 UTC"],
            "Article_title": ["one", "two", "three"],
            "Stock_symbol": ["AAPL"] * 3,
        }
    )
    result = normalize_news(source, source_timezone="America/New_York", date_only=False)
    assert result.articles["Available_At"].iloc[0] == pd.Timestamp("2020-07-02 14:00Z")
    assert result.articles["Available_At"].iloc[1] == pd.Timestamp("2020-07-02 15:00Z")
    assert result.articles["Available_At"].iloc[2] == pd.Timestamp("2020-07-03 04:00Z")
    assert result.audit["date_precision_rows"] == 1


def test_url_date_mode_never_falls_back_to_an_earlier_unverified_source_date():
    source = pd.DataFrame(
        {
            "Date": ["2016-01-01 12:00:00 UTC"] * 3,
            "Article_title": ["News one", "News two", "News three"],
            "Stock_symbol": ["AAPL"] * 3,
            "Url": [
                "https://www.nasdaq.com/articles/example-2016-01-04",
                "https://www.nasdaq.com/articles/undated-example",
                "https://not-nasdaq.com/articles/example-2016-01-04",
            ],
        }
    )
    result = normalize_news(source, require_url_dates=True, publication_delay_hours=24)
    assert len(result.articles) == 1
    assert result.articles["Available_At"].iloc[0] == pd.Timestamp("2016-01-06 05:00Z")
    assert result.audit["rows_without_supported_url_date"] == 2
    assert result.audit["url_date_disagrees_with_source_date"] == 1


def test_vader_scores_without_training_and_records_lexicon():
    source = pd.DataFrame(
        {
            "Ticker": ["AAPL", "AAPL"],
            "DATE": ["2020-01-02"] * 2,
            "Headline": ["Excellent profits and great growth!", "Terrible losses and bankruptcy!"],
        }
    )
    result = score_news(normalize_news(source))
    assert result.articles["Sentiment"].iloc[0] > 0
    assert result.articles["Sentiment"].iloc[1] < 0
    assert len(result.audit["lexicon_sha256"]) == 64


def test_calendar_handles_dst_and_early_closes():
    dates = pd.to_datetime(["2020-03-06", "2020-03-09", "2020-11-27"])
    expected = pd.to_datetime(["2020-03-06 21:00Z", "2020-03-09 20:00Z", "2020-11-27 18:00Z"])
    assert session_closes(dates).equals(expected)
    with pytest.raises(ValueError, match="outside"):
        session_closes(pd.to_datetime(["2020-12-25"]))


def test_after_close_and_weekend_news_enter_next_session():
    prices = pd.DataFrame({"DATE": pd.to_datetime(["2020-03-06", "2020-03-09", "2020-03-10"])})
    news = articles(["2020-03-06 20:50Z", "2020-03-06 21:00Z", "2020-03-07 10:00Z"])
    result, coverage = add_news_features(prices, news, "AAPL", latency_minutes=0)
    assert coverage["News_Count"].tolist() == [1, 2, 0]
    assert result["news_present"].tolist() == [1, 1, 0]
    assert result["news_sentiment_mean"].tolist() == [0.5, 0.5, 0]


def test_early_close_and_ingestion_delay_do_not_leak():
    prices = pd.DataFrame({"DATE": pd.to_datetime(["2020-11-27", "2020-11-30"])})
    news = articles(["2020-11-27 17:54Z", "2020-11-27 17:56Z", "2020-11-27 19:00Z"])
    _, coverage = add_news_features(prices, news, "AAPL", latency_minutes=5)
    assert coverage["News_Count"].tolist() == [1, 2]


def test_news_features_unchanged_when_future_articles_are_added():
    prices = pd.DataFrame({"DATE": pd.to_datetime(["2020-03-06", "2020-03-09", "2020-03-10"])})
    news = articles(["2020-03-06 20:00Z", "2020-03-09 19:00Z", "2020-03-10 19:00Z"])
    prefix, _ = add_news_features(prices.iloc[:2], news.iloc[:2], "AAPL")
    complete, _ = add_news_features(prices, news, "AAPL")
    assert_frame_equal(prefix, complete.iloc[:2])


def test_other_ticker_news_and_old_archive_do_not_inflate_counts():
    prices = pd.DataFrame({"DATE": pd.to_datetime(["2020-03-06", "2020-03-09"])})
    news = pd.concat(
        [articles(["2019-01-01 10:00Z"]), articles(["2020-03-06 19:00Z"], ticker="MSFT")]
    )
    result, coverage = add_news_features(prices, news, "AAPL")
    assert coverage["News_Count"].sum() == 0
    assert (result[list(NEWS_FEATURE_COLUMNS)] == 0).all().all()


def test_prepared_news_rejects_availability_before_publication(tmp_path):
    news = articles(["2020-03-06 20:00Z"])
    news["Available_At"] = pd.to_datetime(["2020-03-06 19:00Z"])
    path = tmp_path / "news.csv"
    news.to_csv(path, index=False)
    with pytest.raises(ValueError, match="before publication"):
        load_scored_news(path)


def test_prepare_command_preserves_source_and_produces_audit(tmp_path):
    path = tmp_path / "raw.csv"
    pd.DataFrame({"Ticker": ["AAPL"], "DATE": ["2020-01-02"], "Headline": ["Great growth"]}).to_csv(
        path, index=False
    )
    original = path.read_bytes()
    output = tmp_path / "prepared.csv"
    assert news_main(["prepare", str(path), "--output", str(output)]) == 0
    data = load_scored_news(output)
    assert data["Available_At"].iloc[0] == pd.Timestamp("2020-01-04 05:00Z")
    assert path.read_bytes() == original
    manifest = json.loads(output.with_suffix(".metadata.json").read_text())
    assert manifest["sentiment_backend"] == "vader"
    assert manifest["accepted_rows"] == 1


def test_known_fnspid_source_cannot_silently_use_its_unverified_timestamps(tmp_path):
    path = tmp_path / "fnspid.csv"
    pd.DataFrame({"Ticker": ["AAPL"], "DATE": ["2020-01-02"], "Headline": ["News"]}).to_csv(
        path, index=False
    )
    path.with_suffix(".metadata.json").write_text(
        json.dumps({"original_dataset": "Zihan1004/FNSPID"})
    )
    with pytest.raises(SystemExit):
        news_main(["prepare", str(path), "--output", str(tmp_path / "scored.csv")])


def test_downloader_scans_all_shards_and_saves_only_selected_headlines(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    import fsspec

    from stock_direction import news_cli

    shards = ["first.parquet", "second.parquet"]
    for shard, tickers in zip(shards, [["AAPL", "aapl", "META"], [" goog ", "msft", "ZZZZ"]]):
        table = pa.Table.from_pydict(
            {
                "Date": ["2020-01-02 12:00:00 UTC"] * 3,
                "Article_title": [f"News for {ticker}" for ticker in tickers],
                "Stock_symbol": tickers,
                "Url": ["https://example.test"] * 3,
                "Publisher": ["test"] * 3,
                "Article": ["unneeded body"] * 3,
            }
        )
        pq.write_table(table, tmp_path / shard, row_group_size=1)

    class LocalFiles:
        def open(self, url, mode, **kwargs):
            return (tmp_path / url.rsplit("/", 1)[-1]).open(mode)

    monkeypatch.setattr(fsspec, "filesystem", lambda *args, **kwargs: LocalFiles())
    monkeypatch.setattr(
        news_cli,
        "get_json",
        lambda url: {
            "sha": "test-revision",
            "siblings": [{"rfilename": shard} for shard in shards],
        },
    )
    path = tmp_path / "headlines.csv"
    metadata = news_cli.fetch_fnspid(path)
    saved = pd.read_csv(path)
    assert len(saved) == 4
    assert set(saved["Stock_symbol"]) == {"AAPL", "GOOG", "MSFT"}
    assert "Article" not in saved
    assert metadata["shards_scanned"] == shards
    assert metadata["complete_query"] is True


@pytest.mark.parametrize("compressed", [False, True])
def test_benzinga_download_filters_complete_csv_and_records_provenance(
    tmp_path, monkeypatch, compressed
):
    from stock_direction import news_cli

    raw = b"title,date,stock\nGreat news,2020-01-02 10:00:00-04:00,AAPL\nGood earnings,2020-01-03 10:00:00-04:00,msft\nNews,2020-01-03,GOOG\nOther,2020-01-03,TSLA\nIncomplete row\n"
    if compressed:
        raw = raw.replace(b"title,date,stock", b"headline,date,stock")
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("raw_analyst_ratings.csv", raw)
        payload = archive.getvalue()
    else:
        payload = raw

    class Response(io.BytesIO):
        headers = {"Content-Type": "application/zip" if compressed else "text/csv"}

    monkeypatch.setattr(news_cli, "urlopen", lambda *args, **kwargs: Response(payload))
    path = tmp_path / "benzinga.csv"
    metadata = news_cli.fetch_benzinga(path)
    output = pd.read_csv(path)
    assert metadata["scanned_rows"] == 5
    assert metadata["rows_without_symbol"] == 1
    assert metadata["matching_rows"] == 3
    assert set(output["Ticker"]) == {"AAPL", "MSFT", "GOOG"}
    assert metadata["complete_query"] is True
    assert len(metadata["download_sha256"]) == 64


def test_cli_news_ablation_uses_identical_dates_and_rejects_uncovered_years(prices, tmp_path):
    import exchange_calendars as xcals

    calendar = xcals.get_calendar("XNYS", start="2013-01-01", end="2018-12-31")
    prices = prices.loc[prices["DATE"].isin(calendar.sessions)].copy()
    prices.to_csv(tmp_path / "historical_AAPL_data.csv", index=False)
    times = pd.DatetimeIndex(prices["DATE"]).tz_localize("UTC") + pd.Timedelta(hours=12)
    news = articles(times, scores=np.resize([-0.5, 0.5], len(times)))
    path = tmp_path / "news.csv"
    news.to_csv(path, index=False)
    output = tmp_path / "run"
    args = [
        "--data-dir",
        str(tmp_path),
        "--news-csv",
        str(path),
        "--models",
        "lr",
        "majority",
        "--start-year",
        "2013",
        "--first-train-end-year",
        "2015",
        "--last-test-year",
        "2017",
        "--output-dir",
        str(output),
    ]
    assert main(args) == 0
    before = pd.read_csv(output / "lr_prices" / "predictions.csv")
    after = pd.read_csv(output / "lr_prices_news" / "predictions.csv")
    assert_frame_equal(before[["Date", "Actual"]], after[["Date", "Actual"]])
    result = pd.read_csv(output / "comparison.csv")
    assert len(result) == 4
    assert result["n_predictions"].nunique() == 1
    majority = result.loc[
        (result["model"] == "majority") & result["feature_set"].eq("prices_news")
    ].iloc[0]
    assert majority["accuracy_delta_vs_prices"] == 0
    assert majority["paired_news_delta_ci_low"] == majority["paired_news_delta_ci_high"] == 0
    args[args.index("--last-test-year") + 1] = "2020"
    args[args.index("--output-dir") + 1] = str(tmp_path / "uncovered_run")
    with pytest.raises(SystemExit):
        main(args)
