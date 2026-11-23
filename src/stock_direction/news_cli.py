"""Acquire a ticker subset, normalize headlines and score them for offline experiments."""

import argparse
import csv
import hashlib
import io
import json
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import product
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from .news import normalize_news, score_news

DEFAULT_DATASET = "benstaf/FNSPID-filtered-nasdaq-100"
BENZINGA_DATASET = "miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests"


def fetch_benzinga(output: Path, *, tickers=("AAPL", "MSFT", "GOOGL", "GOOG")):
    """Read the original raw Kaggle headlines, keeping only our issuers.

    Public Kaggle file endpoint, no API token. Coverage is audited per issuer;
    the inspected snapshot is incomplete for this project's research period.
    Dates have limited precision; preparation uses conservative date mode.
    """
    if output.exists() or output.with_suffix(".metadata.json").exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    filename = "raw_analyst_ratings.csv"
    url = f"https://www.kaggle.com/api/v1/datasets/download/{BENZINGA_DATASET}/{filename}?datasetVersionNumber=2"
    source_hash = hashlib.sha256()
    rows, scanned, missing_symbols = [], 0, 0
    with tempfile.TemporaryFile() as archive:
        with urlopen(
            Request(url, headers={"User-Agent": "stock-direction-research/0.1"}), timeout=60
        ) as response:
            content_type = response.headers.get("Content-Type", "")
            size = 0
            while chunk := response.read(1024 * 1024):
                source_hash.update(chunk)
                archive.write(chunk)
                size += len(chunk)
                if size > 300 * 1024 * 1024:
                    raise ValueError("News source exceeded the 300 MB download limit")
                if size % (10 * 1024 * 1024) == 0:
                    print(f"Downloaded {size // (1024 * 1024)} MB", flush=True)
            if "text/html" in content_type or "json" in content_type:
                raise ValueError("Kaggle returned an error/login page instead of the news file")
        archive.seek(0)

        def select_rows(stream):
            nonlocal scanned, missing_symbols
            reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig"))
            title_column = "headline" if "headline" in (reader.fieldnames or []) else "title"
            required = {title_column, "date", "stock"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Unexpected Benzinga source columns: {reader.fieldnames}")
            for item in reader:
                scanned += 1
                symbol = (item.get("stock") or "").strip().upper()
                if symbol in ("", "NAN", "NONE", "NULL"):
                    missing_symbols += 1
                    continue
                if symbol in tickers:
                    rows.append(
                        {
                            "Ticker": symbol,
                            "Headline": item[title_column],
                            "Published_At": item["date"],
                            "URL": item.get("url", ""),
                        }
                    )

        if zipfile.is_zipfile(archive):
            archive.seek(0)
            with zipfile.ZipFile(archive) as zipped:
                matching = [
                    name for name in zipped.namelist() if name.rsplit("/", 1)[-1] == filename
                ]
                if len(matching) != 1:
                    raise ValueError(f"Cannot identify {filename} in Kaggle archive")
                with zipped.open(matching[0]) as stream:
                    select_rows(stream)
        else:
            archive.seek(0)
            # The file endpoint can return a bare CSV rather than a zip.
            select_rows(archive)
    if not rows:
        raise ValueError("No matching issuer headlines in downloaded source")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, mode="x")
    metadata = {
        "dataset": BENZINGA_DATASET,
        "requested_version": 2,
        "filename": filename,
        "download_url": url,
        "source_url": f"https://www.kaggle.com/datasets/{BENZINGA_DATASET}",
        "download_sha256": source_hash.hexdigest(),
        "scanned_rows": scanned,
        "rows_without_symbol": missing_symbols,
        "matching_rows": len(rows),
        "complete_query": True,
        "tickers": list(tickers),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        "published_times_are_not_verified_first_seen_times": True,
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


def get_json(url):
    for attempt in range(3):
        try:
            with urlopen(
                Request(url, headers={"User-Agent": "stock-direction-research/0.1"}), timeout=60
            ) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
        time.sleep(2**attempt)


def fetch_fnspid(
    output: Path, *, dataset=DEFAULT_DATASET, tickers=("AAPL", "MSFT", "GOOGL", "GOOG")
):
    """Read pinned parquet shards through HTTP ranges and normalize source tickers.

    The original CSV viewer fails schema conversion and full-dataset filtering
    can be unavailable. This path reads every shard's metadata and only required
    headline columns, without downloading article bodies or trusting a preview.
    """
    try:
        import fsspec
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Install stock-direction[news-download] to acquire the parquet subset"
        ) from exc
    if output.exists() or output.with_suffix(".metadata.json").exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not all(symbol.isalnum() for symbol in tickers):
        raise ValueError("Tickers must be alphanumeric")
    info = get_json(f"https://huggingface.co/api/datasets/{dataset}")
    revision = info["sha"]
    shards = [
        item["rfilename"] for item in info["siblings"] if item["rfilename"].endswith(".parquet")
    ]
    if not shards:
        raise ValueError("Source has no parquet shards; choose a parquet FNSPID mirror")
    fields = ["Date", "Article_title", "Stock_symbol", "Url", "Publisher"]
    case_variants = {
        "".join(letters)
        for ticker in tickers
        for letters in product(*[(char.lower(), char.upper()) for char in ticker])
    }
    print(f"Scanning {len(shards)} parquet shards at revision {revision}", flush=True)

    def scan_shard(shard):
        url = f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{shard}"
        filesystem = fsspec.filesystem("http", client_kwargs={"trust_env": True})
        result = []
        with filesystem.open(url, "rb", block_size=256 * 1024, cache_type="blockcache") as stream:
            parquet = pq.ParquetFile(stream)
            symbol_index = parquet.schema.names.index("Stock_symbol")
            for group in range(parquet.num_row_groups):
                stats = parquet.metadata.row_group(group).column(symbol_index).statistics
                if stats is not None and stats.has_min_max:
                    low, high = stats.min, stats.max
                    if isinstance(low, bytes):
                        low, high = low.decode(), high.decode()
                    if low == high == "":
                        continue
                    # Only prune intervals bounded by plain ASCII symbols. All
                    # case permutations and arbitrary suffixes are covered;
                    # whitespace/non-ASCII bounds fall back to reading symbols.
                    if (
                        low.isascii()
                        and high.isascii()
                        and low.isalnum()
                        and high.isalnum()
                        and not any(
                            low <= variant + "\U0010ffff" and variant <= high
                            for variant in case_variants
                        )
                    ):
                        continue
                # Normalize possible case differences between archive components.
                # Raw uppercase-only min/max pruning would miss lowercase values.
                symbols = parquet.read_row_group(group, columns=["Stock_symbol"])["Stock_symbol"]
                normalized = pc.utf8_upper(pc.utf8_trim_whitespace(symbols))
                mask = pc.is_in(normalized, value_set=pa.array(tickers))
                if not pc.any(mask).as_py():
                    continue
                table = parquet.read_row_group(group, columns=fields)
                table = table.set_column(
                    table.schema.get_field_index("Stock_symbol"), "Stock_symbol", normalized
                )
                selected = table.filter(mask)
                result.extend(selected.to_pylist())
        print(f"{shard}: {len(result)} matching headlines", flush=True)
        return result

    rows = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for matches in pool.map(scan_shard, shards):
            rows.extend(matches)
    total = len(rows)
    if not total:
        raise ValueError("No matching headlines in source")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            writer.writerow({name: item.get(name) for name in fields})
    metadata = {
        "dataset": dataset,
        "revision": revision,
        "matching_rows": total,
        "complete_query": True,
        "shards_scanned": shards,
        "tickers": list(tickers),
        "original_dataset": "Zihan1004/FNSPID",
        "dataset_url": f"https://huggingface.co/datasets/{dataset}",
        "original_url": "https://huggingface.co/datasets/Zihan1004/FNSPID",
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "published_times_are_not_verified_first_seen_times": True,
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved {total} source headlines to {output}")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    benzinga = commands.add_parser(
        "fetch-benzinga", help="Fetch the original processed historical Kaggle headlines"
    )
    benzinga.add_argument(
        "--output", type=Path, default=Path("data/raw/news/benzinga_headlines.csv")
    )
    fetch = commands.add_parser(
        "fetch-fnspid", help="Fetch all matching headlines from a public mirror"
    )
    fetch.add_argument("--output", type=Path, default=Path("data/raw/news/fnspid_nasdaq100.csv"))
    fetch.add_argument("--dataset", default=DEFAULT_DATASET)
    prepare = commands.add_parser("prepare", help="Clean and score a locally saved news CSV")
    prepare.add_argument("input", type=Path)
    prepare.add_argument("--output", type=Path, default=Path("data/processed/news/vader_news.csv"))
    prepare.add_argument("--source-timezone", default="UTC")
    prepare.add_argument(
        "--require-url-date",
        action="store_true",
        help="Use only Nasdaq URLs ending in an encoded publication date; ignore source Date",
    )
    prepare.add_argument(
        "--trust-intraday-times",
        action="store_true",
        help="Opt in only after validating source clock and timezone",
    )
    prepare.add_argument(
        "--delay-hours",
        type=float,
        default=24,
        help="Additional publication delay; default 24 h for uncertain historical archives",
    )
    args = parser.parse_args(argv)
    if args.command == "fetch-benzinga":
        fetch_benzinga(args.output)
        return 0
    if args.command == "fetch-fnspid":
        fetch_fnspid(args.output, dataset=args.dataset)
        return 0
    if args.output.exists() or args.output.with_suffix(".metadata.json").exists():
        parser.error(f"Refusing to overwrite prepared news: {args.output}")
    provenance = args.input.with_suffix(".metadata.json")
    source_manifest = json.loads(provenance.read_text()) if provenance.exists() else None
    if (
        source_manifest
        and source_manifest.get("original_dataset") == "Zihan1004/FNSPID"
        and not args.require_url_date
    ):
        parser.error(
            "FNSPID timestamps failed the source audit; use --require-url-date to retain only dated Nasdaq URLs"
        )
    data = pd.read_csv(args.input)
    prepared = score_news(
        normalize_news(
            data,
            source_timezone=args.source_timezone,
            date_only=not args.trust_intraday_times,
            publication_delay_hours=args.delay_hours,
            source_name=str(args.input),
            require_url_dates=args.require_url_date,
        )
    )
    if prepared.articles.empty:
        parser.error("No usable news remained; inspect the source timestamps and headlines")
    metadata = {
        **prepared.audit,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "vader_version": version("vaderSentiment"),
        "coverage_by_ticker": {},
    }
    for ticker, group in prepared.articles.groupby("Ticker"):
        metadata["coverage_by_ticker"][ticker] = {
            "articles": len(group),
            "first_available": group["Available_At"].min().isoformat(),
            "last_available": group["Available_At"].max().isoformat(),
        }
    if source_manifest is not None:
        metadata["source_manifest"] = source_manifest
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prepared.articles.to_csv(args.output, index=False, mode="x")
    metadata["output_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    coverage = (
        prepared.articles.assign(Year=prepared.articles["Available_At"].dt.year)
        .groupby(["Ticker", "Year"])
        .size()
    )
    coverage.rename("Articles").to_csv(args.output.with_suffix(".coverage.csv"))
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
