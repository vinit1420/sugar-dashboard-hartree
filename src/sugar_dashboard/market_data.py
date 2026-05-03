from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from sugar_dashboard.config import MARKET_HISTORY_DIR


@dataclass(frozen=True)
class MarketSeries:
    symbol: str
    label: str
    unit: str
    frame: pd.DataFrame
    source: str
    error: str | None = None


MARKET_SYMBOLS = {
    "SB=F": ("NY11 Sugar Continuous", "c/lb"),
    "BZ=F": ("Brent Crude Continuous", "$/bbl"),
}
MARKET_HISTORY_CACHE = MARKET_HISTORY_DIR / "yfinance_daily_history.csv"
MARKET_REFRESH_MARKER = MARKET_HISTORY_DIR / "yfinance_refresh_marker.json"


def _empty_series(symbol: str, label: str, unit: str, error: str) -> MarketSeries:
    return MarketSeries(
        symbol=symbol,
        label=label,
        unit=unit,
        frame=pd.DataFrame(columns=["date", "close", "symbol", "label", "unit"]),
        source="Yahoo Finance via yfinance",
        error=error,
    )


def _read_cached_history() -> pd.DataFrame:
    if not MARKET_HISTORY_CACHE.exists():
        return pd.DataFrame(columns=["date", "close", "symbol", "label", "unit"])
    frame = pd.read_csv(MARKET_HISTORY_CACHE)
    if frame.empty:
        return pd.DataFrame(columns=["date", "close", "symbol", "label", "unit"])
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    return frame.dropna(subset=["close"])


def _write_cached_history(frame: pd.DataFrame) -> None:
    MARKET_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    cache_frame = frame.copy()
    cache_frame["date"] = pd.to_datetime(cache_frame["date"]).dt.date
    cache_frame = cache_frame.dropna(subset=["date", "close", "symbol"])
    cache_frame = cache_frame.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last")
    cache_frame.to_csv(MARKET_HISTORY_CACHE, index=False)


def _mark_refresh_attempted() -> None:
    MARKET_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    MARKET_REFRESH_MARKER.write_text(json.dumps({"last_attempt": date.today().isoformat()}, indent=2), encoding="utf-8")


def _refresh_attempted_today() -> bool:
    if not MARKET_REFRESH_MARKER.exists():
        return False
    try:
        payload = json.loads(MARKET_REFRESH_MARKER.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("last_attempt") == date.today().isoformat()


def _cached_symbol_frame(symbol: str, label: str, unit: str, years: int) -> pd.DataFrame:
    cached = _read_cached_history()
    if cached.empty:
        return pd.DataFrame(columns=["date", "close", "symbol", "label", "unit"])
    start_date = date.today() - timedelta(days=365 * years + 14)
    symbol_frame = cached[cached["symbol"] == symbol].copy()
    if symbol_frame.empty:
        return symbol_frame
    symbol_frame = symbol_frame[pd.to_datetime(symbol_frame["date"]).dt.date >= start_date]
    symbol_frame["label"] = label
    symbol_frame["unit"] = unit
    return symbol_frame


def _needs_daily_refresh(symbol_frame: pd.DataFrame) -> bool:
    if symbol_frame.empty:
        return True
    latest_date = pd.to_datetime(symbol_frame["date"]).dt.date.max()
    return latest_date < date.today() - timedelta(days=1)


def _download_yfinance_history(
    symbol: str,
    label: str,
    unit: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("yfinance is not installed.")

    raw = yf.download(
        symbol,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False,
    )

    if raw.empty:
        return pd.DataFrame(columns=["date", "close", "symbol", "label", "unit"])

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [column[0] for column in raw.columns]

    close_column = "Adj Close" if "Adj Close" in raw.columns else "Close"
    frame = raw.reset_index()[["Date", close_column]].rename(columns={"Date": "date", close_column: "close"})
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["close"])
    frame["symbol"] = symbol
    frame["label"] = label
    frame["unit"] = unit
    return frame


def seed_market_history_cache(years: int = 2) -> list[MarketSeries]:
    all_frames = []
    series_list = []
    existing_cache = _read_cached_history()
    end_date = date.today() + timedelta(days=1)
    start_date = end_date - timedelta(days=365 * years + 14)
    for symbol, (label, unit) in MARKET_SYMBOLS.items():
        try:
            frame = _download_yfinance_history(symbol, label, unit, start_date=start_date, end_date=end_date)
        except Exception as exc:
            frame = _cached_symbol_frame(symbol, label, unit, years)
            series_list.append(
                MarketSeries(
                    symbol=symbol,
                    label=label,
                    unit=unit,
                    frame=frame,
                    source="Local market history cache",
                    error=f"Unable to refresh {symbol}: {exc}",
                )
            )
            continue
        all_frames.append(frame)
        series_list.append(
            MarketSeries(symbol=symbol, label=label, unit=unit, frame=frame, source="Yahoo Finance via yfinance")
        )

    frames_to_cache = [existing_cache] + all_frames if not existing_cache.empty else all_frames
    if frames_to_cache:
        _write_cached_history(pd.concat(frames_to_cache, ignore_index=True))
    _mark_refresh_attempted()
    return series_list


def fetch_yfinance_history(symbol: str, label: str, unit: str, years: int = 2) -> MarketSeries:
    cached_frame = _cached_symbol_frame(symbol, label, unit, years)
    if not cached_frame.empty and _refresh_attempted_today():
        return MarketSeries(symbol=symbol, label=label, unit=unit, frame=cached_frame, source="Local market history cache")

    if not _needs_daily_refresh(cached_frame):
        return MarketSeries(symbol=symbol, label=label, unit=unit, frame=cached_frame, source="Local market history cache")

    start_date = date.today() - timedelta(days=365 * years + 14) if cached_frame.empty else pd.to_datetime(cached_frame["date"]).dt.date.max()
    end_date = date.today() + timedelta(days=1)
    try:
        fresh_frame = _download_yfinance_history(symbol, label, unit, start_date=start_date, end_date=end_date)
    except Exception as exc:
        _mark_refresh_attempted()
        if not cached_frame.empty:
            return MarketSeries(
                symbol=symbol,
                label=label,
                unit=unit,
                frame=cached_frame,
                source="Local market history cache",
                error=f"Daily Yahoo refresh failed; using cached history. {exc}",
            )
        return _empty_series(symbol, label, unit, f"Unable to download {symbol}: {exc}")

    if fresh_frame.empty and cached_frame.empty:
        _mark_refresh_attempted()
        return _empty_series(symbol, label, unit, f"No history returned for {symbol}.")

    if fresh_frame.empty:
        _mark_refresh_attempted()
        return MarketSeries(symbol=symbol, label=label, unit=unit, frame=cached_frame, source="Local market history cache")

    merged = pd.concat([cached_frame, fresh_frame], ignore_index=True)
    all_cached = _read_cached_history()
    other_symbols = all_cached[all_cached["symbol"] != symbol] if not all_cached.empty else pd.DataFrame()
    _write_cached_history(pd.concat([other_symbols, merged], ignore_index=True))
    _mark_refresh_attempted()
    final_frame = _cached_symbol_frame(symbol, label, unit, years)
    source = "Local market history cache + daily Yahoo refresh"
    return MarketSeries(symbol=symbol, label=label, unit=unit, frame=final_frame, source=source)


def fetch_market_history(years: int = 2) -> list[MarketSeries]:
    return [
        fetch_yfinance_history(symbol=symbol, label=label, unit=unit, years=years)
        for symbol, (label, unit) in MARKET_SYMBOLS.items()
    ]


def build_market_explorer_frame(series_list: list[MarketSeries]) -> pd.DataFrame:
    frames = [series.frame for series in series_list if not series.frame.empty]
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined["year"] = combined["date"].dt.year
    combined["day_of_year"] = combined["date"].dt.dayofyear
    combined["display_date"] = combined["date"]

    max_year = int(combined["year"].max())
    current_year = combined[combined["year"] == max_year].copy()
    last_year = combined[combined["year"] == max_year - 1].copy()
    last_year["display_date"] = last_year["date"] + pd.DateOffset(years=1)
    last_year["label"] = last_year["label"] + " LY"

    overlay = pd.concat([current_year, last_year], ignore_index=True)
    overlay = overlay.sort_values(["label", "display_date"]).reset_index(drop=True)
    overlay["indexed"] = overlay.groupby("label")["close"].transform(lambda values: (values / values.iloc[0]) * 100)
    overlay["mom_change"] = overlay.groupby("label")["close"].pct_change() * 100

    derived_frames = [overlay]
    for suffix in ("", " LY"):
        sugar_label = f"NY11 Sugar Continuous{suffix}"
        brent_label = f"Brent Crude Continuous{suffix}"
        sugar = overlay[overlay["label"] == sugar_label][["display_date", "date", "indexed"]].rename(
            columns={"indexed": "sugar_indexed"}
        )
        brent = overlay[overlay["label"] == brent_label][["display_date", "indexed"]].rename(
            columns={"indexed": "brent_indexed"}
        )
        spread = sugar.merge(brent, on="display_date", how="inner")
        if spread.empty:
            continue
        spread["close"] = spread["sugar_indexed"] - spread["brent_indexed"]
        spread["indexed"] = spread["close"]
        spread["mom_change"] = spread["close"].diff()
        spread["symbol"] = f"SB_BZ_INDEX_SPREAD{suffix}"
        spread["label"] = f"NY11-Brent Relative Spread{suffix}"
        spread["unit"] = "index pts"
        spread["year"] = pd.to_datetime(spread["display_date"]).dt.year
        spread["day_of_year"] = pd.to_datetime(spread["display_date"]).dt.dayofyear
        derived_frames.append(spread[["date", "close", "symbol", "label", "unit", "display_date", "year", "day_of_year", "indexed", "mom_change"]])

    return pd.concat(derived_frames, ignore_index=True).sort_values(["label", "display_date"]).reset_index(drop=True)
