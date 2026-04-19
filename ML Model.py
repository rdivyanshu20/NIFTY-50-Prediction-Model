import argparse
import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SYMBOL = "^NSEI"
DEFAULT_RANGE = "10y"
DEFAULT_INTERVAL = "1d"
DEFAULT_ALPHA = 5.0
HORIZONS = {
    "1_month": 21,
    "1_year": 252,
}
FEATURE_COLUMNS = [
    "close",
    "volume",
    "ema_50",
    "ema_100",
    "ema_200",
    "rsi_14",
    "ret_1d",
    "ret_5d",
    "ret_21d",
    "volatility_21d",
    "avg_volume_21d",
    "avg_volume_63d",
    "volume_ratio_21d",
    "volume_ratio_63d",
    "price_vs_ema50",
    "price_vs_ema100",
    "price_vs_ema200",
    "ema50_vs_ema200",
    "ema100_vs_ema200",
    "high_low_range",
    "close_open_range",
]


@dataclass
class PredictionMetrics:
    horizon_days: int
    predicted_return: float
    predicted_price: float
    mae: float
    rmse: float
    direction_accuracy: float


def fetch_yahoo_chart(
    symbol: str = DEFAULT_SYMBOL,
    history_range: str = DEFAULT_RANGE,
    interval: str = DEFAULT_INTERVAL,
) -> pd.DataFrame:
    encoded_symbol = urllib.parse.quote(symbol, safe="")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_symbol}"
        f"?range={history_range}&interval={interval}"
        "&includePrePost=false&events=div%2Csplits"
    )
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Yahoo Finance request failed with HTTP {exc.code}. "
            "Retry later or switch to another data source."
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Unable to reach Yahoo Finance. Check internet access and retry."
        ) from exc

    result = payload["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    frame = pd.DataFrame(
        {
            "timestamp": result["timestamp"],
            "open": quote.get("open", []),
            "high": quote.get("high", []),
            "low": quote.get("low", []),
            "close": quote.get("close", []),
            "volume": quote.get("volume", []),
        }
    )

    frame["date"] = (
        pd.to_datetime(frame["timestamp"], unit="s", utc=True)
        .dt.tz_convert("Asia/Kolkata")
        .dt.date
    )
    frame = frame.drop(columns=["timestamp"]).dropna(subset=["close"]).reset_index(drop=True)
    frame["volume"] = frame["volume"].fillna(0.0)
    return frame


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def build_feature_frame(price_df: pd.DataFrame) -> pd.DataFrame:
    frame = price_df.copy()
    frame["ema_50"] = frame["close"].ewm(span=50, adjust=False).mean()
    frame["ema_100"] = frame["close"].ewm(span=100, adjust=False).mean()
    frame["ema_200"] = frame["close"].ewm(span=200, adjust=False).mean()
    frame["rsi_14"] = compute_rsi(frame["close"], 14)

    frame["ret_1d"] = frame["close"].pct_change(1)
    frame["ret_5d"] = frame["close"].pct_change(5)
    frame["ret_21d"] = frame["close"].pct_change(21)
    frame["volatility_21d"] = frame["ret_1d"].rolling(21).std()

    frame["avg_volume_21d"] = frame["volume"].rolling(21).mean()
    frame["avg_volume_63d"] = frame["volume"].rolling(63).mean()
    frame["volume_ratio_21d"] = frame["volume"] / frame["avg_volume_21d"]
    frame["volume_ratio_63d"] = frame["volume"] / frame["avg_volume_63d"]

    frame["price_vs_ema50"] = frame["close"] / frame["ema_50"] - 1
    frame["price_vs_ema100"] = frame["close"] / frame["ema_100"] - 1
    frame["price_vs_ema200"] = frame["close"] / frame["ema_200"] - 1
    frame["ema50_vs_ema200"] = frame["ema_50"] / frame["ema_200"] - 1
    frame["ema100_vs_ema200"] = frame["ema_100"] / frame["ema_200"] - 1
    frame["high_low_range"] = (frame["high"] - frame["low"]) / frame["close"]
    frame["close_open_range"] = (frame["close"] - frame["open"]) / frame["open"]

    for horizon_name, horizon_days in HORIZONS.items():
        frame[f"target_{horizon_name}"] = frame["close"].shift(-horizon_days) / frame["close"] - 1

    return frame.replace([np.inf, -np.inf], np.nan)


def fit_ridge_regression(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    identity = np.eye(x.shape[1], dtype=float)
    return np.linalg.solve(x.T @ x + alpha * identity, x.T @ y)


def run_single_horizon_model(
    feature_df: pd.DataFrame,
    latest_feature_row: pd.Series,
    horizon_name: str,
    alpha: float,
) -> PredictionMetrics:
    target_column = f"target_{horizon_name}"
    model_df = feature_df.dropna(subset=FEATURE_COLUMNS + [target_column]).copy()
    split_index = int(len(model_df) * 0.8)
    train_df = model_df.iloc[:split_index]
    test_df = model_df.iloc[split_index:]

    feature_mean = train_df[FEATURE_COLUMNS].mean()
    feature_std = train_df[FEATURE_COLUMNS].std().replace(0.0, 1.0)

    x_train = (train_df[FEATURE_COLUMNS] - feature_mean) / feature_std
    x_test = (test_df[FEATURE_COLUMNS] - feature_mean) / feature_std
    x_latest = ((latest_feature_row[FEATURE_COLUMNS] - feature_mean) / feature_std).to_frame().T

    x_train = np.column_stack([np.ones(len(x_train)), x_train.values])
    x_test = np.column_stack([np.ones(len(x_test)), x_test.values])
    x_latest = np.column_stack([np.ones(len(x_latest)), x_latest.values])

    y_train = train_df[target_column].values
    y_test = test_df[target_column].values

    coefficients = fit_ridge_regression(x_train, y_train, alpha=alpha)
    test_predictions = x_test @ coefficients
    latest_prediction = float((x_latest @ coefficients)[0])

    mae = float(np.mean(np.abs(test_predictions - y_test)))
    rmse = float(math.sqrt(np.mean((test_predictions - y_test) ** 2)))
    direction_accuracy = float(np.mean((test_predictions > 0) == (y_test > 0)))
    predicted_price = float(latest_feature_row["close"] * (1 + latest_prediction))

    return PredictionMetrics(
        horizon_days=HORIZONS[horizon_name],
        predicted_return=latest_prediction,
        predicted_price=predicted_price,
        mae=mae,
        rmse=rmse,
        direction_accuracy=direction_accuracy,
    )


def classify_market_state(latest_row: pd.Series) -> dict[str, str]:
    if latest_row["rsi_14"] >= 70:
        rsi_state = "overbought"
    elif latest_row["rsi_14"] <= 30:
        rsi_state = "oversold"
    else:
        rsi_state = "neutral"

    if latest_row["close"] > latest_row["ema_50"] and latest_row["close"] > latest_row["ema_200"]:
        trend_state = "bullish across short and long trend filters"
    elif latest_row["close"] > latest_row["ema_50"]:
        trend_state = "short-term bullish, still below longer trend filters"
    elif latest_row["close"] < latest_row["ema_50"] and latest_row["close"] < latest_row["ema_200"]:
        trend_state = "bearish across short and long trend filters"
    else:
        trend_state = "mixed trend structure"

    if latest_row["volume_ratio_21d"] >= 1.10:
        volume_state = "above recent average volume"
    elif latest_row["volume_ratio_21d"] <= 0.90:
        volume_state = "below recent average volume"
    else:
        volume_state = "near recent average volume"

    return {
        "rsi_state": rsi_state,
        "trend_state": trend_state,
        "volume_state": volume_state,
    }


def create_report(symbol: str, history_range: str, alpha: float) -> dict:
    price_df = fetch_yahoo_chart(symbol=symbol, history_range=history_range)
    feature_df = build_feature_frame(price_df)
    latest_feature_row = feature_df.dropna(subset=FEATURE_COLUMNS).iloc[-1]
    market_state = classify_market_state(latest_feature_row)

    predictions = {}
    for horizon_name in HORIZONS:
        metrics = run_single_horizon_model(
            feature_df=feature_df,
            latest_feature_row=latest_feature_row,
            horizon_name=horizon_name,
            alpha=alpha,
        )
        predictions[horizon_name] = {
            "horizon_days": metrics.horizon_days,
            "predicted_return_pct": round(metrics.predicted_return * 100, 2),
            "predicted_price": round(metrics.predicted_price, 2),
            "test_mae_pct": round(metrics.mae * 100, 2),
            "test_rmse_pct": round(metrics.rmse * 100, 2),
            "test_direction_accuracy_pct": round(metrics.direction_accuracy * 100, 2),
        }

    return {
        "symbol": symbol,
        "history_range": history_range,
        "model": "ridge_regression_numpy",
        "alpha": alpha,
        "latest_market_date": latest_feature_row["date"].isoformat(),
        "latest_close": round(float(latest_feature_row["close"]), 2),
        "current_indicators": {
            "ema_50": round(float(latest_feature_row["ema_50"]), 2),
            "ema_100": round(float(latest_feature_row["ema_100"]), 2),
            "ema_200": round(float(latest_feature_row["ema_200"]), 2),
            "rsi_14": round(float(latest_feature_row["rsi_14"]), 2),
            "volume": int(float(latest_feature_row["volume"])),
            "volume_ratio_21d": round(float(latest_feature_row["volume_ratio_21d"]), 3),
            "volume_ratio_63d": round(float(latest_feature_row["volume_ratio_63d"]), 3),
        },
        "market_state": market_state,
        "predictions": predictions,
        "notes": [
            "This is a quantitative estimate, not investment advice.",
            "The Yahoo Finance chart endpoint is unofficial and may change.",
            "Long-horizon direction accuracy can look overstated because daily targets overlap.",
            "Index volume is a market-data proxy and may differ from cash-market stock volume measures.",
        ],
    }


def print_report(report: dict) -> None:
    print(f"Symbol: {report['symbol']}")
    print(f"Latest market date: {report['latest_market_date']}")
    print(f"Latest close: {report['latest_close']:.2f}")
    print(
        "EMA50 / EMA100 / EMA200: "
        f"{report['current_indicators']['ema_50']:.2f} / "
        f"{report['current_indicators']['ema_100']:.2f} / "
        f"{report['current_indicators']['ema_200']:.2f}"
    )
    print(
        "RSI14 / Volume ratio 21d: "
        f"{report['current_indicators']['rsi_14']:.2f} / "
        f"{report['current_indicators']['volume_ratio_21d']:.3f}"
    )
    print(
        "Market state: "
        f"{report['market_state']['trend_state']}; "
        f"RSI is {report['market_state']['rsi_state']}; "
        f"volume is {report['market_state']['volume_state']}."
    )
    print()

    for horizon_name, details in report["predictions"].items():
        readable_name = horizon_name.replace("_", " ")
        print(f"{readable_name.title()} prediction")
        print(f"  Horizon days: {details['horizon_days']}")
        print(f"  Predicted return: {details['predicted_return_pct']:.2f}%")
        print(f"  Predicted price: {details['predicted_price']:.2f}")
        print(f"  Holdout MAE: {details['test_mae_pct']:.2f}%")
        print(f"  Holdout RMSE: {details['test_rmse_pct']:.2f}%")
        print(f"  Holdout direction accuracy: {details['test_direction_accuracy_pct']:.2f}%")
        print()

    print("Notes")
    for note in report["notes"]:
        print(f"  - {note}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch live NIFTY 50 data, compute EMA/RSI/volume features, and "
            "produce 1-month and 1-year machine-learning price forecasts."
        )
    )
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Ticker symbol. Default: ^NSEI")
    parser.add_argument(
        "--history-range",
        default=DEFAULT_RANGE,
        help="Yahoo range string such as 5y, 10y, max. Default: 10y",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Ridge regularization strength. Default: 5.0",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the full report as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = create_report(
        symbol=args.symbol,
        history_range=args.history_range,
        alpha=args.alpha,
    )
    print_report(report)

    if args.output_json is not None:
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print()
        print(f"Saved JSON report to: {args.output_json}")


if __name__ == "__main__":
    main()
