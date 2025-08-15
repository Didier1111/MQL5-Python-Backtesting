import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class AbsorptionVolumeProfile:
    def __init__(self, file_path_or_data, max_window: int = 400):
        # Accept either a list-of-lists (history) or a CSV path / DataFrame-like
        if isinstance(file_path_or_data, (list, tuple)) and len(file_path_or_data) > 0 and isinstance(file_path_or_data[0], (list, tuple)):
            self.df = pd.DataFrame(file_path_or_data, columns=("time", "open", "high", "low", "close", "tick_volume", "pos"))
        elif isinstance(file_path_or_data, pd.DataFrame):
            self.df = file_path_or_data.copy(deep=True)
        else:
            # Assume it is a CSV-like path
            self.df = pd.read_csv(file_path_or_data)
            # Try to normalize expected columns if needed
            expected_cols = {"time", "open", "high", "low", "close", "tick_volume"}
            if not expected_cols.issubset(set(map(str.lower, self.df.columns))):
                raise ValueError("Input data must have columns: time, open, high, low, close, tick_volume")
            self.df.columns = [c.lower() for c in self.df.columns]
            if "pos" not in self.df.columns:
                self.df["pos"] = "NA"

        # Enforce types
        self.df["time"] = pd.to_datetime(self.df["time"]) if not np.issubdtype(self.df["time"].dtype, np.datetime64) else self.df["time"]
        for c in ["open", "high", "low", "close"]:
            self.df[c] = self.df[c].astype(float)
        if "tick_volume" in self.df.columns:
            self.df["tick_volume"] = pd.to_numeric(self.df["tick_volume"], errors="coerce").fillna(0).astype(float)
        else:
            self.df["tick_volume"] = 0.0
        if "pos" not in self.df.columns:
            self.df["pos"] = "NA"

        # Use only the most recent window for profile computations
        if max_window is not None and max_window > 0:
            self.df = self.df.tail(max_window).reset_index(drop=True)

    # ---------------------- Utility calculations ----------------------
    def _compute_atr(self, period: int = 14) -> pd.Series:
        high = self.df["high"]
        low = self.df["low"]
        close = self.df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=period).mean()
        return atr

    def _nearest_round_levels(self, price: float, step: float = 0.005) -> list:
        # Round-number style levels (e.g., 00/50): step=0.005 ~ 50 pips in EURUSD terms
        base = round(price / step) * step
        return sorted(list({round(base - step, 6), round(base, 6), round(base + step, 6)}))

    def _build_volume_profile(self, window: int = 200, bin_size: float = 0.0002) -> pd.DataFrame:
        d = self.df.tail(window)
        lo = float(d["low"].min())
        hi = float(d["high"].max())
        if hi <= lo:
            hi = lo + bin_size
        # Define bin edges
        num_bins = int(math.ceil((hi - lo) / bin_size)) + 1
        edges = np.array([lo + i * bin_size for i in range(num_bins + 1)])
        vol = np.zeros(num_bins, dtype=float)

        # Distribute each candle's tick_volume uniformly across the bins it spans
        for _, row in d.iterrows():
            clow, chigh = float(row["low"]), float(row["high"])
            if math.isnan(clow) or math.isnan(chigh):
                continue
            if chigh < clow:
                clow, chigh = chigh, clow
            start_idx = max(0, int(math.floor((clow - lo) / bin_size)))
            end_idx = min(num_bins - 1, int(math.floor((chigh - lo) / bin_size)))
            if end_idx < start_idx:
                continue
            span = end_idx - start_idx + 1
            if span <= 0:
                continue
            add = float(row["tick_volume"]) / span if span > 0 else 0.0
            vol[start_idx:end_idx + 1] += add

        centers = (edges[:-1] + edges[1:]) / 2.0
        profile = pd.DataFrame({"price": centers, "volume": vol})
        return profile

    def _is_hvn_at_level(self, profile: pd.DataFrame, level: float, top_pct: float = 0.1) -> bool:
        if profile.empty:
            return False
        # Find nearest bin to level
        idx = (profile["price" ] - level).abs().idxmin()
        level_vol = profile.loc[idx, "volume"]
        if level_vol <= 0:
            return False
        threshold = profile["volume"].quantile(1 - top_pct)
        return level_vol >= threshold

    def _count_level_tests(self, level: float, tolerance: float = 0.0002, lookback: int = 50) -> int:
        d = self.df.tail(lookback)
        touched = ((d["high"] >= level - tolerance) & (d["low"] <= level + tolerance)).sum()
        return int(touched)

    def _recent_failure_candle(self, level: float, side: str, atr: pd.Series, tol: float = 0.0001, vol_q: float = 0.8, lookback: int = 30) -> bool:
        # side: "resistance" or "support"
        d = self.df.tail(lookback).copy(deep=True)
        if d.empty:
            return False
        d["atr"] = atr.reindex(d.index)
        d["body"] = (d["close"] - d["open"]).abs()
        d["range"] = (d["high"] - d["low"]).abs()
        vol_threshold = d["tick_volume"].quantile(vol_q)

        for _, r in d[::-1].iterrows():  # iterate from most recent backward
            if side == "resistance":
                # Tested above level but failed to close above
                touched = r["high"] >= level - tol
                failed_close = r["close"] <= level + tol
            else:  # support
                touched = r["low"] <= level + tol
                failed_close = r["close"] >= level - tol

            if not touched:
                continue
            high_vol = r["tick_volume"] >= vol_threshold
            small_progress = r["body"] <= max(1e-9, 0.5 * (r["atr"] if not math.isnan(r["atr"]) else d["range"].mean()))
            wickish = r["range"] > 0 and (r["body"] / r["range"]) <= 0.4
            if high_vol and failed_close and (small_progress or wickish):
                return True
        return False

    # ---------------------- Public API ----------------------
    def detect_absorption(self,
                          window: int = 200,
                          bin_size: float = 0.0002,
                          hvn_top_pct: float = 0.1,
                          level_step: float = 0.005,
                          tests_lookback: int = 50,
                          failure_lookback: int = 30) -> dict:
        if len(self.df) < max(50, window // 2):
            return {"absorbed": False, "side": None, "level": None, "details": {}}

        last_price = float(self.df["close"].iloc[-1])
        levels = self._nearest_round_levels(last_price, step=level_step)
        atr = self._compute_atr(period=14)
        profile = self._build_volume_profile(window=window, bin_size=bin_size)

        best = {"absorbed": False, "side": None, "level": None, "details": {}}
        for level in levels:
            # Check both resistance and support contexts
            # Resistance context: price near/under level
            res_checks = 0
            if self._is_hvn_at_level(profile, level, top_pct=hvn_top_pct):
                res_checks += 1
            if self._count_level_tests(level, tolerance=bin_size, lookback=tests_lookback) >= 2:
                res_checks += 1
            if self._recent_failure_candle(level, side="resistance", atr=atr, tol=bin_size, lookback=failure_lookback):
                res_checks += 1
            if res_checks >= 2:
                best = {"absorbed": True, "side": "resistance", "level": level,
                        "details": {"hvn": True, "tests": True, "failure": True}}
                break

            sup_checks = 0
            if self._is_hvn_at_level(profile, level, top_pct=hvn_top_pct):
                sup_checks += 1
            if self._count_level_tests(level, tolerance=bin_size, lookback=tests_lookback) >= 2:
                sup_checks += 1
            if self._recent_failure_candle(level, side="support", atr=atr, tol=bin_size, lookback=failure_lookback):
                sup_checks += 1
            if sup_checks >= 2:
                best = {"absorbed": True, "side": "support", "level": level,
                        "details": {"hvn": True, "tests": True, "failure": True}}
                break

        return best

    def run(self):
        # Return a tuple similar to other strategies: (signal, meta), df
        result = self.detect_absorption()
        # Map to signal: if resistance absorption (bearish) -> -1, if support absorption (bullish) -> 1, else 0
        signal = 0
        if result.get("absorbed"):
            signal = -1 if result.get("side") == "resistance" else 1
        # For parity with other strategies that return (signal, extra)
        return (signal, result), self.df