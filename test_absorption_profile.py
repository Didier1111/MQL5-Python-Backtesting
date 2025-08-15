import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from trading_strategies.absorption_volume_profile import AbsorptionVolumeProfile


def generate_synthetic_eurusd(n=600, start=1.1000):
	# Simple random walk with periodic spikes in tick volume near round numbers
	times = [datetime.utcnow() - timedelta(minutes=n - i) for i in range(n)]
	prices = [start]
	for i in range(1, n):
		step = np.random.normal(0, 0.0002)
		prices.append(max(0.5, prices[-1] + step))
	opens, highs, lows, closes, vols = [], [], [], [], []
	for i in range(n):
		p = prices[i]
		oc = p + np.random.normal(0, 0.00005)
		h = max(p, oc) + abs(np.random.normal(0, 0.00015))
		l = min(p, oc) - abs(np.random.normal(0, 0.00015))
		# add vol spike near round-number steps of 0.005
		lvl = round(round(p / 0.005) * 0.005, 6)
		dist = abs(p - lvl)
		base_vol = np.random.randint(50, 200)
		spike = 400 if dist < 0.0002 else 0
		vol = base_vol + spike
		opens.append(p)
		highs.append(h)
		lows.append(l)
		closes.append(oc)
		vols.append(vol)
	data = {
		"time": times,
		"open": opens,
		"high": highs,
		"low": lows,
		"close": closes,
		"tick_volume": vols,
		"pos": ["NA"] * n,
	}
	return pd.DataFrame(data)


def main():
	csv_path = os.environ.get("EURUSD_CSV", "")
	if csv_path and os.path.isfile(csv_path):
		print(f"Loading CSV: {csv_path}")
		df = pd.read_csv(csv_path)
	else:
		print("No CSV provided. Generating synthetic data...")
		df = generate_synthetic_eurusd()

	model = AbsorptionVolumeProfile(df)
	(signal, details), used_df = model.run()
	print("Signal:", signal)
	print("Details:", details)
	print("Last price:", float(used_df["close"].iloc[-1]))


if __name__ == "__main__":
	main()