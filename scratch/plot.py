# scratch/plot.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import timedelta

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
csv_path = Path(__file__).parent.parent / "outputs" / "backtest_results" / "nav_series.csv"
df = pd.read_csv(csv_path, parse_dates=["date"])
df.set_index("date", inplace=True)

# ----------------------------------------------------------------------
# 2. Filter data: start 120 days after the first date
# ----------------------------------------------------------------------
first_date = df.index.min()
start_date = first_date + timedelta(days=1)
df_filtered = df[df.index >= start_date]

if df_filtered.empty:
    raise ValueError(f"No data after {start_date}. The CSV ends on {df.index.max()}.")

print(f"Original data: {first_date.date()} to {df.index.max().date()}")
print(f"Filtered data: {start_date.date()} to {df_filtered.index.max().date()}")

# ----------------------------------------------------------------------
# 3. Create figure with two vertically stacked panels
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle("Portfolio Dashboard", fontsize=14, fontweight="bold")

# ----- Panel 1: NAV -----
ax1.plot(df_filtered.index, df_filtered["nav"], color="#1f77b4", linewidth=1.5)
ax1.set_ylabel("NAV ($)", fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

# ----- Panel 2: Drawdown from peak -----
drawdown_pct = df_filtered["drawdown_from_peak"] * 100
ax2.fill_between(df_filtered.index, 0, drawdown_pct, color="#d62728", alpha=0.3)
ax2.plot(df_filtered.index, drawdown_pct, color="#d62728", linewidth=1.5)
ax2.set_ylabel("Drawdown (%)", fontsize=10)
ax2.set_xlabel("Date", fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

# Format x‑axis dates
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

# ----------------------------------------------------------------------
# 4. Save and show
# ----------------------------------------------------------------------
plt.tight_layout()
output_path = Path(__file__).parent / "portfolio_dashboard_filtered.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Chart saved to {output_path}")

plt.show()