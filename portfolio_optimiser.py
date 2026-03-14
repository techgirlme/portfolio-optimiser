import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

# ── Configuration ─────────────────────────────────────────────────────────────
TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS",
           "INFY.NS", "WIPRO.NS", "ASIANPAINT.NS"]
PERIOD = "3y"
RF_RATE = 0.065   # Indian risk-free rate (approx 10yr govt bond yield)
N_SIMS = 10000

# ── Data ──────────────────────────────────────────────────────────────────────
print("Fetching data...")
prices = yf.download(TICKERS, period=PERIOD, auto_adjust=True)["Close"]
returns = prices.pct_change().dropna()
names = [t.replace(".NS", "") for t in TICKERS]

mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252
n_assets = len(TICKERS)

print(f"  Loaded {len(returns)} trading days of data")
print(f"  Assets: {', '.join(names)}\n")

# ── Monte Carlo Simulation ─────────────────────────────────────────────────────
np.random.seed(42)
results = np.zeros((3, N_SIMS))
weights_store = []

for i in range(N_SIMS):
    w = np.random.dirichlet(np.ones(n_assets))
    weights_store.append(w)

    port_return = np.dot(w, mean_returns)
    port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    sharpe = (port_return - RF_RATE) / port_vol

    results[0, i] = port_return
    results[1, i] = port_vol
    results[2, i] = sharpe

results_df = pd.DataFrame(
    results.T, columns=["Return", "Volatility", "Sharpe"])
weights_df = pd.DataFrame(weights_store, columns=names)

# ── Optimal Portfolios ─────────────────────────────────────────────────────────
# Maximum Sharpe Ratio
max_sharpe_idx = results_df["Sharpe"].idxmax()
max_sharpe = results_df.iloc[max_sharpe_idx]
max_sharpe_w = weights_df.iloc[max_sharpe_idx]

# Minimum Volatility
min_vol_idx = results_df["Volatility"].idxmin()
min_vol = results_df.iloc[min_vol_idx]
min_vol_w = weights_df.iloc[min_vol_idx]

print("=" * 50)
print("  OPTIMAL PORTFOLIO — Maximum Sharpe Ratio")
print("=" * 50)
for stock, weight in zip(names, max_sharpe_w):
    print(f"  {stock:<15} {weight:.1%}")
print(f"\n  Expected Annual Return : {max_sharpe['Return']:.2%}")
print(f"  Annual Volatility      : {max_sharpe['Volatility']:.2%}")
print(f"  Sharpe Ratio           : {max_sharpe['Sharpe']:.3f}")

print("\n" + "=" * 50)
print("  MINIMUM VARIANCE PORTFOLIO")
print("=" * 50)
for stock, weight in zip(names, min_vol_w):
    print(f"  {stock:<15} {weight:.1%}")
print(f"\n  Expected Annual Return : {min_vol['Return']:.2%}")
print(f"  Annual Volatility      : {min_vol['Volatility']:.2%}")
print(f"  Sharpe Ratio           : {min_vol['Sharpe']:.3f}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Efficient frontier scatter
sc = ax1.scatter(
    results_df["Volatility"], results_df["Return"],
    c=results_df["Sharpe"], cmap="RdYlGn",
    alpha=0.5, s=8
)
plt.colorbar(sc, ax=ax1, label="Sharpe Ratio")

ax1.scatter(max_sharpe["Volatility"], max_sharpe["Return"],
            color="#1D9E75", s=200, zorder=5, marker="*",
            label=f"Max Sharpe ({max_sharpe['Sharpe']:.2f})")
ax1.scatter(min_vol["Volatility"], min_vol["Return"],
            color="#7F77DD", s=200, zorder=5, marker="D",
            label=f"Min Volatility ({min_vol['Volatility']:.2%})")

ax1.set_title("Efficient Frontier — NSE Portfolio",
              fontsize=13, fontweight="bold")
ax1.set_xlabel("Annual Volatility (Risk)")
ax1.set_ylabel("Annual Return")
ax1.legend(fontsize=9)

# Optimal weights bar chart
colors = ["#1D9E75", "#7F77DD", "#BA7517", "#D85A30", "#378ADD", "#534AB7"]
bars = ax2.bar(names, max_sharpe_w * 100, color=colors,
               edgecolor="white", linewidth=0.8)
ax2.set_title("Optimal Portfolio Weights (Max Sharpe)",
              fontsize=13, fontweight="bold")
ax2.set_ylabel("Weight (%)")
ax2.set_ylim(0, max(max_sharpe_w * 100) * 1.2)

for bar, val in zip(bars, max_sharpe_w * 100):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.suptitle("Modern Portfolio Theory — Markowitz Optimisation",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "efficient_frontier.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()
print("\nChart saved as efficient_frontier.png")
