import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import datetime as dt
import yfinance as yf
import talib

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import warnings
warnings.filterwarnings("ignore")

ticker = "GOOG"
end = dt.datetime.now()
start = end - dt.timedelta(days = 500)

df = yf.download(ticker,start = start, end=end)

class BollingerBand(Strategy):
    BB_n = 3

    def init(self):
        self.close = self.data.Close
        self.bb_upper, _, self.bb_lower = talib.BBANDS(self.close, timeperiod=self.BB_n, nbdevup=1, nbdevdn=1)

    def next(self):
        if (not self.position and
            self.close[-1] < self.bb_lower[-1]):
            self.buy()

        elif self.close[-1] > self.bb_upper[-1]:
            self.position.close()

bt = Backtest(df,BollingerBand,cash = 100000,commission = 0.05)
print(bt.run())

# opt_params = bt.optimize(BB_n=range(3,60,1))
# print(opt_params._strategy)

'''
The results of optimize is: BollingerBand(BB_n=3)
'''

def build_BollingerBand(data, window):
    data["SMA"] = data["Adj Close"].rolling(window=window).mean()
    data["deviation"] = data["Adj Close"].rolling(window=window).std()
    data["std_plus"] = data["SMA"] + 1 * data["deviation"]
    data["std_minus"] = data["SMA"] - 1 * data["deviation"]
    data["distance"] = data["Adj Close"] - data["SMA"]

    data["position"] = np.where(data["Adj Close"] > data["std_plus"], 0, np.nan)
    data["position"] = np.where(data["Adj Close"] < data["std_minus"], 1, data["position"])
    data["position"] = data["position"].ffill()
    data["b_s"] = data["position"].diff()
    data = data.fillna(0)
    return data


def vis_BollingerBand(signals):
    fig = plt.figure(figsize=(24, 16))
    sub = fig.add_subplot(111, ylabel="stock price")
    signals["Adj Close"].plot(ax=sub, color="blue")
    signals[["std_plus", "std_minus"]].plot(ax=sub, style=["--", "--"], color=["grey", "grey"], lw=0.8)
    sub.plot(signals[signals["b_s"] > 0].index, signals[signals["b_s"] > 0]["Adj Close"],
             "v", color="red", markersize=8)
    sub.plot(signals[signals["b_s"] < 0].index, signals[signals["b_s"] < 0]["Adj Close"],
             "^", color="green", markersize=8)

    plt.show()

window = 3

df_BollingerBand = build_BollingerBand(df,window)
vis_BollingerBand(df_BollingerBand)
print(df_BollingerBand)

capital = 100000
commission = 0.05

def calculate_strategy_return(signals, capital, commission):
    portfolio = pd.DataFrame(index=signals.index)

    portfolio["returns"] = signals["Adj Close"].pct_change()
    portfolio["holding"] = signals["position"].shift(1)
    portfolio["adj_returns"] = portfolio["returns"] * portfolio["holding"]
    portfolio.dropna(inplace=True)
    portfolio["BenchMark"] = (1 + portfolio["returns"]).cumprod() * capital
    portfolio["total"] = np.nan
    temp_value = capital
    for idx in portfolio.index:
        temp_value = ((1 + portfolio.loc[idx]["adj_returns"]) *
                      (temp_value - commission * abs(signals.loc[idx]["b_s"])))
        portfolio.loc[idx, "total"] = temp_value

    portfolio["str_ret"] = portfolio["total"].pct_change()
    portfolio["cum_ret"] = (1 + portfolio["returns"]).cumprod()
    portfolio["cum_str_ret"] = (1 + portfolio["str_ret"]).cumprod()
    portfolio["BM_max_perform"] = portfolio["cum_ret"].cummax()
    portfolio["str_max_perform"] = portfolio["cum_str_ret"].cummax()
    # portfolio.dropna(inplace = True)

    # BenchMark
    BM_final_account = portfolio["BenchMark"][-1]
    BM_cumulative_returns = portfolio["cum_ret"][-1]

    BM_average_return = portfolio["returns"].mean()
    BM_std_return = portfolio["returns"].std()
    BM_Sharpe_ratio = np.sqrt(253) * (BM_average_return / BM_std_return)

    BM_days = (portfolio.index[-1] - portfolio.index[0]).days
    BM_CAGR = (portfolio["BenchMark"][-1] / portfolio["BenchMark"][0]) ** (365 / BM_days) - 1

    BM_drawdown = portfolio["BM_max_perform"] - portfolio["cum_ret"]
    BM_max_drawdown = BM_drawdown.max()
    BM_drawdown_period = (BM_drawdown[BM_drawdown == 0].index[1:] - BM_drawdown[BM_drawdown == 0].index[:-1]).days
    BM_longest_drawdown_period = BM_drawdown_period.max()

    # Strategy
    strategy_final_account = portfolio["total"][-1]
    cumulative_returns = portfolio["cum_str_ret"][-1]

    average_daily_return = portfolio["str_ret"].mean()
    daily_std = portfolio["str_ret"].std()
    Sharpe_ratio = np.sqrt(253) * (average_daily_return / daily_std)

    days = (portfolio.index[-1] - portfolio.index[0]).days
    CAGR = (portfolio["total"][-1] / portfolio["total"][0]) ** (365 / days) - 1

    drawdown = portfolio["str_max_perform"] - portfolio["cum_str_ret"]
    max_drawdown = drawdown.max()
    drawdown_period = (drawdown[drawdown == 0].index[1:] - drawdown[drawdown == 0].index[:-1]).days
    longest_drawdown_period = drawdown_period.max()

    plt.figure(figsize=(16, 10))
    plt.plot(portfolio.index, portfolio["cum_ret"], label="BenchMark CumulativeReturns", color="green")
    plt.plot(portfolio.index, portfolio["cum_str_ret"], label="Strategy CumulativeReturns", color="blue")
    plt.legend()
    plt.show()

    print(f"The performance of BenchMark is:")
    print(f"    Sharpe ratio:            {BM_Sharpe_ratio}")
    print(f"    CAGR:                    {BM_CAGR}")
    print(f"    Max drawdown:            {BM_max_drawdown}")
    print(f"    Longest drawdown period: {BM_longest_drawdown_period} days")
    print(f"    Finanl account:          {BM_final_account}")

    print("================================================================")
    print("================================================================")

    print(f"The performance of Strategy is:")

    print(f"    Sharpe ratio:            {Sharpe_ratio}")
    print(f"    CAGR:                    {CAGR}")
    print(f"    Max drawdown:            {max_drawdown}")
    print(f"    Longest drawdown period: {longest_drawdown_period} days")
    print(f"    Finanl account:          {strategy_final_account}")

    return portfolio


df_strategy = calculate_strategy_return(df_BollingerBand, capital, commission)
df_strategy