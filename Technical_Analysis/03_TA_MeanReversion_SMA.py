'''
- Build, execute and backtest a mean-reversion strategy.
- Use a simple moving average of x days ==> select best x using optimize function
- Set a threshold value at two standard deviations of the current price, to deviate from the simple moving average to signal a positioning.

'''



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime as dt
import yfinance as yf
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA

import warnings
warnings.filterwarnings("ignore")

ticker = "AMZN"
end = dt.datetime.now()
start = end - dt.timedelta(days = 500)
df = yf.download(ticker,start = start, end=end)

def SMA(array, n):
    return pd.Series(array).rolling(n).mean()

def Deviate(array,n):
    return pd.Series(array).rolling(n).std()

class MeanReversionSMA(Strategy):
    l_SMA = 15
    s_SMA = 5

    def init(self):
        self.close = self.data.Close
        self.l_sma = self.I(SMA, self.close, self.l_SMA)
        self.s_sma = self.I(SMA, self.close, self.s_SMA)
        distance = self.I(Deviate,self.close,self.s_SMA)
        self.lower_threshold = self.s_sma - 2 * distance
        self.upper_threshold = self.s_sma + 2 * distance

    def next(self):
        if (not self.position and
            self.s_sma[-1] > self.l_sma[-1] and
            self.close[-1] < self.lower_threshold[-1]):
            self.buy()
        elif (self.s_sma[-1] < self.l_sma[-1] and
              self.close[-1] > self.upper_threshold[-1]):
            self.position.close()


backtest = Backtest(df, MeanReversionSMA, cash=100000, commission=0.05, exclusive_orders=True)
print(backtest.run())


# opt_stats = backtest.optimize(l_SMA=range(15,40,1),
#                               s_SMA=range(3,12,1))
# print(opt_stats._strategy)
'''
The best result of this is strategy is MeanReversionSMA(l_SMA=15,s_SMA=4)
'''

l_SMA = 15
s_SMA = 4


def build_MeanReversion_SMA(df, s_SMA, l_SMA):
    df["long_SMA"] = SMA(df["Adj Close"], l_SMA)
    df["short_SMA"] = SMA(df["Adj Close"], s_SMA)
    df["distance"] = Deviate(df["short_SMA"],s_SMA)
    df["upper_bound"] = df["short_SMA"] + 2*df["distance"]
    df["lower_bound"] = df["short_SMA"] - 2*df["distance"]
    df["position"] = np.where((df["Adj Close"] > df["upper_bound"])&(df["short_SMA"]<df["long_SMA"]), -1, np.nan)
    df["position"] = np.where((df["Adj Close"] < df["lower_bound"])&(df["short_SMA"]<df["long_SMA"]), 1, df["position"])
    # df["position"] = np.where(df["distance"] * df["distance"].shift(1) < 0,0,df["position"])
    df["position"] = df["position"].ffill()
    df["b_s"] = df["position"].diff()
    df = df.fillna(0)
    return df


def vis_MeanReversion_SMA(signals):
    signals[["Adj Close", "short_SMA", "upper_bound", "lower_bound"]].plot(figsize=(30, 15),
                                                                           color=["grey", "green", "black", "black"],
                                                                           style=["-", ".", "--", "--"])
    signals[signals["b_s"] > 0]["short_SMA"].plot(style="o", markersize=5, color="red")
    signals[signals["b_s"] < 0]["short_SMA"].plot(style="o", markersize=5, color="blue")


df_MeanReversion_SMA = build_MeanReversion_SMA(df,s_SMA, l_SMA)
vis_MeanReversion_SMA(df_MeanReversion_SMA)

print(df_MeanReversion_SMA)


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


df_strategy = calculate_strategy_return(df_MeanReversion_SMA, capital, commission)
df_strategy

