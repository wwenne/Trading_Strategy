import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import re
import time
import datetime as dt
import yfinance as yf
import talib

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import warnings
warnings.filterwarnings("ignore")

ticker = "GOOG"
end = dt.datetime.now()
start = end - dt.timedelta(days = 500)

df = yf.download(ticker,start = start, end=end)

def RSI(array,n):
    return talib.RSI(array,n)


class MCADRSI(Strategy):
    lower_bound = 30
    upper_bound = 70
    RSI_n = 14

    short_MA = 12
    long_MA = 26
    signal_MA = 9

    def init(self):
        self.close = self.data["Close"]
        self.rsi = self.I(RSI, self.close, self.RSI_n)
        self.macd, self.signal_macd, _ = talib.MACD(self.close, fastperiod=self.short_MA,
                                                    slowperiod=self.long_MA, signalperiod=self.signal_MA)
    def next(self):
        if (self.rsi[-1] < self.lower_bound and self.macd[-1] > self.signal_macd[-1]):
            self.buy()
        elif (self.rsi[-1] > self.upper_bound and self.macd[-1] < self.signal_macd[-1]):
            self.position.close()

bt = Backtest(df,MCADRSI,cash = 100000,commission = 0.05)
print(bt.run())


# opt_params= bt.optimize(short_MA=range(5, 20, 1),long_MA=range(21, 40, 1),signal_MA=range(5, 15, 1),
#                         RSI_n =range(7,21,1))
# print(opt_params._strategy)
'''
The results of optimize is: MCADRSI(short_MA=5,long_MA=21,signal_MA=5,RSI_n=7)
'''

lower_thres = 30
upper_thres = 70
RSI_n = 7
short_MA = 5
long_MA = 21
signal_MA = 5

capital = 100000
commission = 5

def build_MACD_RSI(df, lower_thres, upper_thres,RSI_n,
                   short_MA, long_MA, signal_MA):
    # MACD Calculation
    short_ma = df["Adj Close"].ewm(span=short_MA, min_periods=1).mean()
    long_ma = df["Adj Close"].ewm(span=long_MA, min_periods=1).mean()
    df["MACD_line"] = short_ma - long_ma
    df["MACD_signal_line"] = df["MACD_line"].ewm(span=signal_MA, min_periods=1).mean()
    df["MACD_hist"] = df["MACD_line"] - df["MACD_signal_line"]
    df["MACD_signal"] = np.nan

    # RSI Calculation
    gain = df["Adj Close"].diff()
    loss = -gain.clip(lower=0)
    gain = gain.clip(upper=0)
    avg_gain = gain.rolling(window=RSI_n, min_periods=1).mean()
    avg_loss = loss.rolling(window=RSI_n, min_periods=1).mean()
    df["RS"] = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + df["RS"]))

    # Trading Signals
    df["position"] = np.where((df["MACD_line"] > df["MACD_signal_line"]) & (df["RSI"] < lower_thres), 1, np.nan)
    df["position"] = np.where((df["MACD_line"] < df["MACD_signal_line"]) & (df["RSI"] > upper_thres), -1,
                              df["position"])
    df["position"] = df["position"].ffill()

    df["b_s"] = df["position"].diff()

    return df

def view_MACD_RSI(signals):
    fig = plt.figure(figsize=(18, 10), dpi=300)
    gs = plt.GridSpec(3, 1, height_ratios=[30, 12, 12])
    # Create a 2x1 grid with specified height ratios

    ax1 = plt.subplot(gs[0], ylabel="Price")
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax3 = plt.subplot(gs[2], sharex=ax1)

    ax1.plot(signals.index, signals["Adj Close"], label="Price", color="blue")
    ax1.plot(signals[signals["b_s"] > 0].index, signals[signals["b_s"] > 0]['Adj Close'],
             "g^", markersize=8)
    ax1.plot(signals[signals["b_s"] < 0].index, signals[signals["b_s"] < 0]["Adj Close"],
             "rv", markersize=8)
    ax1.set_title("Price and trade signals with MACD")
    ax1.set_ylabel("Price")
    # view MACD
    ax2.plot(signals.index, signals["MACD_line"], label='MACD Line', color='red')
    ax2.plot(signals.index, signals['MACD_signal_line'], label='Signal Line', color='green')
    ax2.bar(signals.index, signals['MACD_hist'], label='MACD Histogram', color='gray')
    ax2.set_title("MACD Plot")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("MACD")
    ax2.legend()

    # view RSI
    signals["RSI"].plot(ax=ax3, color="purple", lw=0.8)
    ax3.axhline(30, linestyle="--", alpha=0.5, color="grey")
    ax3.axhline(70, linestyle="--", alpha=0.5, color="grey")
    ax3.set_title("RSI Plot")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("RSI")
    ax3.legend()

    plt.tight_layout()
    plt.show()

def calculate_strategy_return(signals, capital, commission):
    portfolio = pd.DataFrame(index=signals.index)

    portfolio["returns"] = signals["Adj Close"].pct_change()
    portfolio["holding"] = signals["position"]
    portfolio["adj_returns"] = portfolio["returns"] * portfolio["holding"]
    portfolio.dropna(inplace=True)
    portfolio["SPY"] = (1 + portfolio["returns"]).cumprod() * capital
    portfolio["total"] = np.nan
    temp_value = capital
    for idx in portfolio.index:
        temp_value = ((1 + portfolio.loc[idx]["adj_returns"]) * temp_value)
        portfolio.loc[idx, "total"] = temp_value

    portfolio["str_ret"] = portfolio["total"].pct_change()
    portfolio["cum_ret"] = (1 + portfolio["returns"]).cumprod()
    portfolio["cum_str_ret"] = (1 + portfolio["str_ret"]).cumprod()
    portfolio["SPY_max_perform"] = portfolio["cum_ret"].cummax()
    portfolio["str_max_perform"] = portfolio["cum_str_ret"].cummax()
    # portfolio.dropna(inplace = True)
    # SPY
    SPY_final_account = portfolio["SPY"][-1]
    SPY_cumulative_returns = portfolio["cum_ret"][-1]

    SPY_average_return = portfolio["returns"].mean()
    SPY_std_return = portfolio["returns"].std()
    SPY_Sharpe_ratio = np.sqrt(253) * (SPY_average_return / SPY_std_return)

    SPY_days = (portfolio.index[-1] - portfolio.index[0]).days
    SPY_CAGR = (portfolio["SPY"][-1] / portfolio["SPY"][0]) ** (365 / SPY_days) - 1

    SPY_drawdown = portfolio["SPY_max_perform"] - portfolio["cum_ret"]
    SPY_max_drawdown = SPY_drawdown.max()
    SPY_drawdown_period = (SPY_drawdown[SPY_drawdown == 0].index[1:] - SPY_drawdown[SPY_drawdown == 0].index[:-1]).days
    SPY_longest_drawdown_period = SPY_drawdown_period.max()
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
    plt.plot(portfolio.index, portfolio["cum_ret"], label="SPY CumulativeReturns", color="green")
    plt.plot(portfolio.index, portfolio["cum_str_ret"], label="Strategy CumulativeReturns", color="blue")
    plt.legend()
    plt.show()

    print(f"The performance of SPY is:")
    print(f"    Sharpe ratio:            {SPY_Sharpe_ratio}")
    print(f"    CAGR:                    {SPY_CAGR}")
    print(f"    Max drawdown:            {SPY_max_drawdown}")
    print(f"    Longest drawdown period: {SPY_longest_drawdown_period} days")
    print(f"    Final account:           {SPY_final_account}")

    print("================================================================")
    print("================================================================")

    #     print(f"The Strategy is Short SMA = {short_ma} Long SMA = {long_ma}")
    print(f"The performance of Strategy is:")

    print(f"    Sharpe ratio:            {Sharpe_ratio}")
    print(f"    CAGR:                    {CAGR}")
    print(f"    Max drawdown:            {max_drawdown}")
    print(f"    Longest drawdown period: {longest_drawdown_period} days")
    print(f"    Finanl account:          {strategy_final_account}")

    return portfolio

df_MACD_RSI = build_MACD_RSI(df, lower_thres, upper_thres,RSI_n,
                   short_MA, long_MA, signal_MA)
print(view_MACD_RSI(df_MACD_RSI))


df_strategy = calculate_strategy_return(df_MACD_RSI,capital,commission)
