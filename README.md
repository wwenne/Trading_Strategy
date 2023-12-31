## Trading Strategy

- These trading strategies are self-practice and to enhance my understanding about the different kinds of trading systems.

### Table of Content
- [Results of the Strategies](#results)
  - [The results tabel of strategies](#results-table)
  - [The explanation of different strategies](#explanation-of-strategies)

- [FAQ](#faq)
    - [What is my Trading Strategy?](#what-is-my-trading-strategy)
    - [What includes these strategies?](#what-includes-a-strategy)
    - [How to test a strategy?](#how-to-test-a-strategy)
    - [How to create/optimize a strategy?](#how-to-optimize-a-strategy)

## Results

We assume that the initial capital is 10000 dollors and the invest period is 500 days, also each time the trading commission fee is 5 dollors. The based on the strategy, we get the results by executing our own strategy.

### Results Table

##### Technical Analysis

|                                                           Strategy                                                            | Shapre ratio | CAGR % | MAX net-drawdown % | Longest drawdown | Final Account | Buy&Hold Account |
|:-----------------------------------------------------------------------------------------------------------------------------:|:------------:|:------:|:------------------:|:----------------:|:-------------:|:----------------:|
|         [01 TA RSI_MACD](https://github.com/wwenne/Trading_Strategy/blob/main/Technical_Analysis/01_TA_RSI%26MACD.py)         |    0.9007    | 28.28  |       20.05        |       142        |   140558.63   |    116838.24     |           
| [02 TA MeanReversion_SMA](https://github.com/wwenne/Trading_Strategy/blob/main/Technical_Analysis/02_TA_MeanReversion_SMA.py) |    0.8695    | 28.79  |       69.12        |        50        |   140834.71   |    120102.93     |
| [03 TA BollingerBand](https://github.com/wwenne/Trading_Strategy/blob/main/Technical_Analysis/03_TA_BollingerBand.py) |    1.209     | 26.09  |       18.54        |       162        |   137034.66   |    116124.94     |

##### Machine Learning -- Supervised Learning

|                                                           Strategy                                                           | Shapre ratio | CAGR % | MAX net-drawdown % | Longest drawdown | Final Account | Buy&Hold Account |
|:----------------------------------------------------------------------------------------------------------------------------:|:------------:|:------:|:------------------:|:----------------:|:-------------:|:----------------:|
| [01 Forex Trading](https://github.com/wwenne/Trading_Strategy/blob/main/Trading_Using_SupervisedLearning/01_ForexTrading.py) |    2.0370    | 15.57  |       0.0244       |        48        |   103114.04   |     98351.30     |



### Explanation of Strategies
##### Technical Analysis
_01 TA MACD_RSI_
- Target asset: "GOOG"
- Method: Technical Analysis 
- Indicators: `MACD & RSI`

_02 TA MeanReversion SMA_
- Target asset: "AMZN"
- Method: Technical Analysis 
- Indicators: `SMA`(Mean Reversion)

_03 TA Bollinger Band_
- Target asset: "GOOG"
- Method: Technical Analysis 
- Indicators: `Bollinger Band`

##### Machine Learning -- Supervised Learning
_01 Forex Trading_
- Target asset: GBP/USD forex price
- Method: Supervised Learning 
- Model: `Ridge`




## FAQ

### What is my Trading Strategy?

**Based on `technical analysis`:**
- Use technical indicators such as **SMA, EMA, RSI, MACD, Stochastic Oscillator, ROC, Standard Deviation** etc. and also select several indicators together at the same time to build our trading strategy.

**Using `Machine Learning`:**
- `Supervised Learning`
  - Use technical indicators and correlated assets as well as  input features and apply different algorithms to train and test the output feature such as returns, signals and directions of the price. Based on the evaluation and performance of different models, pick the best model to predict output feature in the future.
  - Algorithms: **Logistic Regression, Lasso, Ridge, K-Nearest Neighbors, Decision Tree, Random Forest, Support Vector Machine** and so on.
- `Unsupervised Learning`
  - Use Principle Component Analysis(PCA), Clustering.


### What includes a strategy?

Each Strategies includes:  

- [x] **Buy signals**: Using technical indicators or other functions to flag buy signals.
- [x] **Sell signals**: Using technical indicators or other functions to flag sell signals.
- [x] **Indicators**: Includes the indicators required to run the strategy.

### How to execute a strategy?

- use trading signals to form our trading positions and based on these trading positions, we execute our buy and sell signals in the next trading day, using `shift` function.

### How to test a strategy?

- calculate the results of some performance matrix including Sharpe ratio, max drawdown, longest drawdown period, total cumulative returns and the final account if using real capital to invest in the target asset and execute this strategy.

### How to optimize a strategy?

- Use `Backtest` library to test the overall performance
- Use `optimize` function to select the best number of indicators or parameters.

