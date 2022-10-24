# import statements
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
from pandas_datareader.data import DataReader

# header
print("****************************************")
print("Analyzing Stocks using Statistics")
print("****************************************")
print()

# input statements
start_date = input("Enter Start Date (yyyy-mm-dd): ")
end_date = input("Enter End Date (yyyy-mm-dd): ")
print()
stock = input("Enter Ticker: ")
print()

# create stocks using adjusted close
stock = DataReader(stock, 'yahoo', start_date, end_date)['Adj Close']
SPY = DataReader('SPY', 'yahoo', start_date, end_date)['Adj Close']

def metrics():
    # mean daily return
    x = stock.pct_change().dropna()
    # Volatility = standard deviation * sqrt(time period)
    a = x.std() * np.sqrt(252)
    astr = str('{:.5g}'.format(a * 100))
    print("Volatility: " + astr + "%")
    print()
    # 95% Value at Risk (VaR) = PPF (inverse CDF)
    b = stats.norm.ppf(1-0.95, x.mean(), x.std())
    bstr = str('{:.5g}'.format(b * 100))
    print("95% Value at Risk (VaR): " + bstr + "%")
    print()
    # Sharpe Ratio = (mean / standard deviation) * sqrt(time period)
    c = (x.mean() / x.std()) * np.sqrt(252)
    cstr = str('{:.5g}'.format(c))
    print("Sharpe Ratio: " + cstr)
    print()
    # Downside Deviation = sqrt(sum(all negative numbers ^2) / length)
    d = np.sqrt(sum(x[x<0]**2) / len(x))
    dstr = str('{:.5g}'.format(d * 100))
    print("Downside Deviation: " + dstr + "%")
    print()
    # Maximum Drawdown = (Trough Value - Peak Value) / Peak Value
    tv = np.argmax(np.maximum.accumulate(stock) - stock) 
    pv = np.argmax(stock[:tv])
    e = (stock[tv] - stock[pv]) / stock[pv]
    estr = str('{:.5g}'.format(e * 100))
    print("Maximum Drawdown: " + estr + "%")
    print()
metrics()

def capm():
    # mean daily returns
    x = stock.pct_change().dropna()
    y = SPY.pct_change().dropna()
    # beta = slope of linear regression
    regr = stats.linregress(y,x)
    beta = regr[0]
    betastr = str('{:.5g}'.format(beta))
    print("Beta: " + betastr)
    print()
    # alpha = return - (beta * market return)
    xn = (stock[len(stock)-1] - stock[0]) / stock[0]
    yn = (SPY[len(SPY)-1] - SPY[0]) / SPY[0] 
    alpha = xn - (beta * yn)
    alphastr = str('{:.5g}'.format(alpha * 100))
    print("Alpha: " + alphastr + "%")
    print()
capm()


