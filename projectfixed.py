
import bt
from pandas_datareader import data
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import norm
from random import choice
from random import shuffle



# using the function below to download yahoo finance data by using tickers and start/end date


def MR_get(tickers, s, e):
    stock_data = data.DataReader(tickers, data_source='yahoo', start=s,
                                 end=e)['Adj Close']

    return stock_data


# adding the tickers of the companies in the portfolio
tickers = ['GOOG', 'SPOK',  'TM', 'F', 'WMT', 'HEINY', 'XOM', 'BKR', 'JPM', 'BCS', 'JNJ', 'A',
           'NEE', 'PPL', 'HON', 'RSG', 'AAPL', 'HPQ', 'BHP', 'PKX', 'GLD', 'TLT', 'GSG', 'IEI']




# adding weights to our portfolio and defining it as an equal weighted portfolio
asset_weights = pd.Series(index=tickers, dtype=float)
asset_weights[tickers] = 1/len(tickers)

start_date = '2008-6-27'
end_date = '2020-12-31'

# Risk free rate from treasury.gov
rfr = 0.0007

## Benchmark
bm_ticker = ['^GSPC']
bm_weights = pd.Series(index=tickers, dtype=float)
bm_weights[tickers] = 1/len(tickers)


# running the function above to get the data on the selected tickers and removing any missing values
companies = MR_get(tickers, start_date, end_date)
companies = companies.dropna()


equities = ['GOOG', 'SPOK',  'TM', 'F', 'WMT', 'HEINY', 'XOM', 'BKR', 'JPM', 'BCS', 'JNJ', 'A',
           'NEE', 'PPL', 'HON', 'RSG', 'AAPL', 'HPQ', 'BHP', 'PKX']

equity_weight = pd.Series(index=equities, dtype=float)
equity_weight[equities] = 1/len(equities)


equities = MR_get(equities, start_date, end_date)

equity_return = equities.pct_change().dropna()

equity_portfolio = equity_return.dot(equity_weight)


## getting S&P500 data

bm = MR_get(bm_ticker, start_date, end_date)

bm = bm.pct_change()
bm = bm.dropna()

bm_mean = bm.mean()
bm_std = bm.std()

bm_annualized_mean =(1+bm_mean)**252-1
bm_annualized_std = bm_std*np.sqrt(252)

bm_cumulative_return = (1+bm).cumprod()

bm_sharpe = (bm_annualized_mean-rfr)/bm_annualized_std

confidence_level = 0.05
bm_var = norm.ppf(confidence_level, bm_annualized_mean, bm_annualized_std)

## getting Ray Dailo All Weather portfolio data
rdawstart = '2015-1-1'
rdawend = '2021-1-1'

rdawtic = ['TLT', 'IEI', 'VTI', 'GSG', 'GLD']
rdaw_w = [0.4, 0.15, 0.30, 0.075, 0.075]

rdaw = MR_get(rdawtic, rdawstart, rdawend)

rdaw_ret = rdaw.pct_change()
rdaw_ret = rdaw_ret.dropna()

rdaw_mean = rdaw_ret.mean()
rdaw_std = rdaw_ret.std()

rdaw_pf = rdaw_ret.dot(rdaw_w)

rdaw_pf_mean = rdaw_pf.mean()
rdaw_pf_std = rdaw_pf.std()

annu_rdaw_pf_mean = (1+rdaw_pf_mean)**252-1
annu_rdaw_pf_std = rdaw_pf_std*np.sqrt(252)

rdaw_pf_sharpe = (annu_rdaw_pf_mean - rfr) / annu_rdaw_pf_std



# CALCULATIONS ON DATA
# calculating the returns on the tickers using .pct_change() and removes the missing values
returns = companies.pct_change()
returns = returns.dropna()

# calulating the mean of the returns on each company and the standard deviation
ret_mean = returns.mean()
annualized_return = (1+ret_mean)**252-1
ret_std = returns.std()
annualized_std = ret_std*np.sqrt(252)
cumulative_ret = (1 + returns).cumprod()


# calculating portfolio returns using .dot multiplication
pf_ret = returns.dot(asset_weights)



# calcluating portfolio mean return and standard deviation
pf_ret_mean = pf_ret.mean()
pf_ret_std = pf_ret.std()



# annualizing the portfolio mean return and standard deviation
annu_pfret_mean = ((1+pf_ret_mean)**252)-1
annu_pfret_std = pf_ret_std*np.sqrt(252)
print(annu_pfret_mean)
print(annu_pfret_std)

# Calculating cumulative return on the portfolio
pf_cumulative_ret = (1 + pf_ret).cumprod()

# calculating the sharpe ration using the annualized data
sharpe = (annu_pfret_mean - rfr)/annu_pfret_std

#Sharpe each assets
asset_sharpe = []

for i in tickers:
    asset_sharpe.append((annualized_return[i]-rfr)/annualized_std[i])

asset_sharpe = pd.DataFrame(asset_sharpe, index = tickers)
print(asset_sharpe)

plt.figure()
plt.plot(pf_cumulative_ret)
plt.plot(bm_cumulative_return)
plt.title('Cumulative returns of portfolio vs benchmark')
plt.xlabel('Multiplier')
plt.ylabel('Period')
plt.legend()
plt.show()

# # Optimizing the portfolio
# from pypfopt.efficient_frontier import EfficientFrontier
# from pypfopt import risk_models
# from pypfopt import expected_returns

# mu = ret_mean*252
# sigma = risk_models.sample_cov(companies)

#ef = EfficientFrontier(mu, sigma)
# weights = ef.max_sharpe()
# cleaned_weights = ef.clean_weights()
# print(cleaned_weights)
# ef.portfolio_performance(verbose=True)

# #plotting the weights and efficient frontier
# pypfopt.plotting.plot_weights(cleaned_weights)

def randomNumbers(list_of_weights, ticker, constraints, remaining):
    """
    This function will generate a random set of numbers between specified
    constraints which sum up to a specificied total.
    
    Parameters:
    
    list_of_weights = list of weights with tickers as index
    
    ticker = acts as index --- SET TO 0
    
    constraints = list of two floats to act as your constraints.
                    - 1st float is minimum constraint
                    - 2nd float is maximum constraint
    
    remaining = acts as the sum of random numbers --- SET TO 1
    """
    next_ticker = ticker + 1
    
    if next_ticker == len(list_of_weights):
        
        list_of_weights[ticker] = remaining                        
        
    else:
        
        remaining_tickers = len(list_of_weights) - next_ticker   
    
        remaining_lowerbound = constraints[0] * remaining_tickers  
        
        remaining_upperbound = constraints[1] * remaining_tickers 
        
        actual_lowerbound = max(constraints[0], remaining - remaining_upperbound)
        
        actual_upperbound = min(constraints[1], remaining - remaining_lowerbound)
        
        random_weight = np.random.uniform(low  = actual_lowerbound, 
                                          high = actual_upperbound)

        list_of_weights[ticker] = random_weight
        
        randomNumbers(list_of_weights, 
                      next_ticker, 
                      [actual_lowerbound, actual_upperbound],
                      (remaining - random_weight))




# Create empty dataframes
portfolios         = pd.DataFrame()
portfolios_weights = pd.DataFrame()

# Number of assets
num_of_assets = len(tickers)

# Instantiate covariance matrix
cov_mat = returns.cov()

# Number of iterations
n_iterations = 1000

# Risk free rate
rfr = 0



## Monte Carlo Loop to build portfolio simulations

# Loop over each iteration
for i in range(n_iterations):
    
    # Create placeholder for weights
    weights = pd.Series(index = tickers, dtype = float)
    
    # Generate a random weight between 0-1 with constraints
    randomNumbers(list_of_weights = weights,     # list of weights with ticker index
                  ticker          = 0,              # index of of first ticker
                  constraints     = [0.000001,0.999],    # minimum & maximum constraints
                  remaining       = 1)              # sum of all random numbers 
    
    # Shuffle weights
    shuffle(weights)
    
    # Loop over each ticker to create weights
    for ticker in tickers:
            
        # Assign each weight to ticker in both portfolios & portfolios_weights
        portfolios_weights.loc[i, ticker] = weights[ticker]
        
        portfolios.loc[i, ticker] = weights[ticker]
    
    
    ## Calculate portfolio returns
    
    # Calculate weighted returns
    weighted_returns = returns.mul(portfolios_weights.iloc[i,:], axis = 1)
    
    # Calculate expected returns per portfolio
    portfolios.loc[i, 'expected_return'] = \
        weighted_returns.sum(axis = 1).mean()
    
    # Calculate portfolio volatility
    portfolios.loc[i, 'volatility'] = \
        np.sqrt(np.dot(portfolios_weights.loc[i, :].T, 
                   np.dot(cov_mat, portfolios_weights.loc[i, :])))
    
    # Calculate Sharpe
    portfolios.loc[i, "sharpe_ratio"] = \
        (portfolios.loc[i, "expected_return"] - rfr) / \
         portfolios.loc[i, 'volatility']

 
# Identify max Sharpe Ratio & minimum Variance portfolio

max_sharpe_filter     = portfolios['sharpe_ratio'].idxmax()
min_volatility_filter = portfolios['volatility'].idxmin()

max_sharpe_portfolio     = portfolios.iloc[max_sharpe_filter, :]
min_volatility_portfolio = portfolios.iloc[min_volatility_filter, :]

print(f"""
     Max Sharpe Portfolio: 
-------------------------------
Index: {max_sharpe_filter}

{max_sharpe_portfolio}



   Min Volatility Portfolio: 
-------------------------------
Index: {min_volatility_filter}

{min_volatility_portfolio}""")



# Plot cumulative returns of minimum volatility portfolio

# Get weighted_returns
weighted_returns = returns.mul(portfolios_weights.iloc[328,:], axis = 1)


# Calculate total returns
total_returns    = weighted_returns.sum(axis = 1)


# Calculate cumulative returns
cumulative_returns = ((1 + total_returns).cumprod()-1)



# Plot cumulative returns

# setting figure size
fig, ax = plt.subplots(figsize = (15, 8))


# histogram for returns
sb.lineplot(data  = cumulative_returns)


# title
plt.title(label    = "Cumulative Returns of Minimum Volatility Portfolio",
          fontsize = 20)


# y-label
plt.ylabel(ylabel = 'Cumulative Returns', fontsize = 15)


# Compile and display
plt.tight_layout()
plt.show()

# Plot expected returns from sim

expected_returns = portfolios['expected_return']

# setting figure size
fig, ax = plt.subplots(figsize = (15, 8))

# histogram for returns
sb.histplot(data  = expected_returns, # print portfolio returns
             bins  = 'fd',             # number of bins ('fd' = Freedman-Diaconis Rule) 
             kde   = True,             # kernel density plot (line graph)
             alpha = 0.3,              # transparency of colors
             stat  = 'density')        # can be set to 'count', 'frequency', or 'probability'


# title
plt.title(label    = "Distribution of Simulated Portfolios' Expected Returns",
          fontsize = 20)


# x-label
plt.xlabel(xlabel = 'Expected Returns')


# y-label
plt.ylabel(ylabel = 'Density')


# Compile and display
plt.tight_layout()
plt.show()


# Corr matrix
correlation_matrix = returns.corr()
sb.heatmap(correlation_matrix)
plt.show()


# Hist returns
plt.hist(pf_ret, bins=100)
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.show()


# Calculating Drawdown
# Calculating the running maximum
running_max = np.maximum.accumulate(pf_cumulative_ret)
running_max[running_max < 1] = 1

# Calculating the drawdown by deviding the cumulative return on the portfolio on the running maximum and subtracting 1
drawdown = (pf_cumulative_ret) / running_max - 1

# Plot the drawdown
plt.plot(drawdown)
plt.show()


# VaR calculations on portfolio
# Set VaR level
var_level = 95

# VaR calculation
var_95 = np.percentile(pf_ret, 100 - var_level)

# Calculating conditional value at risk
cvar = pf_ret[pf_ret <= var_95].mean()

# Parametric VaR
mu = pf_ret.mean()
sigma = pf_ret.std()
confidence_level = 0.05
VaR = norm.ppf(confidence_level, mu, sigma)

# estimate n-day VaR
n_days = 5
estimated_var95_ndays = VaR*np.sqrt(n_days)

# VaR on all the assets using parametric VaR
var_asset_list = []
var_all_asset = norm.ppf(confidence_level, ret_mean, ret_std)
var_asset_list.append(var_all_asset)
var_asset_list = pd.DataFrame(np.transpose(var_asset_list), index = tickers)

# VaR of all assets using actual returns. gives you the value of risk of the historical data
VaR_all_assets = []
for i in returns:
    i = np.percentile(returns[i], 100-var_level)
    VaR_all_assets.append(i)

VaR_all_assets = pd.DataFrame(VaR_all_assets, index = tickers)

print(var_asset_list)
print(VaR_all_assets)

diff_var = abs(var_asset_list) - abs(VaR_all_assets)

# VaR on portfolio
# Monte Carlo simulation of Value at Risk using standard normal assumption

# Runs
monte_carlo_runs = 10000

# Simulated
days_to_simulate = 252

# Hits
bad_simulations = 0

total_simulations = 0

loss_cutoff = 0.98

mu = returns.mean()
sigma = returns.std()

compounded_returns = sigma.copy()

total_simulations = 0
bad_simulations = 0

for run_counter in range(0,monte_carlo_runs):
    for i in tickers:
        compounded_temp = 1
        for simulated_day_counter in range(0,days_to_simulate):
            simulated_return = np.random.normal(mu[i],sigma[i],1)
            compounded_temp = compounded_temp * (1+simulated_return)
        compounded_returns[i] = compounded_temp
    portfolio_return = compounded_returns.dot(asset_weights)
    
    if (portfolio_return<loss_cutoff):
        bad_simulations = bad_simulations + 1
    total_simulations = total_simulations + 1
    
print("Your portfolio will lose",round((1-loss_cutoff)*100,3),"%",
      "over",days_to_simulate,"days", 
      bad_simulations/total_simulations, "of the time")


# Historcal bootstrapping
compounded_returns_bs = sigma.copy()
total_simulations = 0
bad_simulations = 0
for run_counter in range(0,monte_carlo_runs): # Loop over runs    
    for i in tickers:
        compounded_temp_bs = 1
        for simulated_day_counter in range(0,days_to_simulate): # loop over days
            simulated_return_bs = choice(returns[i])
            compounded_temp_bs = compounded_temp_bs * (1 + simulated_return_bs)        
        compounded_returns_bs[i]=compounded_temp_bs
    portfolio_return_bs = compounded_returns_bs.dot(asset_weights) # dot product
    if(portfolio_return_bs<loss_cutoff):
        bad_simulations = bad_simulations + 1
    total_simulations = total_simulations + 1

print("Your portfolio will lose",round((1-loss_cutoff)*100,3),"%",
      "over",days_to_simulate,"days", 
      bad_simulations/total_simulations, "of the time")


# Plot Returns + VaR

# setting figure size
fig, ax = plt.subplots(figsize = (13, 5))

# histogram for returns
sb.histplot(data  = pf_ret,  # data set - index Facebook (or AAPL or GOOG)
             bins  = 'fd',          # number of bins ('fd' = Freedman-Diaconis Rule) 
             kde   = True,          # kernel density plot (line graph)
             alpha = 0.2,           # transparency of colors
             stat  = 'count')     # can be set to 'count', 'frequency', or 'probability'



plt.title(label = "Distribution of Portfolio Returns")
plt.xlabel(xlabel = 'Returns')
plt.ylabel(ylabel = 'Count')


# VaR at 95% 
VaR_95 = np.percentile(pf_ret, 5)


# Line
plt.axvline(x         = VaR_95,         # x-axis location
            color     = 'r',            # line color
            linestyle = '--')           # line style


# Linelabel
plt.text(VaR_95,                         # x-axis location
         30,                             # y-axis location
         'VaR',                          # text
         horizontalalignment = 'right',  # alignment ('center' | 'left')
         fontsize = 'x-large')           # fontsize


plt.tight_layout()
plt.show()

# Regression of portfolio returns against the fama french factors
import pandas_datareader as pdr
from pandas_datareader.famafrench import get_available_datasets

datasets = get_available_datasets()

ff = pdr.data.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start=start_date, end=end_date)
ff = pd.DataFrame(ff[0])
ff = ff.iloc[1:]

excess_ret = ff['Mkt-RF']
smb = ff['SMB']
hml = ff['HML']
rf = ff['RF']

reg_pf_ret = pf_ret*100

reg_data=pd.DataFrame(np.transpose([reg_pf_ret, excess_ret, smb, hml, rf]), columns= ['reg_pf_ret', 'excess_ret', 'smb', 'hml', 'rf'])

plt.scatter(pf_ret, reg_data['excess_ret'], reg_data['smb'], reg_data['hml'])
plt.show()


model = smf.ols('reg_pf_ret ~ excess_ret + smb + hml + rf', data=reg_data).fit()
print(model.summary())

""""
regression of the portfolio returns was not significant. 
The factors doesn't have coefficients above 0.008, with an r squared of 0.92, and smb p-value is 0.035

Therefore im going to try it on the equities alone to see if it makes a difference.

"""
equity_portfolio = equity_portfolio*100

reg_data2 = pd.DataFrame(np.transpose([equity_portfolio, excess_ret, smb, hml, rf]), columns= ['equity_portfolio', 'excess_ret', 'smb', 'hml', 'rf'])

plt.scatter(equity_portfolio, reg_data2['excess_ret'], reg_data2['smb'], reg_data2['hml'])
plt.show()

model = smf.ols('equity_portfolio ~ excess_ret + smb + hml + rf', data=reg_data2).fit()
print(model.summary())

"""
Results are that the R Squared increase by 0.01, but the coefficients are still not good'
"""


rfr = 0.0007

allweatherweights = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
                       0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.075, 0.40, 0.075, 0.15]

allweatherpf = returns.dot(allweatherweights)

allweatherpf_mean = allweatherpf.mean()
allweatherpf_std = allweatherpf.std()

awpf_annumean = (1+allweatherpf_mean)**252-1
awpf_annustd = allweatherpf_std*np.sqrt(252)

awpf_sharpe = (awpf_annumean - rfr)/awpf_annustd

aw_var = norm.ppf(confidence_level, allweatherpf_mean, allweatherpf_std)


optimizedweights = max_sharpe_portfolio
optimizedweights = optimizedweights[:-3]
optimizedweights = [0.05435, 0.00064, 0.000001, 0.000002, 0.000001, 0.001236, 0.000001, 0.000009, 0.001032, 0.000002, 0.034660, 0.000024, 0.000467, 0.000001, 0.069429, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.00001, 0.021936, 0.00001, 0.865116]

optimized_pf = returns.dot(optimizedweights)

optimized_mean = optimized_pf.mean()
optimized_std = optimized_pf.std()

optimized_annumean = (1+optimized_mean)**252-1
optimized_annustd = optimized_std*np.sqrt(252)

optimized_sharpe = (optimized_annumean - rfr)/optimized_annustd

optimized_var = norm.ppf(confidence_level, optimized_annumean, optimized_annustd)

optimizedminvol = min_volatility_portfolio[:-3]
optimizedminvol = [0.000002, 0.000008, 0.05428, 0.000001, 0.000001, 0.00024, 0.000001, 0.003685, 0.000699, 0.000483, 0.000001, 0.000001, 0.011383, 0.000010, 0.000116,  0.016059, 0.001754, 0.000001, 0.000001, 0.000001, 0.000001, 0.000207, 0.002491, 0.908618]

minvolpf = returns.dot(optimizedminvol)

minvol_mean = minvolpf.mean()
minvol_std = minvolpf.std()

minvol_annumean = (1+minvol_mean)**252-1
minvol_annustd = (minvol_std)*np.sqrt(252)

minvol_sharpe = (minvol_annumean - rfr) / minvol_annustd


bm = bm.pct_change()
bm = bm.dropna()

bm_mean = bm.mean()
bm_std = bm.std()

bm_annualized_mean =(1+bm_mean)**252-1
bm_annualized_std = bm_std*np.sqrt(252)

bm_cumulative_return = (1+bm).cumprod()

bm_sharpe = (bm_annualized_mean-rfr)/bm_annualized_std

bm_var = norm.ppf(confidence_level, bm_annualized_mean, bm_annualized_std)

max_sharpe_portfolio.to_excel(r'C:\Users\adler\OneDrive\Skrivebord\Hult International Business School\Modelling and Analytics\Project\maxsharpeweights.xlsx')
min_volatility_portfolio.to_excel(r'C:\Users\adler\OneDrive\Skrivebord\Hult International Business School\Modelling and Analytics\Project\minvolweights.xlsx')




# Backtesting over entire data horizon
# Download the portfolio from the bt package in order to use in in the test
start_date = '2018-01-01'

data = bt.get('goog,spok,tm,f,wmt,heiny,xom,bkr,jpm,bcs,jnj,a,nee,ppl,hon,rsg,aapl,hpq,bhp,pkx,gld,tlt,gsg,iei', start=start_date)

# Defines the strategy where we run on transaction on all assets that are weighted equally and rebalanced
s = bt.Strategy('Equally Weighted', [bt.algos.RunOnce(), 
                       bt.algos.SelectAll(), 
                       bt.algos.WeighEqually(), 
                       bt.algos.Rebalance()])

# Test the strategy on the dataset
test = bt.Backtest(s, data)

# Run the test
res = bt.run(test)

# Plot the test
res.plot()

# Display the results
res.display()


## Optimized portfolio
# Getting the assets from the optimized portfolio
data3 = data

# Testing optimized portfolio
optimized_weights = pd.Series(optimizedweights, index = data3.columns)

strategy = bt.Strategy('Optimized', [bt.algos.RunOnce(), 
                                     bt.algos.SelectAll(), 
                                     bt.algos.WeighSpecified(**optimized_weights), 
                                     bt.algos.Rebalance()])

test3 = bt.Backtest(strategy, data3, integer_positions=False)
res3 = bt.run(test3)
res3.plot()
res3.display()




"""

Backtesting the benchmarks which is S&P and the actual all weahter portfolio

"""


sp500 = bt.get('^gspc', start=start_date)

rdaw_data = bt.get('tlt,iei,vti,gsg,gld', start=start_date)
rdaw_weights_lev = pd.Series([0.4,0.15,0.5,0.075,0.075], index = rdaw_data.columns)


sp_strat= bt.Strategy('S&P500', [bt.algos.RunOnce(),
                                 bt.algos.SelectAll(),
                                 bt.algos.WeighEqually(),
                                 bt.algos.Rebalance()])

rdaw_strat = bt.Strategy('Ray Dailo All Weather levered', [bt.algos.RunOnce(),
                                                           bt.algos.SelectAll(),
                                                           bt.algos.WeighSpecified(**rdaw_weights_lev),
                                                           bt.algos.Rebalance()])

sp_test = bt.Backtest(sp_strat, sp500)
rdaw_test = bt.Backtest(rdaw_strat, rdaw_data, integer_positions=False)

benchmark = bt.run(sp_test, rdaw_test)
benchmark.plot()

sp_res = bt.run(sp_test)
rdaw_res = bt.run(rdaw_test)

sp_res.display()
rdaw_res.display()




"""
lever the portfolio > total weight = 1.2
Found the explaination of randomly selecting stocks
weird as it was not random. probably need to change the explaination of
how we decided on the portfolio. 

"""

eqvsbm = bt.run(test,sp_test, rdaw_test)
eqvsbm.plot()

optvsbm = bt.run(test3, sp_test, rdaw_test)
optvsbm.plot()