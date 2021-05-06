import pandas as pd
import scipy.stats
import numpy as np

def drawdown(return_series: pd.Series):
    """
    Takes timeseries of asset returns and computes:
    wealth index
    previous peaks
    percent drawdowns
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown =(wealth_index - previous_peaks )/previous_peaks
    return pd.DataFrame({"Wealth": wealth_index,
                        "Previous Peak": previous_peaks,
                        "Drawdown": drawdown})

def get_ffme_returns():
    returns = pd.read_csv("Data/Portfolios_Formed_on_ME_monthly_EW.csv",
                     header=0,
                     index_col=0,
                     parse_dates=True,
                     na_values=-99.99)
    returns = returns[['Lo 10','Hi 10']]
    returns.columns = ['Small Cap','Large Cap']
    returns.index = pd.to_datetime(returns.index,format='%Y%m')
    returns.index = returns.index.to_period('M')
    returns = (returns)/100
    return returns

def get_hfi_returns():
    """
    Get edhec risk hedge fund index returns
    """
    returns = pd.read_csv("Data/edhec-hedgefundindices.csv",
                     header=0,
                     index_col=0,
                     parse_dates=True,
                     na_values=-99.99)
    returns.index = returns.index.to_period('M')
    returns = (returns)/100
    return returns



def get_ind_file(filetype):
    """
    Load and format the Ken French 30 Industry Portfolios files
    """    
    known_filetypes = ["size","returns","nfirms"]

    if filetype not in known_filetypes:
        raise ValueError(f"File type needs to be in: {','.join(known_filetypes)}")

    if filetype == "returns":
        filetype = "vw_rets"
        divisor = 100
        
    if filetype == "size":
        filetype = "size"
        divisor = 1
        
    if filetype == "nfirms":
        filetype = "nfirms"
        divisor = 1

    results = pd.read_csv(f"Data/ind30_m_{filetype}.csv",header=0,index_col=0,parse_dates=True)
        
    results.index = pd.to_datetime(results.index,format="%Y%m")
    results.index = results.index.to_period('M')
    results.columns = results.columns.str.strip()
        
    return results/divisor
        
def get_ind_size(): 
    """
    Load and format the Ken French 30 Industry Portfolios Average size (market cap)
    """    
    return get_ind_file("size")

def get_ind_nfirms(): 
    """
    Load and format the Ken French 30 Industry Portfolios Average number of Firms
    """    
    return get_ind_file("nfirms")

def get_ind_returns(): 
    """
    Load and format the Ken French 30 Industry Portfolios Value Weighted Monthly Returns
    """    
    return get_ind_file("returns")

def get_total_market_index_returns():
    """
    Load the 30 industry portfolio data and derive the returns of a capweighted total market index
    """   
    ind_returns = get_ind_returns()
    nfirms = get_ind_nfirms()
    size = get_ind_size()
    ind_cap = nfirms*size
    mkt_cap = ind_cap.sum(axis="columns")
    cap_weights = ind_cap.divide(mkt_cap,axis="rows")
    total_cap_weighted_returns = (cap_weights * ind_returns).sum(axis="columns") 
    return total_cap_weighted_returns



def skewness(r):
    """
    Alternative to scipy.stats.skew)()
    """
    
    excess_return = r - r.mean()
    sigma_r = r.std(ddof=0)
    return ((excess_return**3).mean())/(sigma_r**3)

def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    """
    
    excess_return = r - r.mean()
    sigma_r = r.std(ddof=0)
    return ((excess_return**4).mean())/(sigma_r**4)

def is_normal(r,level=0.01):
    """
    Applies Jarque-Bera test 
    r is a Series or a DataFrame
    level is level of confidence; default is 1%
    returns true if normality hypothesis is accepted;false if rejected.
    """
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level

def semideviation(r):
    """
    Returns semideviation i.e. -ve semideviation of r
    r is a Series or a DataFrame
    """
    is_negative = r<0
    
    return r[is_negative].std(ddof=0.01)

def var_historic(r, level=5):
    """
    VaR Historic
    r is a Series or a DataFrame
    level is optional;default is 5%
    """
    if isinstance(r,pd.DataFrame):
        return r.aggregate(var_historic,level=level)
    elif isinstance(r,pd.Series):
        return -np.percentile(r,level)
    else:
        raise TypeError("Expect r to be Series or DataFrame")
        
def var_gaussian(r,level=5,modified=False):
    """
    VaR assuming a Gaussian distriution
    r is a Series or a DataFrame
    level is optional;default is 5%
    modified - if True it means z parameter from normal z; False indicates it needs been modified 
    """
    z = scipy.stats.norm.ppf(level/100)
    if modified:
        excess_return = r - r.mean()
        sigma_r = r.std(ddof=0)
        kurtosis= ((excess_return**4).mean())/(sigma_r**4)
        skewness = ((excess_return**3).mean())/(sigma_r**3)
        z = (z + (z**2 -1)*skewness/6 
                  + (z**3 -3*z)*(kurtosis-3)/24
                  - (2*(z**3) - 5*z)*(skewness**2)/36 )
    return -(r.mean() + (z*r.std(ddof=0)))

def cvar(r,level=5):
    """
    Returns conditional VaR of a Series or Dataframe
    """
    if isinstance(r,pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar, level=level)
    else:
        raise TypeError("Expected r to be a Series or a DataFrame")
        

def annualized_returns(r,periods_per_year):
    """
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods) - 1 


def annualized_volatility(r,periods_per_year):
    """
    """
    
    return r.std() * np.sqrt(periods_per_year)

def sharpe_ratio(r,risk_free_rate,periods_per_year):
    """
    """
    rf_per_period = (1+risk_free_rate)**(1/r.shape[0]) - 1
    excess_returns = r - rf_per_period
    ann_ex_ret = annualized_returns(excess_returns,periods_per_year)
    return ann_ex_ret/annualized_volatility(r,periods_per_year)
    
    
def portfolio_return(weights, returns):
    """
    Weights --> returns
    """
    
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    """
    Weigths -- > vol
    """
    return (weights.T @ covmat @ weights)**0.5

def plt_ef2(n_points,er,cov, style=".-"):
    if er.shape[0] != 2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")
    weights = [np.array([w,1-w]) for w in np.linspace(0,1,n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vol = [portfolio_vol(w,cov) for w in weights]
    ef= pd.DataFrame({"Returns":rets,"Volatility":vol})
    return ef.plot.line(x="Volatility",y="Returns",style=style)

from scipy.optimize import minimize

def minimize_volatility(target_return,er,covr):
    """
    Target return --> Weight vector
    """
    
    n=er.shape[0]
    init_guess = np.repeat(1/n,n)
    bounds = ((0.0,1.0),)*n
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun' : lambda weights, er: target_return -  portfolio_return(weights, er)
    }
    
    
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1 
    }
    results = minimize(portfolio_vol, init_guess,
                      args=(covr,),method="SLSQP",
                      options={'disp': False},
                      constraints=(return_is_target,weights_sum_to_1),
                      bounds=bounds
                      )
    return results.x

def optimal_weights(n_points,er,covr):
    """
    Generates a list of weights to run the optimizer on
    """
    target_rs = np.linspace(er.min(),er.max(),n_points)
    weights = [minimize_volatility(target_return,er,covr) for target_return in target_rs]
    return weights

def gmv(covr):
    """
    Returns weigths of global minimum variance portfolio
    """
    n = covr.shape[0]
    return msr(0,np.repeat(1,n),covr)

def plt_efn(n_points,er,covr, style=".-",show_cml=True,risk_free_rate=0,show_ew=False,show_gmv=False):
    """
    Plot N asset efficient frontier 
    """
    weights = optimal_weights(n_points,er,covr)
    rets = [portfolio_return(w, er) for w in weights]
    vol = [portfolio_vol(w,covr) for w in weights]
    ef= pd.DataFrame({"Returns":rets,"Volatility":vol})
    ax = ef.plot.line(x="Volatility",y="Returns",style=style)
    
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n,n)
        r_ew = portfolio_return(w_ew,er)
        v_ew = portfolio_vol(w_ew,covr)
        ax.plot([v_ew],[r_ew],color="goldenrod",marker="o",markersize=10)
        
    if show_gmv:
        n = er.shape[0]
        w_gmv = gmv(covr)
        r_gmv = portfolio_return(w_gmv,er)
        v_gmv = portfolio_vol(w_gmv,covr)
        ax.plot([v_gmv],[r_gmv],color="midnightblue",marker="o",markersize=10)
        
    if show_cml:
        w_msr = msr(risk_free_rate,er,covr)
        r_msr = portfolio_return(w_msr,er)
        v_msr = portfolio_vol(w_msr,covr)
        cml_x = [0,v_msr]
        cml_y = [risk_free_rate,r_msr]
    return ax.plot(cml_x, cml_y, color="green",marker="o",linestyle="dashed",markersize=12,linewidth=2)
    
def msr(risk_free_rate,er,covr):
    """
    Risk free rate along with expected returns and cov matrix --> max sharpe ration
    """
    n=er.shape[0]
    init_guess = np.repeat(1/n,n)
    bounds = ((0.0,1.0),)*n
    
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1 
    }
    
def negative_sharpe_ratio(weights,er,covr,risk_free_rate):
    r = portfolio_return(weights,er)
    v = portfolio_vol(weights,covr)
    return  (risk_free_rate - r)/v

    results = minimize(negative_sharpe_ratio, init_guess,
                      args=(er,covr,risk_free_rate,),method="SLSQP",
                      options={'disp': False},
                      constraints=(weights_sum_to_1),
                      bounds=bounds
                      )
    return results.x

def run_cppi(risky_r, safe_r=None, m=3, start=100, floor=0.8, riskfree_rate=0.03, drawdown=None):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History
    """
    # set up the CPPI parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start*floor
    peak = account_value
    if isinstance(risky_r, pd.Series): 
        risky_r = pd.DataFrame(risky_r, columns=["R"])

    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/12 # fast way to set all values to a number
    # set up some DataFrames for saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    floorval_history = pd.DataFrame().reindex_like(risky_r)
    peak_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak*(1-drawdown)
        cushion = (account_value - floor_value)/account_value
        risky_w = m*cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1-risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w
        # recompute the new account value at the end of this step
        account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])
        # save the histories for analysis and plotting
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
        floorval_history.iloc[step] = floor_value
        peak_history.iloc[step] = peak
    
    risky_wealth = start*(1+risky_r).cumprod()
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth, 
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r":risky_r,
        "safe_r": safe_r,
        "drawdown": drawdown,
        "peak": peak_history,
        "floor": floorval_history
    }
    return backtest_result


def summary_stats(r):
    annualized_ret = r.aggregate(annualized_returns,periods_per_year=12)
    annualized_vol = r.aggregate(annualized_volatility,periods_per_year=12)
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var = r.aggregate(var_gaussian,modified=True)
    h_var = r.aggregate(var_historic)
    sharpe_r = r.aggregate(sharpe_ratio,risk_free_rate=0.03,periods_per_year=12)
    dd = r.aggregate(lambda r : drawdown(r).Drawdown.min())
    return pd.DataFrame({
        "Annualized Return": annualized_ret,
        "Annualized Volatity": annualized_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR": cf_var,
        "Historic cVaR": h_var,
        "Sharpe Ratio": sharpe_r,
        "Max Drawdown": dd
    })

def gbm(n_years=10,n_scenarios=1000,mu = 0.07, sigma = 0.15,steps_per_year=12, s_0=100.0, prices=True):
    """
    Evolution of a stock price using a Geomtric Brownian motion model
    """
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
#    rets_plus_1 = np.random.normal(loc=(1+mu)**(dt),scale=sigma*np.sqrt(dt),size=(n_steps, n_scenarios))
    rets_plus_1 = np.random.normal(loc=(1+mu)**dt,scale=(sigma*np.sqrt(dt)),size=(n_steps,n_scenarios))
#    rets_plus_1 = pd.DataFrame(rets_plus_1)
    rets_plus_1[0] = 1
#    price = s_0*(pd.DataFrame(rets_plus_1)).cumprod()
    ret_val = s_0*(pd.DataFrame(rets_plus_1).cumprod()) if prices else pd.DataFrame(rets_plus_1) - 1
    return ret_val

def discount(t,r):
    
    """
    Compute the price of a pure discount bond that pays a dollar at time period t
    and r is the per-period interest rate
    returns a |t| x |r| Series or DataFrame
    r can be a float, Series or DataFrame
    returns a DataFrame indexed by t
    """
    
    discounts=pd.DataFrame([(r+1)**-i for i in t])
    discounts.index = t
    return discounts

def pv(flows,r):
    """
    Compute the present value of a sequence of cash flows given by the time (as an index) and amounts
    r can be a scalar, or a Series or DataFrame with the number of rows matching the num of rows in    flows
    """    
    dates = flows.index
    discounts = discount(dates,r)
    return discounts.multiply(flows,axis='rows').sum()

def inst_to_ann(r):
    """
    Converts a short rate to an annualized rate
    """
    return np.expm1(r)
    
def ann_to_inst(r):
    """
    Converts annualized to a short rate
    """
    return np.log1p(r)

import math
def cir(n_years=10, n_scenarios=1,a=0.05,b=0.03,sigma=0.05,steps_per_year=12,r_0=None):
    """
    Implements the CIR model
    """
    if r_0 is None: r_0 = b
    r_0=ann_to_inst(r_0)
    dt = 1/steps_per_year
    
    num_steps = int(n_years*steps_per_year) + 1
    shock = np.random.normal(0,scale=np.sqrt(dt),size=(num_steps,n_scenarios))
    
    rates = np.empty_like(shock)
    rates[0] = r_0
    
    ## for Price Generation
    h = math.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)
    ###
    
    def price(ttm,r):
        _A = ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P
    prices[0] = price(n_years,r_0)
    
    
    for step in range(1,num_steps):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + d_r_t)
        prices[step] = price(n_years-step*dt,rates[step])
    rates = pd.DataFrame(data=inst_to_ann(rates),index=range(num_steps))
    prices = pd.DataFrame(data=prices,index=range(num_steps))
    return rates, prices

def bond_cash_flow(maturity,principal=100,coupon_rate=0.03,coupons_per_year=12):
    """
    Returns a series of cash flows generate by a bond, indexed by coupon number
    """
    n_coupons = round(maturity*coupons_per_year)
    coupon_amt= principal*coupon_rate/coupons_per_year
    coupon_times = np.arange(1,n_coupons + 1)
    cash_flows = pd.Series(data=coupon_amt, index=coupon_times)
    cash_flows.iloc[-1] +=  principal
    return cash_flows


def bond_price(maturity,principal=100,coupon_rate=0.03,coupons_per_year=12,discount_rate=0.03):
    """
    Price a bond by taking into account discount rate 
    """
    if isinstance(discount_rate,pd.DataFrame):
        pricing_dates = discount_rate.index
        prices=pd.DataFrame(index=pricing_dates,columns=discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t]=bond_price(maturity-t/coupons_per_year,principal,coupon_rate,coupons_per_year,discount_rate.loc[t])
        return prices
    else:
        if maturity<=0: return principal+principal*coupon_rate/coupons_per_year
        cash_flows = bond_cash_flow(maturity, principal,coupon_rate, coupons_per_year)
        return pv(cash_flows,discount_rate/coupons_per_year)

def macaulay_duration(flows,discount_rate):
    """
    Computes Macaulay duration of a sequence of cash flows
    """
    discounted_flows = discount(flows.index,discount_rate)[0]*flows
    weights = discounted_flows/discounted_flows.sum()
    return np.average(flows.index, weights=weights)

def match_duration(cf_t,cf_s,cf_l,discount_rate):
    """
    Returns the weight in cf_s that along with the remainder in cf_l will have an effective duration that matches cf_t
    """
    
    d_t = macaulay_duration(cf_t,discount_rate)
    d_s = macaulay_duration(cf_s,discount_rate)
    d_l = macaulay_duration(cf_l,discount_rate)
    
    return (d_l-d_t)/(d_l-d_s)

def funding_ratio(assets,liabilities,r):
    """
    Computes funding ratio of a series of liabilities, based on an interest rate and current value of assets 
    """
    return pv(assets,r)/pv(liabilities,r)


def bond_total_return(monthly_prices, principal,coupon_rate,coupons_per_year):
    """
    Includes bond price as well as coupon payment in calculating returns
    """
    
    coupons = pd.DataFrame(data=0,index=monthly_prices.index,columns=monthly_prices.columns)
    t_max = monthly_prices.index.max()
    pay_date = np.linspace(12/coupons_per_year,t_max,int(coupons_per_year*t_max/12),dtype=int)
    coupons.iloc[pay_date]=principal*coupon_rate/coupons_per_year
    total_returns = (monthly_prices + coupons)/monthly_prices.shift()-1
    return total_returns.dropna()

def bt_mix(r1,r2, allocator, **kwargs):
    """
    Runs a back test (simulation) of allocating between a two sets of returns
    r1 and r2 are T x N DataFrames or returns where T is the time step index and N is the number of scenarios.
    allocator is a function that takes two sets of returns and allocator specific parameters, and produces
    an allocation to the first portfolio (the rest of the money is invested in the GHP) as a T x 1 DataFrame
    Returns a T x N DataFrame of the resulting N portfolio scenarios
    """
    if not r1.shape == r2.shape:
        raise ValueError("r1 and r2 need to be the same shape")
    weights = allocator(r1,r2,**kwargs)
    if not weights.shape == r1.shape:
        raise ValueError("Allocator return weights don't match r1")
    r_mix = weights*r1 + (1-weights)*r2
    return r_mix


def fixedmix_allocator(r1,r2,w1,**kwargs):
    """
    Produces a time series over T steps of allocations between the PSP and GHP across N scenarios
    PSP and GHP are T x N DataFrames that represent the returns of the PSP and GHP such that:
     each column is a scenario
     each row is the price for a timestep
    Returns an T x N DataFrame of PSP Weights
    """
    return pd.DataFrame(data=w1,index=r1.index,columns=r1.columns)

def terminal_values(rets):
    """
    Return the final values at the end of the return period
    """
    return (rets+1).prod()

def terminal_stats(rets, floor = 0.8, cap=np.inf, name="Stats"):
    """
    Produce Summary Statistics on the terminal values per invested dollar
    across a range of N scenarios
    rets is a T x N DataFrame of returns, where T is the time-step (we assume rets is sorted by time)
    Returns a 1 column DataFrame of Summary Stats indexed by the stat name 
    """
    terminal_wealth = (rets+1).prod()
    breach = terminal_wealth < floor
    reach = terminal_wealth >= cap
    p_breach = breach.mean() if breach.sum() > 0 else np.nan
    p_reach = reach.mean() if reach.sum() > 0 else np.nan
    e_short = (floor-terminal_wealth[breach]).mean() if breach.sum() > 0 else np.nan
    e_surplus = (-cap+terminal_wealth[reach]).mean() if reach.sum() > 0 else np.nan
    sum_stats = pd.DataFrame.from_dict({
        "mean": terminal_wealth.mean(),
        "std" : terminal_wealth.std(),
        "p_breach": p_breach,
        "e_short":e_short,
        "p_reach": p_reach,
        "e_surplus": e_surplus
    }, orient="index", columns=[name])
    return sum_stats

def glidepath_allocator(r1,r2,start_glide=1,end_glide=0):
    """
    Simulates a target date fund type gradual move from r1 to r2
    """
    n_points = r1.shape[0]
    n_columns = r1.shape[1]
    path = pd.Series(data=np.linspace(start_glide,end_glide,num=n_points))
    paths = pd.concat([path]*n_columns,axis=1)
    paths.index = r1.index
    paths.columns = r1.columns
    return paths

def floor_allocator(psp_r, ghp_r, floor, zc_prices, m=3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside
    of the PSP without going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
    """
    if zc_prices.shape != psp_r.shape:
        raise ValueError("PSP and ZC Prices must have the same shape")
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = floor*zc_prices.iloc[step] ## PV of Floor assuming today's rates and flat YC
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0, 1) # same as applying min and max
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the new account value at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        w_history.iloc[step] = psp_w
    return w_history

def drawdown_allocator(psp_r, ghp_r, maxdd, m=3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside
    of the PSP without going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
    """
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    peak_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = (1-maxdd)*peak_value ### Floor is based on Prev Peak
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0, 1) # same as applying min and max
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the new account value and prev peak at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        peak_value = np.maximum(peak_value, account_value)
        w_history.iloc[step] = psp_w
    return w_history