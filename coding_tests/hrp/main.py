import scipy.cluster.hierarchy as sch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyfolio as pf
from collections import OrderedDict
import sklearn.covariance

import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd


symbols = [u'EEM', u'EWG', u'TIP', u'EWJ', u'EFA', u'IEF', u'EWQ',
           u'EWU', u'XLB', u'XLE', u'XLF', u'LQD', u'XLK', u'XLU',
           u'EPP', u'FXI', u'VGK', u'VPL', u'SPY', u'TLT', u'BND',
           u'CSJ', u'DIA']
rets = (get_pricing(symbols,
                    fields='price',
                    start_date='2008-1-1',
                    end_date='2016-7-1',
                   )
        .pct_change()
        .rename(columns=lambda x: x.symbol)
)

try:
    network_weights = local_csv('Quantopian.csv', date_column = 'date')
    network_weights = (network_weights.reset_index()
                                  .pivot_table(columns='symbol', values='weight', index='date')
                                  .resample('b', fill_method='ffill')
except IOError:
    pass

# There has to be a more succinct way to do this using rolling_apply or resample
# Would love to see a better version of this.
eoms = rets.resample('1BM').index[13:-1]
covs = pd.Panel(items=eoms, minor_axis=rets.columns, major_axis=rets.columns)
corrs = pd.Panel(items=eoms, minor_axis=rets.columns, major_axis=rets.columns)
covs_robust = pd.Panel(items=eoms, minor_axis=rets.columns, major_axis=rets.columns)
corrs_robust = pd.Panel(items=eoms, minor_axis=rets.columns, major_axis=rets.columns)
for eom in eoms:
    covs.loc[eom] = rets.loc[eom-pd.Timedelta('252d'):eom].cov()
    corrs.loc[eom] = rets.loc[eom-pd.Timedelta('252d'):eom].corr()


portfolio_funcs = OrderedDict((
    ('Equal weighting', lambda returns, cov, corr: np.ones(cov.shape[0]) / len(cov.columns)),
    ('Inverse Variance weighting', lambda returns, cov, corr: getIVP(cov)),
    #('Minimum-variance (CLA) weighting', getCLA),
    ('Mean-Variance weighting', lambda returns, cov, corr: get_mean_variance(returns, cov)),
    ('Robust Mean-Variance weighting', lambda returns, cov, corr: get_mean_variance(returns, cov=cov_robust(cov))),
    ('Min-Variance weighting', lambda returns, cov, corr: get_min_variance(returns, cov)),
    ('Robust Min-Variance weighting', lambda returns, cov, corr: get_min_variance(returns, cov=cov_robust(cov))),
    ('Hierarchical weighting (by LdP)', lambda returns, cov, corr: getHRP(cov, corr)),
    ('Robust Hierarchical weighting (by LdP)', lambda returns, cov, corr: getHRP(cov_robust(cov), corr_robust(corr))),
    #('Network weighting (by Jochen Papenbrock)', lambda x: network_weights.loc[x])
))

weights = pd.Panel(items=portfolio_funcs.keys(), major_axis=eoms, minor_axis=symbols, dtype=np.float32)
port_returns = pd.DataFrame(columns=portfolio_funcs.keys(), index=rets.index)
for name, portfolio_func in portfolio_funcs.iteritems():
    w = pd.DataFrame(index=eoms, columns=symbols, dtype=np.float32)
    for idx in covs:
        w.loc[idx] = portfolio_func(rets.loc[idx - pd.Timedelta('252d'):idx].T,
                                    covs.loc[idx],
                                    corrs.loc[idx]
                                    )

    port_returns[name] = w.loc[rets.index].ffill().multiply(rets).sum(axis='columns')

    weights.loc[name, :, :] = w


# Results
# ==================================================

sns.clustermap(rets.corr())

fig, axs = plt.subplots(nrows=3, figsize=(10, 22))
colors = sns.color_palette(palette='Set3', n_colors=len(port_returns))

for i, name in enumerate(port_returns):
    np.log1p(port_returns[name]).cumsum().plot(ax=axs[0], color=colors[i])

axs[0].legend(loc=0)
axs[0].set(ylabel='Cumulative log returns')

sharpes = port_returns.apply(pf.timeseries.sharpe_ratio)
sns.barplot(x=sharpes.values, y=sharpes.index, ax=axs[1])
axs[1].set(xlabel='Sharpe ratio')

vols = port_returns.apply(pf.timeseries.annual_volatility)
sns.barplot(x=vols.values, y=vols.index, ax=axs[2])
axs[2].set(xlabel='Annual volatility')

# cal = port_returns.apply(pf.timeseries.calmar_ratio)
# sns.barplot(x=cal.values, y=cal.index, ax=ax4)
# ax4.set(xlabel='Calmar Ratio')

# tr = port_returns.apply(tail_ratio)
# sns.barplot(x=tr.values, y=tr.index, ax=ax5)
# ax5.set(xlabel='Tail Ratio')

fig.tight_layout()


# Weights


colors = sns.color_palette(palette='Set3', n_colors=20)
for name in weights:
    weights.loc[name].plot(colors=colors)
    plt.title(name)









