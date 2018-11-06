import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import scipy.stats as stats
from tqdm import tqdm, tqdm_notebook

def mad_outlier(y, thresh=3.):
    median = np.median(y)
    diff = np.sum((y-median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    
    modified_z_score = 0.6745 * diff / med_abs_deviation
    return modified_z_score > thresh

def tick_bars(df, price_column, m):
    t = df[price_column]
    ts = 0
    idx = []
    
    for i, x in enumerate(tqdm(t)):
        ts += 1
        if ts >= m:
            idx.append(i)
            ts = 0
            continue
            
    return idx

def tick_bar_df(df, price_column, m):
    idx = tick_bars(df, price_column, m)
    return df.iloc[idx].drop_duplicates()

def select_sample_data(ref, sub, price_col, date):
    xdf = ref[price_col].loc[date]
    xtdf = sub[price_col].loc[date]
    return xdf, xtdf

def plot_sample_data(ref, sub, bar_type, *args, **kwds):
    f,axes = plt.subplots(3, sharex=True, sharey=True, figsize=(10,7))
    ref.plot(ax=axes[0], label='price', *args, **kwds)
    sub.plot(ax=axes[0], marker='.', ls='', color='r', label=bar_type, *args, **kwds)
    axes[0].legend()
    
    ref.plot(ax=axes[1], label='price', *args, **kwds)
    sub.plot(ax=axes[2], marker='.', ls='', label=bar_type, color='r', *args, **kwds)
    
    for ax in axes[1:]: 
        ax.legend()
    plt.tight_layout()
    
    return

def volume_bars(df, volume_column, m):
    t = df[volume_column]
    ts = 0
    idx = []
    
    for i, x in enumerate(tqdm(t)):
        ts += math.fabs(x)
        if ts >= m:
            idx.append(i)
            ts = 0
            continue
            
    return idx

def volume_bar_df(df, volume_column, m):
    idx = volume_bars(df, volume_column, m)
    return df.iloc[idx].drop_duplicates()

def dollar_bars(df, dollar_column, m):
    t = df[dollar_column]
    ts = 0
    idx = []
    for i, x in enumerate(tqdm(t)):
        ts += math.fabs(x)
        if ts >= m:
            idx.append(i)
            ts = 0
            continue
            
    return idx

def dollar_bar_df(df, dollar_column, m):
    idx = dollar_bars(df, dollar_column, m)    
    return df.iloc[idx].drop_duplicates()

def dollar_bar_df0(df, dollar_column, m):
    idx = dollar_bars(df, dollar_column, m)

    df0 = df.iloc[idx].drop_duplicates()
    out = pd.DataFrame(index = df0.index)

    ####################################### CRAP:
    i0 = df.index[0]
    i1 = df0.index[0]

    df1 = df[i0:i1]

    price = df1.price    
    volume = np.abs(df1.volume)

    volPos = df1.volume[df1.volume > 0]
    volNeg = df1.volume[df1.volume < 0]

    vwap = price.dot(volume).sum() / volume.sum()

    out.loc[i1, 'vwap'] = vwap
    out.loc[i1, 'high'] = price.max()
    out.loc[i1, 'low'] = price.min()
    out.loc[i1, 'vol_pos'] = volPos.sum()
    out.loc[i1, 'vol_neg'] = np.abs(volNeg.sum())
    ####################################### CRAP:
    
    for i in tqdm(range(len(df0.index) - 1)):
    	i0 = df0.index[i]
    	i1 = df0.index[i + 1]

    	df1 = df[i0:i1]

    	price = df1.price
    	volume = np.abs(df1.volume)

        volPos = df1.volume[df1.volume > 0]
    	volNeg = df1.volume[df1.volume < 0]

    	vwap = price.dot(volume).sum() / volume.sum()

    	out.loc[i1, 'vwap'] = vwap
    	out.loc[i1, 'high'] = price.max()
    	out.loc[i1, 'low'] = price.min()
    	out.loc[i1, 'vol_pos'] = volPos.sum()
    	out.loc[i1, 'vol_neg'] = np.abs(volNeg.sum())

    df0['vwap'] = out.vwap
    df0['high'] = out.high
    df0['low']  = out.low
    df0['vol_pos'] = out.vol_pos
    df0['vol_neg'] = out.vol_neg

    return df0

def count_bars(df, price_col='price'):
    return df.groupby(pd.TimeGrouper('1d'))[price_col].count()

def scale(s):
    return (s-s.min())/(s.max()-s.min())

def returns(s):
    arr = np.diff(np.log(s))
    return pd.Series(arr, index=s.index[1:])

def get_test_stats(bar_types_, bar_returns_, test_func_, *args, **kwds):
    dct = { bar : (int(bar_returns_.shape[0]), test_func_(bar_returns_,*args,**kwds)) for bar, bar_returns_ in zip(bar_types_, bar_returns_) }
    df = (pd.DataFrame.from_dict(dct).rename(index = {0 : 'sample_size', 1:'{test_func.__name__}_stat'}).T)
    
    return df

def jb(x, test=True):
    np.random.seed(12345678)
    if test: 
        return stats.jarque_bera(x)[0]
    
    return stats.jarque_bera(x)[1]

def shapiro(x, test=True):
    np.random.seed(12345678)
    if test: 
        return stats.shapiro(x)[0]
    
    return stats.shapiro(x)[1]

def bt(p0, p1, bs):    
    if np.isclose((p1-p0), 0.0, atol = 0.001):
        b = bs[-1]
        return b
    else:
        b = np.abs(p1-p0)/(p1-p0)
        return b
    
def get_imbalance(t):
    bs = np.zeros_like(t)
    for i in np.arange(1, bs.shape[0]):
        t_bt = bt(t[i-1], t[i], bs[:i])
        bs[i-1] = t_bt
        
    return bs[:-1]

def test_t_abs(absTheta, t, E_bs):
    return (absTheta >= t * E_bs)

def agg_imbalance_bars(df):
    start = df.index[0]
    bars = []
    for row in df.itertuples():
        t_abs = row.absTheta
        rowIdx = row.Index
        E_bs = row.E_bs
        
        t = df.loc[start:rowIdx].shape[0]
        if t < 1: 
            t = 1
        if test_t_abs(t_abs, t, E_bs):
            bars.append((start, rowIdx, t))
            start = rowIdx
            
    return bars