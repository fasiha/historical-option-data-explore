import numpy as np
import pandas as pd


def addInMoney(df):
  """df.InMoney is positive when the option is in the money
  
  InMoney = price - strike, for call options
          = strike - price, for puts
  """
  df['InMoney'] = df.underlying_last - df.strike
  df.loc[df.type == 'put', ('InMoney')] *= -1
  return df


def isStrictlyAscending(v):
  return np.all(np.array(v[1:]) > np.array(v[:-1]))


def addUnderlyingDiff(df):
  """Add `UnderlyingDiff` field"""
  df['UnderlyingDiff'] = df.underlying_last.loc[:] + 0
  grouped = df.groupby(['expiration', 'type', 'strike'])
  for ((expiration, callPut, strike), idx) in grouped.groups.items():
    subdf = df.loc[idx, :]  # copy, because non-contiguous!
    assert isStrictlyAscending(subdf.quotedate)

    price = subdf.loc[:, 'underlying_last'].values
    underlyingDiffCol = subdf.columns.values == 'UnderlyingDiff'
    subdf.iloc[1:, underlyingDiffCol] = price[1:] - price[:-1]
    subdf.iloc[0, underlyingDiffCol] = np.nan

    # recopy to original df
    df.loc[idx, 'UnderlyingDiff'] = subdf.loc[:, 'UnderlyingDiff']
  return df


df = pd.read_csv(
    'SPY_2018.csv',
    parse_dates=['expiration', 'quotedate'],
    dtype={
        'underlying': str,
        'underlying_last': float,
        'strike': float,
        'last': float,
        'bid': float,
        'ask': float,
        'volume': int,
        'openinterest': int,
        'impliedvol': float,
        'delta': float,
        'theta': float,
        'vega': float,
        'gamma': float,
        'IVBid': float,
        'IVAsk': float,
    })

df = addInMoney(df)
df['TimeToExpiration'] = df.expiration - df.quotedate
df = addUnderlyingDiff(df)
