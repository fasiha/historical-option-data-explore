import numpy as np
import pandas as pd


def addInMoney(df):
  """df.InMoney is positive when the option is in the money
  
  InMoney = price - strike, for call options
          = strike - price, for puts
  """
  df['InMoney'] = df.UnderlyingPrice - df.Strike
  df.loc[df.Type == 'put', ('InMoney')] *= -1
  return df


def isStrictlyAscending(v):
  return np.all(np.array(v[1:]) > np.array(v[:-1]))


def addUnderlyingDiff(df):
  """Add `UnderlyingDiff` field"""
  df['UnderlyingDiff'] = df.UnderlyingPrice.loc[:] + 0
  grouped = df.groupby(['Expiration', 'Type', 'Strike'])
  for ((expiration, callPut, strike), idx) in grouped.groups.items():
    subdf = df.loc[idx, :]  # copy, because non-contiguous!
    assert isStrictlyAscending(subdf.DataDate)

    price = subdf.loc[:, 'UnderlyingPrice'].values
    underlyingDiffCol = subdf.columns.values == 'UnderlyingDiff'
    subdf.iloc[1:, underlyingDiffCol] = price[1:] - price[:-1]
    subdf.iloc[0, underlyingDiffCol] = np.nan

    # recopy to original df
    df.loc[idx, 'UnderlyingDiff'] = subdf.loc[:, 'UnderlyingDiff']
  return df


names = 'UnderlyingSymbol,UnderlyingPrice,Exchange,OptionSymbol,Type,Expiration,DataDate,Strike,Last,Bid,Ask,Volume,OpenInterest,OI2,IV,G1,G2,G3,G4,G5,G6,Alias'.split(
    ',')
df = pd.read_csv('200805.csv', header=0, names=names, parse_dates=['Expiration', 'DataDate'])

df = addInMoney(df)
df['TimeToExpiration'] = df.Expiration - df.DataDate
df = addUnderlyingDiff(df)
