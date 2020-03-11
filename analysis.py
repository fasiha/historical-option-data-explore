import pandas as pd
df = pd.read_csv("vixcomp.csv", parse_dates=['date'])
df.plot(x='date', y=['spy', 'vix'])

import pylab as plt
df2 = pd.read_csv("vixcomp2.csv", parse_dates=['date', 'exp'])
# df2.groupby('exp').apply(lambda df: df.plot(x='date', y=['spy', 'vix']))  # two plots
foo = df2.groupby('exp').apply(lambda df: [df.date, df[['spy', 'vix']]])
plt.figure()
plt.plot(*foo[0], *foo[1])
plt.legend(('spy1', 'vix', 'spy2', 'vix'))


def geometricDiff1(y):
  return y[1:].values / y[:-1].values


def geometricDiff(x, y):
  return (x[1:], geometricDiff1(y))


plt.figure()
plt.plot(*geometricDiff(*foo[0]), *geometricDiff(*foo[1]))
plt.legend(('spy1', 'vix', 'spy2', 'vix'))

plt.figure()
for (key, df) in df2.groupby('exp'):
  plt.scatter(geometricDiff1(df.vix), geometricDiff1(df.spy), label=key)
plt.xlabel('diff vix')
plt.ylabel('diff option')

plt.figure()
for (key, df) in df2.groupby('exp'):
  tmp = df[['spy', 'vix']].pct_change()
  plt.plot(df.date, tmp.vix, 'k-', linewidth=0.5, label='vix')
  plt.plot(df.date, tmp.spy, label=key)
plt.legend()

plt.figure()
for (key, df) in df2.groupby('exp'):
  ytmp = geometricDiff1(df[['spy', 'vix']])  # two cols, [spy, vix], today/yesterday
  xtmp = (df.date[1:] - key).dt.days
  plt.plot(xtmp, ytmp[:, 0] / ytmp[:, 1], label=key)  # diff spy / diff vix
plt.legend()
