import statsmodels.api as sm
import seaborn as sns
import dateutil
import numpy as np
import pandas as pd
import pylab as plt
import sqlite3 as sql
plt.ion()

conn = sql.connect('spy.sqlite3')
cursor = conn.cursor()
cursor.execute('''select quotedate, last, open from underlyings
where underlying="SPX"
''')
spx = cursor.fetchall()
spx = pd.DataFrame([(dateutil.parser.parse(a), b, c) for a, b, c in spx],
                   columns=['date', 'last', 'open'])

makeroll = lambda n: spx['last'].rolling(n).apply(
    lambda df: (df[-1] - df[0]) / df[0], raw=True) * 100

predWindow = 15

reg = pd.DataFrame(
    np.array([makeroll(predWindow).shift(+1), makeroll(2)]).T, columns='x y'.split()).dropna()

plt.figure()
sns.regplot(x='x', y='y', data=reg)
plt.plot(reg.iloc[-1, 0], reg.iloc[-1, 1], 'ro', label=spx.iloc[-1, 0].isoformat().split('T')[0])
plt.xlabel('{}-session pct change'.format(predWindow))
plt.ylabel('{}th-to-{}th session (one-session) pct change'.format(predWindow, predWindow + 1))
plt.title('S&P 500, 1928–present (Yahoo Finance)')
plt.grid()
plt.legend()

plt.show()

center = -20
radius = 5
regClip = reg[(reg.x <= (center + radius)) & (reg.x >= (center - radius))]

plt.figure()
sns.regplot(x='x', y='y', data=regClip)
plt.plot(
    regClip.iloc[-1, 0], regClip.iloc[-1, 1], 'ro', label=spx.iloc[-1, 0].isoformat().split('T')[0])
plt.xlabel('{}-session pct change'.format(predWindow))
plt.ylabel('{}th-to-{}th session (one-session) pct change'.format(predWindow, predWindow + 1))
plt.title('S&P 500, 1928–present (data: Yahoo Finance)')
plt.legend()
plt.grid()
plt.show()

futureWindow = 2  # 2: final two-day window, i.e., one-day *change*

predWindows = [2, 5, 15, 52 // 2 * 5, 52 * 5]
predColumns = ['x{}'.format(x) for x in predWindows] + ['y']
reg2 = pd.DataFrame(
    np.array([makeroll(win).shift(futureWindow - 1) for win in predWindows] +
             [makeroll(futureWindow)]).T,
    columns=predColumns).dropna()

plotPredictor = 'x15'
assert (plotPredictor in predColumns)
plt.figure()
sns.regplot(x=plotPredictor, y='y', data=reg2)

sns.regplot(x=plotPredictor, y='y', data=reg2[reg2[plotPredictor] <= -15])
sns.regplot(x=plotPredictor, y='y', data=reg2[reg2[plotPredictor] <= -20])
sns.regplot(x=plotPredictor, y='y', data=reg2[reg2[plotPredictor] <= -25])
sns.regplot(x=plotPredictor, y='y', data=reg2[reg2[plotPredictor] <= -27])
plt.xlabel('{}-session pct change'.format(plotPredictor[1:]))
plt.ylabel('Subsequent {}-session pct change'.format(futureWindow - 1))
plt.title('S&P 500, 1928–present (data: Yahoo Finance)')
plt.grid()

# Predictors: trailing returns (we should add interest rates, day of week, proximity to holidays, etc.)
for w in predWindows:
  spx['x' + str(w)] = makeroll(w).shift(futureWindow - 1)
spx['y'] = makeroll(futureWindow)  # prediction: subsequent one year/day returns

# clip = spx[spx.x <= -20]
clip = spx.dropna()
model = sm.OLS(clip.y, sm.add_constant(clip[predColumns[:-1]]))
fit = model.fit()
print(fit.summary())
print(fit.params)

# Predict the future!!
print(fit.predict([1.] + np.array([makeroll(win).iloc[-1] for win in predWindows]).tolist()))

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(spx.x, spx.x2, spx.y)

from sklearn.linear_model import BayesianRidge
reg = BayesianRidge(normalize=True, verbose=True)
reg.fit(clip[predColumns[:-1]].values, clip.y.values)
print(np.hstack((reg.coef_, reg.intercept_)) * 100)
print(
    reg.predict(
        np.array([makeroll(win).iloc[-1] for win in predWindows])[np.newaxis, :], return_std=True))

# from sklearn.linear_model import ARDRegression
# clf = ARDRegression(compute_score=True)
# clf.fit(clip[predColumns[:-1]].values, clip.y.values)

train = clip[predColumns].iloc[:-260]
from sklearn import preprocessing
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
scaler = preprocessing.StandardScaler().fit(train.values[:, :-1])
tscv = TimeSeriesSplit(n_splits=6)
regressor = BayesianRidge()
scores = cross_val_score(
    regressor, scaler.transform(train.iloc[:, :-1]), train.iloc[:, -1], cv=tscv)
plt.figure()
plt.plot(scores)

spx['weekday'] = [x.isoweekday() for x in spx.date]

weekdays = 'Mon,Tue,Wed,Thu,Fri'.split(',')
plt.figure()
[
    sns.regplot(x='x15', y='y', data=spx[spx.weekday == w], label=l, scatter_kws={'alpha': 0.5})
    for w, l in zip(range(1, 6), weekdays)
]
plt.legend()
plt.grid()
plt.xlabel('{}-session pct change'.format(plotPredictor[1:]))
plt.ylabel('Subsequent {}-session pct change'.format(futureWindow - 1))
plt.title('S&P 500, 1928–present (data: Yahoo Finance)')

# f = sm.OLS(
#     spx[spx.weekday == 1].y, sm.add_constant(spx[spx.weekday == 1].x15), missing='drop').fit()
# print(f.params)


def pct(new, old):
  "Percent change between new and old"
  return (new - old) / old


def findTopBottom(x, Nwin, minDrop, argmax=None):
  """Tries to find a top and bottom in a vector

  Returns a tuple of two indexes of `x` such that
  1. the two indexes are within `Nwin` apart and
  2. the percent change between them (in fractional percent) is at least `minDrop`.

  If no such indexes can be found, returns None.
  """
  argmax = x.argmax() if argmax is None else argmax
  argmin = x[argmax:(argmax + Nwin)].argmin() + argmax
  if pct(x[argmin], x[argmax]) <= minDrop:
    return (argmax, argmin)
  return None


def findTopsBottoms(x, Nwin, minDrop, _sofar=None, _xStartIdx=0):
  """Finds all tops and bottoms in a vector.

  A top and bottom pair are defined as less than `Nwin` elements apart and with a fractional percent
  change of less than `minDrop`.

  Return tuple of indexes is unsorted.
  """
  assert (minDrop < 0)
  assert (Nwin > 0)
  _sofar = [] if _sofar is None else _sofar

  # Recursive function has to have an escape hatch.
  if len(x) < 2:
    return _sofar

  # Find the max. If there's room before this, search for tops/bottoms to the left. Move one index
  # to make sure we don't miss anything.
  argmax = x.argmax()
  if argmax >= 2:
    findTopsBottoms(x[:argmax], Nwin, minDrop, _sofar, _xStartIdx)

  # See if this max is a top, i.e., if there's a bottom to the right of the max that meets the
  # requirements.
  res = findTopBottom(x, Nwin, minDrop, argmax)
  if res:
    # yes, this max is a top, we have a top and bottom. Keep track of it (in global coordinates, not
    # the piece `x` we received)
    argmax, argmin = res
    _sofar.append((argmax + _xStartIdx, argmin + _xStartIdx))
    newStart = argmin
  else:
    # no, couldn't find a bottom: `x` never dropped below the threshold soon enough
    newStart = argmax
  # In either case, search for tops/bottoms to the right of the max. Move just one index to make
  # sure we don't miss anything
  findTopsBottoms(x[newStart + 1:], Nwin, minDrop, _sofar, newStart + 1 + _xStartIdx)
  return _sofar


fmtDate = lambda d: d.isoformat().split('T')[0]
datesDrops = [(fmtDate(spx.iloc[top, 0]), fmtDate(spx.iloc[bot, 0]), spx.loc[top, 'last'],
               pct(spx.loc[bot, 'last'], spx.loc[top, 'last']) * 100)
              for top, bot in findTopsBottoms(spx['last'].values, 5 * 52 * 3, -0.1)]
datesDrops = sorted(datesDrops, key=lambda x: x[0])

res = findTopsBottoms(spx['last'].values, 5 * 52 * 3, -0.2)
datesDrops = [(fmtDate(spx.iloc[top, 0]), fmtDate(spx.iloc[bot, 0]),
               pct(spx.loc[bot, 'last'], spx.loc[top, 'last']) * 100) for top, bot in res]
datesDrops = sorted(datesDrops, key=lambda x: x[0])

plt.figure()
for idx, (top, bottom) in enumerate(res):
  pre = 5 * 4
  tmp = spx['last'].iloc[(top - pre):(bottom + 1)]
  tmpx = (np.arange(len(tmp)) - pre) / (52 * 5)
  plt.plot(
      tmpx,
      tmp.values / tmp.loc[top],
      label=fmtDate(spx.date.loc[top]),
      linewidth=1 + (idx // 10),
      alpha=0.75)
plt.grid()
plt.legend()
plt.xlabel('years')
