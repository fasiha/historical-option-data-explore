import dateutil
import statsmodels.api as sm
import seaborn as sns
import dateutil
import numpy as np
import pandas as pd
import pylab as plt
import sqlite3 as sql
plt.ion()
plt.style.use('dark_background')

conn = sql.connect('spy.sqlite3')
cursor = conn.cursor()
cursor.execute('''select quotedate, last, open, low, high from underlyings
where underlying="SPX"
''')
spx = cursor.fetchall()
spx = pd.DataFrame([[dateutil.parser.parse(l[0])] + list(l[1:]) for l in spx],
                   columns=['date', 'last', 'open', 'low', 'high'])
spx['weekday'] = [x.isoweekday() for x in spx.date]
spx['year'] = [x.year for x in spx.date]

makeroll = lambda n: spx['last'].rolling(n).apply(
    lambda df: (df[-1] - df[0]) / df[0], raw=True) * 100


def rollworst(n):
  todayClose = spx['last'].rolling(n).apply(lambda df: df[0], raw=True)
  minLow = spx['low'].rolling(n - 1).min()
  return (minLow - todayClose) / todayClose * 100


def rollbest(n):
  todayClose = spx['last'].rolling(n).apply(lambda df: df[0], raw=True)
  maxHigh = spx['high'].rolling(n - 1).max()
  return (maxHigh - todayClose) / todayClose * 100


def makePredictorDataframe(spx, predWindows, futureWindow):
  clip = spx.copy()
  predColumns = ['x{}'.format(x) for x in predWindows] + ['y']
  # Predictors: trailing returns (we should add interest rates, day of week, proximity to holidays, etc.)
  for w in predWindows:
    clip['x' + str(w)] = makeroll(w).shift(futureWindow - 1)
  clip['y'] = makeroll(futureWindow)  # prediction: subsequent one year/day returns etc.
  clip = clip.dropna()
  return clip, predColumns


def makePredictorDataframeMaxLoss(spx, predWindows, futureWindow, roll=rollworst):
  clip = spx.copy()
  predColumns = ['x{}'.format(x) for x in predWindows] + ['y']
  for w in predWindows:
    clip['x' + str(w)] = makeroll(w).shift(futureWindow - 1)
  clip['y'] = roll(futureWindow)
  # Also add the open on the first day into the future for analysis the night before
  clip['xOvernight'] = pct(clip['open'].rolling(futureWindow).apply(
      lambda df: df[1], raw=True), clip['last'].rolling(futureWindow).apply(
          lambda df: df[0], raw=True)) * 100
  clip = clip.dropna()
  clip['year'] = [x.year for x in clip.date]
  return clip, predColumns, [makeroll(w).iloc[-1] for w in predWindows]


def pct(new, old):
  "Percent change between new and old"
  return (new - old) / old


def dayOfWeek(df, best=True):
  fig, dowaxs = plt.subplots(nrows=5, sharex=True, sharey=True)
  for predDayOfWeek, ax in zip(range(1, 6), dowaxs):
    df[df.weekday == predDayOfWeek].plot.scatter(
        x=predcols[-2], y='y', c='year', cmap='viridis', alpha=0.95, grid=True, s=40, ax=ax)
    ax.set_ylabel(
        ('highest {}-session spike' if best else 'lowest {}-session drop').format(futureWindow - 1))

  dowaxs[-1].tick_params(axis='x', bottom=True, labelbottom=True)
  dowaxs[0].set_title(
      ('Trailing {}-day vs ' + ('highest' if best else 'lowest') +
       ' forward {}-day % return: day of week').format(predWindows[-1], futureWindow - 1))


def minmax(v):
  minloc = v.argmin()
  maxloc = v.argmax()
  oldidx, newidx = (minloc, maxloc) if minloc < maxloc else (maxloc, minloc)
  return pct(v[newidx], v[oldidx])


def worstDrawdown(v):
  maxloc = v.argmax()
  minloc = (v[maxloc:].argmin() + maxloc) if maxloc < (len(v) - 1) else maxloc
  return pct(v[minloc], v[maxloc])


def makeBearMarketDf(spx, trailingLoss=260, trailingChange=4, future=260):
  clip = spx.copy()

  trailingWorstDrawdown = clip['last'].rolling(trailingLoss).apply(worstDrawdown, raw=True) * 100
  trailingPctChange = clip['last'].rolling(trailingChange).apply(
      lambda df: pct(df[-1], df[0]), raw=True) * 100
  futureMaxLoss = clip['last'].rolling(future).apply(
      lambda df: pct(df[1:].min(), df[0]), raw=True) * 100
  futureMaxGain = clip['last'].rolling(future).apply(
      lambda df: pct(df[1:].max(), df[0]), raw=True) * 100

  current = dict(
      trailingWorstDrawdown=trailingWorstDrawdown.iloc[-1],
      trailingPctChange=trailingPctChange.iloc[-1])

  clip['trailingWorstDrawdown'] = trailingWorstDrawdown.shift(future - 1)
  clip['trailingPctChange'] = trailingPctChange.shift(future - 1)
  clip['futureMaxLoss'] = futureMaxLoss
  clip['futureMaxGain'] = futureMaxGain

  return (clip, current)


# bear market analysis
trailingLong = 260
trailingShort = 4
future = 260
clip, current = makeBearMarketDf(spx, trailingLong, trailingShort, future)
print('CURRENT', current)

clip.plot.scatter(
    x='trailingPctChange', y='futureMaxLoss', c='year', cmap='viridis', alpha=0.5, grid=True, s=40)
plt.xlabel('Trailing {}-session pct change'.format(trailingShort))
plt.ylabel('Future {}-session lowest pct change'.format(future))

from matplotlib import colors
divnorm = colors.TwoSlopeNorm(
    vmin=clip.trailingWorstDrawdown.min(),
    vcenter=current['trailingWorstDrawdown'],
    vmax=clip.trailingWorstDrawdown.max())
clip.plot.scatter(
    x='trailingPctChange',
    y='futureMaxLoss',
    c='trailingWorstDrawdown',
    cmap='PiYG',
    norm=divnorm,
    alpha=0.5,
    grid=True,
    s=40)
plt.xlabel('Trailing {}-session pct change'.format(trailingShort))
plt.ylabel('Future {}-session lowest pct change'.format(future))
plt.gcf().get_axes()[1].set_ylabel('Trailing {}-session worst drawdown'.format(trailingLong))

#
clip.plot.scatter(
    x='trailingPctChange',
    y='futureMaxLoss',
    c='futureMaxGain',
    cmap='viridis',
    alpha=0.5,
    grid=True,
    s=40)
clip.plot.scatter(
    x='trailingPctChange',
    y='futureMaxGain',
    c='trailingWorstDrawdown',
    cmap='PiYG',
    norm=divnorm,
    alpha=0.5,
    grid=True,
    s=40)

# clip.plot.hexbin(x='trailingPctChange', y='futureMaxLoss')

# Plotting the worst loss over a period (for potential put strategy)
predWindows = [21]
futureWindow = 9 + 1
worstdf, predcols, current = makePredictorDataframeMaxLoss(spx, predWindows, futureWindow,
                                                           rollworst)
bestdf, _, _ = makePredictorDataframeMaxLoss(spx, predWindows, futureWindow, rollbest)
print('CURRENTLY', current)

dayOfWeek(bestdf)
plt.xlim([-45, -10])
dayOfWeek(worstdf, False)
plt.xlim([-45, -10])


def align(shax1, shax2, low=True):
  ylabel = 'lowest next {}-session drop'.format(
      futureWindow - 1) if low else 'highest next {}-session spike'.format(futureWindow - 1)
  shax2.set_ylabel(ylabel)
  shax2.set_xlabel('trailing {}-session pct return'.format(predWindows[-1]))
  shax1.get_shared_x_axes().join(shax1, shax2)
  shax1.get_shared_y_axes().join(shax1, shax2)
  shax1.set_title('S&P 500 trailing vs. forward returns, 1928–present (Yahoo Finance)')
  shax2.set_title('S&P 500 trailing vs. forward returns, 1928–present (Yahoo Finance)')


shax1 = worstdf.plot.scatter(
    x=predcols[-2], y='y', c='year', cmap='viridis', alpha=0.95, grid=True, s=40)
plt.ylabel('lowest next {}-session drop'.format(futureWindow - 1))
plt.xlabel('trailing {}-session pct return'.format(predWindows[-1]))

shax2 = worstdf.plot.scatter(
    x=predcols[-2], y='y', c='xOvernight', cmap='viridis', alpha=0.95, grid=True, s=40)
align(shax1, shax2)

shax3 = bestdf.plot.scatter(
    x=predcols[-2], y='y', c='year', cmap='viridis', alpha=0.95, grid=True, s=40)
shax3.set_ylabel('highest next {}-session spike'.format(futureWindow - 1))
shax3.set_xlabel('trailing {}-session pct return'.format(predWindows[-1]))

shax4 = bestdf.plot.scatter(
    x=predcols[-2], y='y', c='xOvernight', cmap='viridis', alpha=0.95, grid=True, s=40)
align(shax3, shax4, low=False)

## Plots
futureWindow = 2  # 2: final two-day window, i.e., one-day *change*
predWindows = [2, 5, 15, 52 // 2 * 5, 52 * 5]
clip, predColumns = makePredictorDataframe(spx, predWindows, futureWindow)

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

model = sm.OLS(clip.y, sm.add_constant(clip[predColumns[:-1]]))
fit = model.fit()
print(fit.summary())
print(fit.params)

# Predict the future!!
print(fit.predict([1.] + np.array([makeroll(win).iloc[-1] for win in predWindows]).tolist()))

# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(spx.x, spx.x2, spx.y)

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

from sklearn import preprocessing
import sklearn.gaussian_process as gp


def fitGp(clip, scaler, kernel, predColumns, Nsamples):
  gpr = gp.GaussianProcessRegressor(kernel=kernel)
  gpr.fit(scaler.transform(clip[predColumns[:-1]].values[-Nsamples:]), clip.y.values[-Nsamples:])
  pred = gpr.predict(
      scaler.transform(np.array([makeroll(win).iloc[-1] for win in predWindows])[np.newaxis, :]),
      return_std=True)
  pastPred = gpr.predict(scaler.transform(clip[predColumns[:-1]].values), return_std=True)
  return gpr, pred, pastPred


scaler = preprocessing.StandardScaler().fit(clip[predColumns[:-1]].values)
kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(
    10.0, (1e-3, 1e3)) + gp.kernels.WhiteKernel(1)
gpSimple = fitGp(clip, scaler, kernel, predColumns, 5000)

# show the past
pastPred = gpSimple[2]
plt.figure()
plt.errorbar(clip.date, pastPred[0], yerr=pastPred[1], label='pred')
plt.plot(clip.date, clip.y, 'o', alpha=0.5, label='actual', linewidth=2)
plt.grid()
plt.legend()
# interestingly, while stdev does vary between 0 and 2~, in out-of-training, it does spike.
# Does this mean it hasn't seen that particular sample before?
plt.figure()
plt.plot(clip.date, pastPred[1])

# Try more complex kernels.

k1 = gp.kernels.ConstantKernel(10) * gp.kernels.RBF(10.0)
# k2 = gp.kernels.ConstantKernel(.1) * gp.kernels.RBF() * gp.kernels.ExpSineSquared()
k3 = gp.kernels.ConstantKernel(.1) * gp.kernels.RBF() * gp.kernels.RationalQuadratic()
kernel2 = k1 + k3 + gp.kernels.WhiteKernel(1)

# But really what I think will be needed is to reparameterize the data. Don't just fit next day's actual move.
# Fit next day or next week's log-absolute move (Mandelbrot?).
clip5, predColumns5 = makePredictorDataframe(spx, predWindows, 5)
logret = clip5.copy()
logret = logret[np.isfinite(np.log10(np.abs(logret.y)))]
logret.y = np.log10(np.abs(logret.y))
gpLogret = fitGp(logret, scaler, kernel, predColumns, 5000)
pastPred = gpLogret[2]
plt.figure()
plt.errorbar(logret.date, pastPred[0], yerr=pastPred[1], label='pred')
plt.plot(logret.date, logret.y, 'o', alpha=0.5, label='actual', linewidth=2)
plt.grid()
plt.legend()
plt.figure()
plt.plot(logret.date, pastPred[1])
pred2 = gpLogret[0].predict(
    scaler.transform(np.array([makeroll(win).iloc[-1:] for win in predWindows])), return_std=True)
# stdev spikes up on 22Oct1987 and really on 23oct1987, when the crash was 19oct1987... It is out of sample though.

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
Nsamples = 5000
Nsamplesend = 60
gpr = gp.GaussianProcessRegressor(kernel=kernel)
tscv = TimeSeriesSplit(n_splits=60)
scores = cross_val_score(
    gpr,
    scaler.transform(clip[predColumns[:-1]].values[-Nsamples:-Nsamplesend]),
    clip.y.values[-Nsamples:-Nsamplesend],
    cv=tscv)
plt.figure()
plt.plot(scores)

# Ridge regression
train = clip[predColumns].iloc[:-260]
scaler = preprocessing.StandardScaler().fit(train.values[:, :-1])
tscv = TimeSeriesSplit(n_splits=6)
regressor = BayesianRidge()
scores = cross_val_score(
    regressor, scaler.transform(train.iloc[:, :-1]), train.iloc[:, -1], cv=tscv)
plt.figure()
plt.plot(scores)

# Day of week
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


def bullBearHelper(priceSeries, dateSeries, Nwin, minDrop):
  res = findTopsBottoms(priceSeries.values, Nwin, minDrop)
  res = sorted(res, key=lambda x: x[0])
  return res


def bullBearPlotter(res,
                    priceSeries,
                    dateSeries,
                    ax,
                    pre=(52 * 5) // 4,
                    post=1,
                    elementsPerUnit=(52 * 5 / 12)):
  table = []
  for top, bottom in res:
    topStr = fmtDate(dateSeries.iloc[top])
    botStr = fmtDate(dateSeries[bottom])
    drop = pct(priceSeries.loc[bottom], priceSeries[top]) * 100
    table.append((topStr, botStr, drop))
    tmp = priceSeries.iloc[(top - pre):(bottom + 1 + post)]
    tmpx = (np.arange(len(tmp)) - pre) / elementsPerUnit
    ax.plot(tmpx, tmp.values / tmp.loc[top], label='{}: {:.0f}%'.format(topStr, drop), alpha=0.75)
  ax.grid('on')
  ax.legend(fontsize='x-small')
  ax.set_ylabel('Close (top @ 1.0)')
  return table


res10 = bullBearHelper(spx['last'], spx['date'], 5 * 52 * 3, -0.1)
res20 = bullBearHelper(spx['last'], spx['date'], 5 * 52 * 3, -0.2)

plt.figure()
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)
bullBearPlotter(res20[:9], spx['last'], spx['date'], ax1)
bullBearPlotter(res20[9:], spx['last'], spx['date'], ax2)
ax1.set_title('S&P500 drawdowns exceeding -20% (data: Yahoo Finance)')
ax2.set_xlabel('months after bull market top')

with open('ie_data.xls.json', 'r') as fid:
  import json
  shiller = pd.DataFrame(json.load(fid))
shiller['date'] = [
    dateutil.parser.parse('{}-{:02}-01'.format(int(x[0]), int(x[1]))) for x in shiller.values
]
realax = shiller.plot(
    x='date', y='realPrice', logy=True, label='real (CPI-adjusted) price', linewidth=2.5)
shiller.plot(x='date', y='price', logy=True, ax=realax, grid=True, label='nominal price')
realax.set_title('S&P500: real and nominal price (data: Robert Shiller)')
realax.set_ylabel('price (US dollars)')
realax.yaxis.set_ticks_position('both')
realax.xaxis.set_ticks_position('both')
import matplotlib.ticker as ticker

nominal20raw = bullBearHelper(shiller['price'], shiller['date'], 12 * 3, -0.2)

plt.figure()
ax1 = plt.subplot(311)
ax2 = plt.subplot(312, sharex=ax1, sharey=ax1)
ax3 = plt.subplot(313, sharex=ax1, sharey=ax1)

shillerTable = bullBearPlotter(nominal20raw[:9], shiller['price'], shiller['date'], ax1, 3, 1, 1)
yahooTable1 = bullBearPlotter(res20[:9], spx['last'], spx['date'], ax2)
yahooTable2 = bullBearPlotter(res20[9:], spx['last'], spx['date'], ax3)
ax1.set_title('S&P500 drawdowns exceeding -20% (data: Robert Shiller; Yahoo Finance)')
ax3.set_xlabel('months after bull market top')

real20raw = bullBearHelper(shiller['realPrice'], shiller['date'], 12 * 3, -0.2)
plt.figure()
ax1 = plt.subplot(311)
ax2 = plt.subplot(312, sharex=ax1, sharey=ax1)
ax3 = plt.subplot(313, sharex=ax1, sharey=ax1)
bullBearPlotter(real20raw[:9], shiller['realPrice'], shiller['date'], ax1, 3, 1, 1)
bullBearPlotter(real20raw[9:18], shiller['realPrice'], shiller['date'], ax2, 3, 1, 1)
bullBearPlotter(real20raw[18:], shiller['realPrice'], shiller['date'], ax3, 3, 1, 1)
