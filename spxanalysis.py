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
