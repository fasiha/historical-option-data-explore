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

import seaborn as sns
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

import statsmodels.api as sm
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
