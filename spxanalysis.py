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

futureWindow = 5 * 52

reg2 = pd.DataFrame(
    np.array([
        makeroll(predWindow).shift(futureWindow - 1),
        makeroll(futureWindow),
    ]).T,
    columns='x  y'.split()).dropna()
plt.figure()
sns.regplot(x='x', y='y', data=reg2)

sns.regplot(x='x', y='y', data=reg2[reg2.x <= -15])
sns.regplot(x='x', y='y', data=reg2[reg2.x <= -20])
sns.regplot(x='x', y='y', data=reg2[reg2.x <= -25])
sns.regplot(x='x', y='y', data=reg2[reg2.x <= -27])
plt.xlabel('{}-session pct change'.format(predWindow))
plt.ylabel('Subsequent one year pct change')
plt.title('S&P 500, 1928–present (data: Yahoo Finance)')
plt.grid()

spx['x'] = makeroll(predWindow).shift(futureWindow -
                                      1)  # first predictor: trailing fifteen day returns
spx['x2'] = makeroll(52 // 2 * 5).shift(futureWindow -
                                        1)  # next predictor: trailing six month returns
spx['y'] = makeroll(futureWindow)  # prediction: subsequent one year returns

import statsmodels.api as sm
model = sm.OLS(reg2[reg2.x <= -20].y, sm.add_constant(reg2[reg2.x <= -20].x))
print(model.fit().params)

clip = spx[spx.x <= -20]
print(sm.OLS(clip.y, sm.add_constant(clip[['x', 'x2']])).fit().params)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(spx.x, spx.x2, spx.y)
