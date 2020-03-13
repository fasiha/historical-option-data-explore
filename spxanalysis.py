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
makerollLast = lambda n: spx['last'].rolling(n).apply(
    lambda df: (df[-1] - df[-2]) / df[-2], raw=True) * 100

N = 15
r15 = makeroll(N)
r16 = makeroll(N + 1)
lastAfter16 = makerollLast(16)

import seaborn as sns
reg = pd.DataFrame(
    np.array([makeroll(N).shift(+1), makerollLast(16)]).T, columns='x y'.split()).dropna()

plt.figure()
sns.regplot(x='x', y='y', data=reg)
plt.plot(reg.iloc[-1, 0], reg.iloc[-1, 1], 'ro', label=spx.iloc[-1, 0].isoformat().split('T')[0])
plt.xlabel('{}-session pct change'.format(N))
plt.ylabel('{}th-to-{}th session (one-session) pct change'.format(N, N + 1))
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
plt.xlabel('{}-session pct change'.format(N))
plt.ylabel('{}th-to-{}th session (one-session) pct change'.format(N, N + 1))
plt.title('S&P 500, 1928–present (data: Yahoo Finance)')
plt.legend()
plt.grid()
plt.show()

import statsmodels.api as sm
model = sm.OLS(regClip.y, sm.add_constant(regClip.x))
print(model.fit().params)