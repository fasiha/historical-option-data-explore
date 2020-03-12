import dateutil
import numpy as np
import pandas as pd
import pylab as plt
import sqlite3 as sql

conn = sql.connect('spy.sqlite3')

dateOfInterest = '2020-02-03'

cursor = conn.cursor()
cursor.execute('select underlying_last from spy where quotedate=? limit 1', (dateOfInterest,))
underlier = cursor.fetchone()

cursor = conn.cursor()
cursor.execute('''select last from underlyings where quotedate=? and underlying="VIX"''',
               (dateOfInterest,))
vix = cursor.fetchone()

cursor = conn.cursor()
cursor.execute(
    '''select spy.last, spy.expiration, spy.strike from spy
where spy.type="put"
and spy.quotedate=?
''', (dateOfInterest,))
options = cursor.fetchall()

options = pd.DataFrame([(a, dateutil.parser.parse(b), c) for a, b, c in options],
                       columns=['last', 'expiration', 'strike'])
options['dt'] = options.expiration - dateutil.parser.parse(dateOfInterest)  # delta time
options['dp'] = underlier - options.strike  # delta price of underlier

heatmap = pd.DataFrame([], columns=[], index=np.sort(options.dp.unique()))
for dp, dt, p in options.loc[:, ['dt', 'dp', 'last']].values:
  heatmap.loc[dt, dp] = p

x = np.array(heatmap.loc[heatmap.index > -5, :], dtype=float)
x[np.isnan(x)] = 0

plt.ion()
plt.figure()
plt.imshow(x, aspect='auto', interpolation='none')