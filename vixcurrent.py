import dateutil
import pandas as pd
import sqlite3 as sql

vix = pd.read_csv('vixcurrent.csv', skiprows=1)

conn = sql.connect('spy.sqlite3')
cursor = conn.cursor()

select = 'select underlying, quotedate from underlyings where underlying="VIX" order by quotedate desc limit 1'
cursor.execute(select)
print('before', cursor.fetchone())

cursor.executemany(
    '''insert or ignore into underlyings (underlying, quotedate, open, high, low, last) values(?,?,?,?,?,?)''',
    [['VIX', dateutil.parser.parse(d).isoformat().split('T')[0], o, h, l, c]
     for d, o, h, l, c in vix.values])
conn.commit()

cursor.execute(select)
print('after', cursor.fetchone())
