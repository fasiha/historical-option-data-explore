import pandas as pd
import sqlite3 as sql

spx = pd.read_csv('^GSPC.csv')

conn = sql.connect('spy.sqlite3')
cursor = conn.cursor()

cursor.execute(
    'select underlying, quotedate from underlyings where underlying="SPX" order by quotedate desc limit 1'
)
print('before', cursor.fetchone())

cursor.executemany(
    '''insert or ignore into underlyings (underlying, quotedate, open, high, low, last) values(?,?,?,?,?,?)''',
    [['SPX'] + x[:5].tolist() for x in spx.values])
conn.commit()

cursor.execute(
    'select underlying, quotedate from underlyings where underlying="SPX" order by quotedate desc limit 1'
)
print('after', cursor.fetchone())
