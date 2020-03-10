-- sqlite3 spy.sqlite3 < setup.sql
CREATE TABLE IF NOT EXISTS spy(
  underlying TEXT,
  underlying_last DOUBLE,
  exchange TEXT,
  optionroot TEXT,
  optionext TEXT,
  type TEXT,
  expiration DATE,
  quotedate DATE,
  strike DOUBLE,
  last DOUBLE,
  bid DOUBLE,
  ask DOUBLE,
  volume INT,
  openinterest INT,
  impliedvol DOUBLE,
  delta DOUBLE,
  gamma DOUBLE,
  theta DOUBLE,
  vega DOUBLE,
  optionalias TEXT,
  IVBid DOUBLE,
  IVAsk DOUBLE
);

CREATE TABLE IF NOT EXISTS vix(
  quotedate DATE, -- Date
  open DOUBLE, -- VIX Open
  high DOUBLE, -- VIX High
  low DOUBLE, -- VIX Low
  last DOUBLE -- VIX Close
);