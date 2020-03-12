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
  IVAsk DOUBLE,
  PRIMARY KEY(underlying, quotedate, type, expiration, strike)
);

CREATE TABLE IF NOT EXISTS underlyings(
  underlying TEXT NOT NULL,
  quotedate DATE NOT NULL,
  open DOUBLE,
  high DOUBLE,
  low DOUBLE,
  last DOUBLE,
  PRIMARY KEY(underlying, quotedate)
);