import * as lib from './lib';
// var lib = require('./lib');
var CSV_FILE = 'SPYJan2018-Feb2020.csv';

// underlying,underlying_last,optionroot,type,expiration,quotedate,strike,last,bid,ask,volume,openinterest,impliedvol,delta,gamma,theta,vega,IVBid,IVAsk

var WANTED =
    'underlying,underlying_last,optionroot,type,expiration,quotedate,strike,last,bid,ask,volume,openinterest,impliedvol,delta,gamma,theta,vega,IVBid,IVAsk'
        .split(',');
var NUMCOLS =
    'underlying_last,strike,last,bid,ask,volume,openinterest,impliedvol,delta,gamma,theta,vega,IVBid,IVAsk'.split(',');
var DATECOLS = 'expiration,quotedate'.split(',');
var arr = lib.loadCsv(CSV_FILE, WANTED, NUMCOLS, DATECOLS);

function tosql(arr: lib.Csv) {
  console.log('-- ok!');
  const cols = arr.cols.join(',');
  for (const row of arr.rows) {
    let [underlying, underlying_last, optionroot, type, expiration, quotedate, strike, last, bid, ask, volume,
         openinterest, impliedvol, delta, gamma, theta, vega, IVBid, IVAsk] = row;
    let fixdate = (d: any) => `"${d.toISOString()}"`;
    const vals = `"${underlying}", ${underlying_last}, "${optionroot}", "${type}", ${fixdate(expiration)}, ${
        fixdate(quotedate)}, ${strike}, ${last}, ${bid}, ${ask}, ${volume}, ${openinterest}, ${impliedvol}, ${delta}, ${
        gamma}, ${theta}, ${vega}, ${IVBid}, ${IVAsk}`;
    const cmd = `insert into spy (${cols}) values (${vals});`;
    console.log(cmd);
  }
}

tosql(arr);