import * as lib from './lib';
// var lib = require('./lib');
var CSV_FILE = 'SPYJan2018-Feb2020.csv';

var WANTED =
    'underlying,underlying_last,optionroot,type,expiration,quotedate,strike,last,bid,ask,volume,openinterest,impliedvol,delta,gamma,theta,vega,IVBid,IVAsk'
        .split(',');
var NUMCOLS =
    'underlying_last,strike,last,bid,ask,volume,openinterest,impliedvol,delta,gamma,theta,vega,IVBid,IVAsk'.split(',');
var DATECOLS = 'expiration,quotedate'.split(',');
var arr = lib.loadCsv(CSV_FILE, WANTED, NUMCOLS, DATECOLS);
lib.tosql(arr, 'spy');