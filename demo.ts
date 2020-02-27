// import * as lib from './lib';
var lib = require('./lib');
var CSV_FILE = '200805.csv';
var HEADERS =
    'UnderlyingSymbol,UnderlyingPrice,Exchange,OptionSymbol,Type,Expiration,DataDate,Strike,Last,Bid,Ask,Volume,OpenInterest,OI2,IV,G1,G2,G3,G4,G5,G6,Alias'
        .split(',');

var arr = lib.loadCsv(CSV_FILE, HEADERS);
