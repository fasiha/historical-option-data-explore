// http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/vixcurrent.csv

import assert from 'assert';
import * as lib from './lib';
var WANTED = 'Date,VIX Open,VIX High,VIX Low,VIX Close'.split(',');
var NUMCOLS = 'VIX Open,VIX High,VIX Low,VIX Close'.split(',');
var DATECOLS = 'Date'.split(',');

var CSV_FILE = 'vixcurrent.csv';
var arr = lib.loadCsv(CSV_FILE, WANTED, NUMCOLS, DATECOLS, 1);
// console.log(arr);
var headerMap: Map<string, string> =
    new Map('Date,quotedate|VIX Open,open|VIX High,high|VIX Low,low|VIX Close,last'.split('|').map(
        v => v.split(',').slice(0, 2) as [string, string]));
assert(arr.cols.every(s => headerMap.has(s)));
var newCols = arr.cols.map(s => headerMap.get(s) || '');
assert(newCols.every(s => !!s));
arr.cols = newCols;
lib.tosql(arr, 'vix');
