import assert from 'assert';
import * as fs from 'fs';

const dateRe = /^([0-9]{2})\/([0-9]{2})\/([0-9]{4})$/;
function MMDDYYYYSlashToDate(s: string): Date {
  const hit = s.match(dateRe);
  if (hit) {
    const mm = +hit[1];
    assert(mm >= 1 && mm <= 12);
    const dd = +hit[2];
    assert(dd >= 1 && dd <= 31);
    const yyyy = +hit[3];
    assert(yyyy >= 1970 && yyyy <= 2120);
    return new Date(Date.UTC(yyyy, mm - 1, dd));
  }
  throw new Error('not MM/DD/YYYY ' + s);
}

export type Row = (string|number|Date)[];
export interface Csv {
  rows: Row[];
  cols: string[]
}
export function loadCsv(fname: string, wantedCols: string[], numberCols: string[], dateCols: string[]): Csv {
  let lines = fs.readFileSync(fname, 'utf8').trim().split('\n');
  let header = lines[0].split(',').map(s => s.trim());

  {
    let check = (v: string[], msg: string) => assert(v.every(col => header.indexOf(col) >= 0), msg);
    check(wantedCols, 'wanted column(s) not found');
    check(numberCols, 'number column(s) not found');
    check(dateCols, 'date column(s) not found');
    // dates and numbers are also in wanted
    let w = new Set(wantedCols);
    assert(numberCols.every(x => w.has(x)));
    assert(dateCols.every(x => w.has(x)));
    // no overlap
    let n = new Set(numberCols);
    let d = new Set(dateCols);
    assert(numberCols.every(x => !d.has(x)));
    assert(dateCols.every(x => !n.has(x)));
  }

  let wantedIdx: number[] = [];
  let numIdx: number[] = [];
  let dateIdx: number[] = [];
  {
    let col2idx = new Map(header.map((col, i) => [col, i]));
    wantedIdx = wantedCols.map(col => col2idx.get(col) ?? -1);
    numIdx = numberCols.map(c => col2idx.get(c) ?? -1);
    dateIdx = dateCols.map(c => col2idx.get(c) ?? -1);
    assert(wantedIdx.every(n => n >= 0));
    assert(numIdx.every(n => n >= 0));
    assert(dateIdx.every(n => n >= 0));
  }

  const processRow = (raw: string, id: number) => {
    let v: (string|number|Date)[] = raw.split(',');
    numIdx.forEach(i => v[i] = parseFloat(v[i] as string));
    try {
      dateIdx.forEach(i => v[i] = MMDDYYYYSlashToDate(v[i] as string));
    } catch (err) {
      console.error(`error in ${id}`, {numIdx, dateIdx, raw, v, header});
      throw err;
    }
    return wantedIdx.map(i => v[i]);
  };

  // lines might be very long so I don't want to unshift it nor do I want to append potentially millions of times to an
  // array. So I use this Array.from approach.
  return {cols: wantedCols, rows: Array.from(Array(lines.length - 1), (_, i) => processRow(lines[i + 1], i + 1))};
}
