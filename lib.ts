import assert from 'assert';
import * as fs from 'fs';

const numRe = /^-?[.0-9]+$/;
const dateRe = /^([0-9]{2})\/([0-9]{2})\/([0-9]{4})$/;
function MMDDYYYYSlashToDate(s: string) {
  const hit = s.match(dateRe);
  if (hit) {
    const mm = +hit[1];
    assert(mm >= 1 && mm <= 12);
    const dd = +hit[2];
    assert(dd >= 1 && dd <= 31);
    const yyyy = +hit[3];
    assert(yyyy >= 1985 && yyyy <= 3000);
    return new Date(Date.UTC(yyyy, mm - 1, dd));
  }
  return undefined;
}

export function loadCsv(fname: string, header: string[]) {
  return fs.readFileSync(fname, 'utf8').trim().split('\n').map((o, lino) => {
    const arr = o.trim().split(',');
    if (arr.length !== header.length) { throw new Error('uneven lengths @ ' + lino); }
    const ret: {[key: string]: any} = {};
    for (const [idx, key] of header.entries()) {
      const x = arr[idx];
      const parsed = parseFloat(x);
      ret[key] = (x.match(numRe) && !isNaN(parsed)) ? parsed : (MMDDYYYYSlashToDate(x) || x);
    }
    return ret;
  });
}
