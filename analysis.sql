-- select quotedate, last from spy where type="put" and strike==245 and expiration==date("2021-12-17");

select spy.quotedate, spy.last, vix.last
from spy
inner join vix on vix.quotedate = spy.quotedate
where type="put" and strike==245 and expiration==date("2021-12-17");