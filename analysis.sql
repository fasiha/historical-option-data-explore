-- select quotedate, last from spy where type="put" and strike==245 and expiration==date("2021-12-17");

-- select spy.quotedate, spy.last, vix.last
-- from spy
-- inner join vix on vix.quotedate = spy.quotedate
-- where type="put" and strike==245 and expiration==date("2021-12-17");


-- select * from vix where vix.quotedate>=date("2018-09-01")  limit 10;

-- -- Doesn't work for some reason, need the restrictions on spy to be in the JOIN as below
-- select vix.quotedate, vix.last, spy.last
-- from vix
-- left join spy
-- on spy.quotedate = vix.quotedate
-- where vix.quotedate>=date("2018-09-01")
-- and (spy.type="put" or spy.type is null)
-- and (spy.strike=245 or spy.strike is null)
-- and (spy.expiration=date("2021-12-17") or spy.expiration is null)
-- order by vix.quotedate;

select vix.quotedate, vix.last, spy.last
from vix
left join spy
on (spy.quotedate = vix.quotedate) 
and (spy.type="put" or spy.type is null)
and (spy.strike=245 or spy.strike is null)
and (spy.expiration=date("2021-12-17") or spy.expiration is null)
where vix.quotedate>=date("2018-09-01")
order by vix.quotedate;