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

-- select vix.quotedate, vix.last, spy.last
-- from vix
-- left join spy
-- on (spy.quotedate = vix.quotedate) 
-- and (spy.type="put" or spy.type is null)
-- and (spy.strike=245 or spy.strike is null)
-- and (spy.expiration=date("2021-12-17") or spy.expiration is null)
-- where vix.quotedate>=date("2018-09-01")
-- order by vix.quotedate;

-- select distinct expiration from spy order by expiration;
-- 2020-03-02
-- 2020-03-04
-- 2020-03-06
-- 2020-03-09
-- 2020-03-11
-- 2020-03-13
-- 2020-03-16
-- 2020-03-18
-- 2020-03-20
-- 2020-03-23
-- 2020-03-25
-- 2020-03-27
-- 2020-03-30
-- 2020-03-31
-- select min(quotedate) from spy where expiration = date("2020-03-20"); -- 2018-03-19

select vix.quotedate, vix.last, spy.last, spy.expiration
from vix
left join spy
on (spy.quotedate = vix.quotedate) 
and (spy.type="put" or spy.type is null)
and (spy.strike=245 or spy.strike is null)
and (spy.expiration=date("2021-12-17") or spy.expiration=date("2020-03-20") or spy.expiration is null)
where vix.quotedate>=date("2018-09-01")
order by vix.quotedate, spy.expiration;
