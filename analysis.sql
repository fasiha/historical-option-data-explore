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
-- select quotedate,last from spy where expiration = date("2020-03-20") and strike=245 and type='put'; -- 2018-03-19

-- select distinct expiration, min(quotedate) from spy group by expiration order by expiration;
-- select distinct expiration, min(quotedate) from spy where strike=245 group by expiration order by expiration;

select vix.quotedate, vix.last, spy.last, spy.expiration
from vix
left join spy
on (spy.quotedate = vix.quotedate) 
and (spy.type="put" or spy.type is null)
and (spy.strike=245 or spy.strike is null)
and (spy.expiration in (
  date("2018-12-31"),
  date("2019-01-18"),
  date("2019-03-15"),
  date("2019-06-21"),
  date("2019-09-20"),
  date("2019-12-20"),
  date("2020-01-17"),
  date("2020-03-20"),
  date("2020-06-19"),
  date("2020-09-18"),
  date("2020-12-18"),
  date("2021-01-15"),
  date("2021-03-19"),
  date("2021-06-18"),
  date("2021-09-17"),
  date("2021-12-17")
) or spy.expiration is null)
where vix.quotedate>=date("2018-01-01")
order by vix.quotedate, spy.expiration;
