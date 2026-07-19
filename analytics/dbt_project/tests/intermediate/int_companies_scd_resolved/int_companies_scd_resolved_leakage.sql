-- Anti-leakage test
select
    r.company_id,
    r.snapshot_date,
    c.dbt_valid_from
from {{ ref('int_companies_scd_resolved') }} r
join {{ ref('stg_companies') }} c 
    on r.company_id = c.company_id
    and r.company_key = {{ generate_company_key('c.company_id', 'c.dbt_valid_from') }}
where c.dbt_valid_from::date > r.snapshot_date
