select spine.company_id, spine.snapshot_date, c.dbt_valid_to
from {{ ref('int_company_date_spine') }} as spine
inner join {{ ref('stg_companies') }} as c
    on spine.company_id = c.company_id
where c.dbt_valid_to is not null
    and spine.snapshot_date >= c.dbt_valid_to
    and c.dbt_valid_to = (
        select max(c2.dbt_valid_to)
        from {{ ref('stg_companies') }} as c2
        where c2.company_id = spine.company_id
    )
