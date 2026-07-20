select spine.company_id, spine.snapshot_date
from {{ ref('int_company_date_spine') }} as spine
inner join {{ ref('stg_companies') }} as c
    on spine.company_id = c.company_id
where spine.snapshot_date < date_trunc('month', c.foundation_date)
