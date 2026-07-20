select spine.company_id, spine.snapshot_date
from {{ ref('int_company_date_spine') }} as spine
left join {{ ref('fct_company_credit_profile') }} as fct
    on spine.company_id = fct.company_id
    and spine.snapshot_date = fct.snapshot_date
where fct.company_profile_key is null
