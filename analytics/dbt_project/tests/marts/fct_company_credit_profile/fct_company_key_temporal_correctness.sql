select
    fct.company_profile_key,
    fct.snapshot_date,
    dc.foundation_date,
    dc.dbt_valid_to
from {{ ref('fct_company_credit_profile') }} as fct
inner join {{ ref('dim_companies') }} as dc
    on fct.company_key = dc.company_key
where fct.snapshot_date < dc.foundation_date
   or (dc.dbt_valid_to is not null and fct.snapshot_date >= dc.dbt_valid_to)
