select
    company_id,
    snapshot_date,
    foundation_date,
    company_age_days
from {{ ref('int_companies_scd_resolved') }}
where snapshot_date < foundation_date
   or company_age_days < 0
