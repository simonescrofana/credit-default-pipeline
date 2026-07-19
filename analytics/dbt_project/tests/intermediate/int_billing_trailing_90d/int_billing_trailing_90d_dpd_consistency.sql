select
    company_id,
    snapshot_date,
    max_dpd_trailing_90d,
    avg_dpd_trailing_90d
from {{ ref('int_billing_trailing_90d') }}
where avg_dpd_trailing_90d > max_dpd_trailing_90d
   or (max_dpd_trailing_90d = 0 and avg_dpd_trailing_90d > 0)
