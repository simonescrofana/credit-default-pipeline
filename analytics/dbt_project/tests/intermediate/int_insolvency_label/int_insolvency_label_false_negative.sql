select
    company_id,
    snapshot_date,
    max_dpd_trailing_90d,
    is_insolvent
from {{ ref('int_insolvency_label') }}
where max_dpd_trailing_90d >= 90
  and is_insolvent = 0
