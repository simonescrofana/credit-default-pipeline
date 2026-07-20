select
    company_id,
    snapshot_date,
    unpaid_ratio_trailing_90d,
    total_outstanding_debt
from {{ ref('int_billing_trailing_90d') }}
where total_outstanding_debt > 0 
  and unpaid_ratio_trailing_90d = 0
