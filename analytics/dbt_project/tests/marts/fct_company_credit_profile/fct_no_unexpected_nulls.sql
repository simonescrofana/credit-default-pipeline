select company_profile_key
from {{ ref('fct_company_credit_profile') }}
where active_contracts_count is null
   or max_dpd_trailing_90d is null
   or avg_dpd_trailing_90d is null
   or unpaid_ratio_trailing_90d is null
   or total_outstanding_debt is null
   or billing_disputes_count is null
