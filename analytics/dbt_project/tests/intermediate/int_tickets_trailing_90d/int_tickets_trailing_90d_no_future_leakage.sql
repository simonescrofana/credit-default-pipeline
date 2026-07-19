select
    t.company_id,
    t.snapshot_date,
    t.billing_disputes_count
from {{ ref('int_tickets_trailing_90d') }} t
left join {{ ref('stg_crm_support_tickets') }} stg
    on t.company_id = stg.company_id
    and stg.created_at::date > (t.snapshot_date - interval '90 days')::date
    and stg.created_at::date <= t.snapshot_date
where (t.billing_disputes_count > 0 or t.average_satisfaction_score is not null)
  and stg.ticket_id is null
