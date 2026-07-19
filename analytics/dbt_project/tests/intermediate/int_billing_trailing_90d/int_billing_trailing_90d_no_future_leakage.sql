select
    f.company_id,
    f.snapshot_date,
    i.invoice_id,
    i.issue_date,
    i.due_date
from {{ ref('int_billing_trailing_90d') }} as f
inner join {{ ref('stg_energy_contracts') }} as c
    on f.company_id = c.company_id
inner join {{ ref('stg_invoices') }} as i
    on i.contract_id = c.contract_id
where i.issue_date > f.snapshot_date
    and i.due_date > f.snapshot_date - interval '90 days'
    and i.due_date <= f.snapshot_date
