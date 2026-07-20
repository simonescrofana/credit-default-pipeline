with spine as (
    select * from {{ ref('int_company_date_spine') }}
),

contracts as (
    select * from {{ ref('stg_energy_contracts') }}
),

invoices as (
    select * from {{ ref('stg_invoices') }}
),

payments as (
    select * from {{ ref('stg_payments') }}
),

invoices_with_company as (
    select
        invoices.invoice_id,
        contracts.company_id,
        invoices.due_date,
        invoices.total_amount,
        invoices.invoice_status,
        payments.payment_date
    from invoices
    inner join contracts
        on invoices.contract_id = contracts.contract_id
    left join payments
        on invoices.invoice_id = payments.invoice_id
        and payments.payment_status = 'completed'
),

invoices_asof as (
    select
        spine.company_id,
        spine.snapshot_date,
        invoices_with_company.invoice_id,
        invoices_with_company.total_amount,
        invoices_with_company.invoice_status,
        greatest(
            0,
            (
                coalesce(invoices_with_company.payment_date, spine.snapshot_date)
                - invoices_with_company.due_date
            )
        ) as days_past_due,
        (
            invoices_with_company.payment_date is null
            and invoices_with_company.due_date <= spine.snapshot_date
        ) as is_outstanding_asof
    from spine
    inner join invoices_with_company
        on spine.company_id = invoices_with_company.company_id
        and invoices_with_company.due_date > spine.snapshot_date - interval '90 days'
        and invoices_with_company.due_date <= spine.snapshot_date
),

aggregated as (
    select
        company_id,
        snapshot_date,
        coalesce(max(days_past_due), 0) as max_dpd_trailing_90d,
        coalesce(avg(days_past_due), 0)::decimal(10, 2) as avg_dpd_trailing_90d,
        coalesce(
            (
                count(*) filter (where invoice_status in ('unpaid', 'overdue'))::decimal
                / nullif(count(*), 0)
            ),
            0
        )::decimal(5, 4) as unpaid_ratio_trailing_90d,
        coalesce(
            sum(total_amount) filter (where is_outstanding_asof), 0
        )::decimal(15, 2) as total_outstanding_debt
    from invoices_asof
    group by company_id, snapshot_date
)

select * from aggregated
