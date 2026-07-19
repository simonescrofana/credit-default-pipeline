with spine as (
    select * from {{ ref('int_company_date_spine') }}
),

tickets as (
    select * from {{ ref('stg_crm_support_tickets') }}
),

tickets_asof as (
    select
        spine.company_id,
        spine.snapshot_date,
        tickets.ticket_category,
        tickets.satisfaction_score,
        tickets.created_at,
        tickets.resolved_at
    from spine
    inner join tickets
        on spine.company_id = tickets.company_id
        and tickets.created_at::date > spine.snapshot_date - interval '90 days'
        and tickets.created_at::date <= spine.snapshot_date
),

aggregated as (
    select
        company_id,
        snapshot_date,
        coalesce(
            count(*) filter (where ticket_category = 'billing'), 0
        ) as billing_disputes_count,
        avg(satisfaction_score) filter (where resolved_at is not null)
            ::decimal(3, 2) as average_satisfaction_score
    from tickets_asof
    group by company_id, snapshot_date
)

select * from aggregated
