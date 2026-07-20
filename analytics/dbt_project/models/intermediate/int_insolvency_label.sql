with spine as (
    select * from {{ ref('int_company_date_spine') }}
),

billing as (
    select * from {{ ref('int_billing_trailing_90d') }}
),

labeled as (
    select
        spine.company_id,
        spine.snapshot_date,
        coalesce(billing.max_dpd_trailing_90d, 0) as max_dpd_trailing_90d,
        case
            when coalesce(billing.max_dpd_trailing_90d, 0) >= 90 then 1
            else 0
        end as is_insolvent
    from spine
    left join billing
        on spine.company_id = billing.company_id
        and spine.snapshot_date = billing.snapshot_date
)

select * from labeled
