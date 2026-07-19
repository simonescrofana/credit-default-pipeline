with billing as (
    select * from {{ ref('int_billing_trailing_90d') }}
),

labeled as (
    select
        company_id,
        snapshot_date,
        max_dpd_trailing_90d,
        case
            when max_dpd_trailing_90d >= 90 then 1
            else 0
        end as is_insolvent

    from billing
)

select * from labeled
