{{ config(materialized='table') }}

with spine as (
    select * from {{ ref('int_company_date_spine') }}
),

logins as (
    select * from {{ ref('stg_user_web_logins') }}
),

logins_asof as (
    select
        spine.company_id,
        spine.snapshot_date,
        logins.login_timestamp,
        (logins.login_timestamp::date > spine.snapshot_date - interval '30 days')
            as is_in_last_30d
    from spine
    inner join logins
        on spine.company_id = logins.company_id
        and logins.login_timestamp::date > spine.snapshot_date - interval '90 days'
        and logins.login_timestamp::date <= spine.snapshot_date
),

last_login as (
    select
        company_id,
        snapshot_date,
        max(login_timestamp) as last_login_timestamp
    from logins_asof
    group by company_id, snapshot_date
),

velocity as (
    select
        company_id,
        snapshot_date,
        count(*) filter (where is_in_last_30d) as logins_last_30d,
        (count(*)::decimal / 3) as avg_monthly_logins_last_90d
    from logins_asof
    group by company_id, snapshot_date
),

combined as (
    select
        spine.company_id,
        spine.snapshot_date,
        (spine.snapshot_date - last_login.last_login_timestamp::date)
            as days_since_last_login,
        case
            when coalesce(velocity.avg_monthly_logins_last_90d, 0) > 0
                then (
                    velocity.logins_last_30d::decimal
                    / velocity.avg_monthly_logins_last_90d
                )::decimal(5, 4)
            else null
        end as login_velocity
    from spine
    left join last_login
        on spine.company_id = last_login.company_id
        and spine.snapshot_date = last_login.snapshot_date
    left join velocity
        on spine.company_id = velocity.company_id
        and spine.snapshot_date = velocity.snapshot_date
)

select * from combined
