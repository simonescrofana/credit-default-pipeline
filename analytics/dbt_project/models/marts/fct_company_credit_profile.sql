with spine as (
    select * from {{ ref('int_company_date_spine') }}
),

companies_scd as (
    select * from {{ ref('int_companies_scd_resolved') }}
),

contracts as (
    select * from {{ ref('int_contracts_asof') }}
),

billing as (
    select * from {{ ref('int_billing_trailing_90d') }}
),

financials as (
    select * from {{ ref('int_financial_asof') }}
),

logins as (
    select * from {{ ref('int_logins_trailing') }}
),

tickets as (
    select * from {{ ref('int_tickets_trailing_90d') }}
),

insolvency as (
    select * from {{ ref('int_insolvency_label') }}
),

final as (
    select
        -- Surrogate PK for this fact table, one hash per (company_id, snapshot_date)
        {{ dbt_utils.generate_surrogate_key(
            ['spine.company_id', 'spine.snapshot_date']
        ) }} as company_profile_key,
        companies_scd.company_key,
        spine.company_id,
        spine.snapshot_date,
        companies_scd.company_age_days,
        coalesce(contracts.active_contracts_count, 0) as active_contracts_count,
        coalesce(contracts.has_active_electricity_contract, false)
            as has_active_electricity_contract,
        coalesce(contracts.has_active_gas_contract, false)
            as has_active_gas_contract,
        financials.leverage_ratio,
        financials.cash_to_debt_ratio,
        financials.net_profit_margin,
        financials.ebitda,
        coalesce(billing.max_dpd_trailing_90d, 0) as max_dpd_trailing_90d,
        coalesce(billing.avg_dpd_trailing_90d, 0) as avg_dpd_trailing_90d,
        coalesce(billing.unpaid_ratio_trailing_90d, 0) as unpaid_ratio_trailing_90d,
        coalesce(billing.total_outstanding_debt, 0) as total_outstanding_debt,
        logins.days_since_last_login,
        logins.login_velocity,
        coalesce(tickets.billing_disputes_count, 0) as billing_disputes_count,
        tickets.average_satisfaction_score,
        insolvency.is_insolvent

    from spine
    inner join companies_scd
        on spine.company_id = companies_scd.company_id
        and spine.snapshot_date = companies_scd.snapshot_date
    left join contracts
        on spine.company_id = contracts.company_id
        and spine.snapshot_date = contracts.snapshot_date
    left join financials
        on spine.company_id = financials.company_id
        and spine.snapshot_date = financials.snapshot_date
    left join billing
        on spine.company_id = billing.company_id
        and spine.snapshot_date = billing.snapshot_date
    left join logins
        on spine.company_id = logins.company_id
        and spine.snapshot_date = logins.snapshot_date
    left join tickets
        on spine.company_id = tickets.company_id
        and spine.snapshot_date = tickets.snapshot_date
    left join insolvency
        on spine.company_id = insolvency.company_id
        and spine.snapshot_date = insolvency.snapshot_date
)

select * from final
