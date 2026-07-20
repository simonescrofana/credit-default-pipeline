with spine as (
    select * from {{ ref('int_company_date_spine') }}
),

financials as (
    select * from {{ ref('stg_financial_statements') }}
),

-- A fiscal_year's statement is treated as publicly available starting
-- 6 months after the fiscal year closes (assumes calendar-year closing on
-- 31/12): fiscal_year=2023 becomes available_from 2024-06-30. This models
-- the real-world approval/filing delay for Italian companies, so a
-- snapshot_date never "sees" a balance sheet before it could plausibly
-- have been public.
financials_with_availability as (
    select
        *,
        make_date(fiscal_year + 1, 6, 30) as available_from
    from financials
),

ranked as (
    select
        spine.company_id,
        spine.snapshot_date,
        financials_with_availability.fiscal_year,
        financials_with_availability.total_revenue,
        financials_with_availability.net_income,
        financials_with_availability.total_debt,
        financials_with_availability.liquidity_cash,
        financials_with_availability.share_capital,
        financials_with_availability.ebitda,
        financials_with_availability.leverage_ratio,
        financials_with_availability.cash_to_debt_ratio,
        financials_with_availability.net_profit_margin,
        row_number() over (
            partition by spine.company_id, spine.snapshot_date
            order by financials_with_availability.fiscal_year desc
        ) as recency_rank
    from spine
    inner join financials_with_availability
        on spine.company_id = financials_with_availability.company_id
        and financials_with_availability.available_from <= spine.snapshot_date
),

latest_only as (
    select
        company_id,
        snapshot_date,
        fiscal_year,
        total_revenue,
        net_income,
        total_debt,
        liquidity_cash,
        share_capital,
        ebitda,
        leverage_ratio,
        cash_to_debt_ratio,
        net_profit_margin
    from ranked
    where recency_rank = 1
)

select * from latest_only
