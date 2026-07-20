with source as (
    select * from {{ source('db_source', 'financial_statements') }}
),

transformed as (
    select
        company_id::bigint as company_id,
        fiscal_year::int as fiscal_year,
        total_revenue::decimal(15,2) as total_revenue,
        net_income::decimal(15,2) as net_income,
        total_debt::decimal(15,2) as total_debt,
        liquidity_cash::decimal(15,2) as liquidity_cash,
        share_capital::decimal(15,2) as share_capital,
        ebitda::decimal(15,2) as ebitda,
        case 
            when share_capital > 0 then (total_debt::decimal(15,2) / share_capital::decimal(15,2))
            else null
        end as leverage_ratio,
        case 
            when total_debt > 0 then (liquidity_cash::decimal(15,2) / total_debt::decimal(15,2))
            else null
        end as cash_to_debt_ratio,
        case 
            when total_revenue > 0 then (net_income::decimal(15,2) / total_revenue::decimal(15,2))
            else null
        end as net_profit_margin

    from source
)

select * from transformed