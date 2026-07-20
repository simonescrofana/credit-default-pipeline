with spine as (
    select * from {{ ref('int_company_date_spine') }}
),

contracts as (
    select * from {{ ref('stg_energy_contracts') }}
),

contracts_asof as (
    select
        spine.company_id,
        spine.snapshot_date,
        contracts.contract_id,
        contracts.commodity_type,
        contracts.contract_status
    from spine
    inner join contracts
        on spine.company_id = contracts.company_id
        and spine.snapshot_date >= contracts.activation_date
        and (
            spine.snapshot_date < contracts.dbt_valid_to
            or contracts.dbt_valid_to is null
        )
),

aggregated as (
    select
        company_id,
        snapshot_date,
        count(*) filter (where contract_status = 'active') as active_contracts_count,
        bool_or(
            commodity_type = 'electricity' and contract_status = 'active'
        ) as has_active_electricity_contract,
        bool_or(
            commodity_type = 'gas' and contract_status = 'active'
        ) as has_active_gas_contract
    from contracts_asof
    group by company_id, snapshot_date
)

select * from aggregated
