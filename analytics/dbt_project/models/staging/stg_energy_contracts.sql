with source as (
    select * from {{ ref('energy_contracts_snapshot') }}
),

transformed as (
    select
        id::bigint as contract_id,
        company_id::bigint as company_id,
        lower(trim(commodity_type)) as commodity_type,
        lower(trim(market_type)) as market_type,
        lower(trim(contract_status)) as contract_status,
        lower(trim(voltage_level)) as voltage_level,
        power_committed_kw::decimal(10,2) as power_committed_kw,
        lower(trim(pressure_level)) as pressure_level,
        upper(trim(gas_meter_class)) as gas_meter_class,
        activation_date::date as activation_date,
        termination_date::date as termination_date,
        dbt_valid_from,
        dbt_valid_to

    from source
)

select * from transformed
