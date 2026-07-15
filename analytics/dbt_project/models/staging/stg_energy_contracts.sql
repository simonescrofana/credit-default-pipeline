with source as (
    select * from {{ source('db_source', 'energy_contracts') }}
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
        case 
            when termination_date is not null then (termination_date::date - activation_date::date)::int
            else (current_date - activation_date::date)::int
        end as contract_duration_days

    from source
)

select * from transformed
