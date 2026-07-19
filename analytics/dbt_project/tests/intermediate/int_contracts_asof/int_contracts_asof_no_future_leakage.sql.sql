--- Anti-leakage test
select
    c_asof.company_id,
    c_asof.snapshot_date,
    c_asof.active_contracts_count
from {{ ref('int_contracts_asof') }} c_asof
left join {{ ref('stg_energy_contracts') }} stg
    on c_asof.company_id = stg.company_id
    and c_asof.snapshot_date >= stg.dbt_valid_from
    and (c_asof.snapshot_date < stg.dbt_valid_to or stg.dbt_valid_to is null)
    and stg.contract_status = 'active'
where c_asof.active_contracts_count > 0
  and stg.contract_id is null
