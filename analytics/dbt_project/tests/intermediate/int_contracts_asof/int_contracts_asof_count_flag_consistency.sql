select
    company_id,
    snapshot_date,
    active_contracts_count,
    has_active_electricity_contract,
    has_active_gas_contract
from {{ ref('int_contracts_asof') }}
where active_contracts_count = 0
  and (has_active_electricity_contract = true or has_active_gas_contract = true)
