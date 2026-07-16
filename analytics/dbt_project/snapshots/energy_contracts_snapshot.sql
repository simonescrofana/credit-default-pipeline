{% snapshot energy_contracts_snapshot %}

{{
    config(
      target_schema='snapshots',
      unique_key='id',
      strategy='check',
      check_cols=['contract_status', 'voltage_level', 'power_committed_kw',
                  'gas_meter_class', 'termination_date'],
    )
}}

select * from {{ source('db_source', 'energy_contracts') }}

{% endsnapshot %}
